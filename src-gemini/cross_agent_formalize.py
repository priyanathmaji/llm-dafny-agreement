"""
Cross-agent formalization: Specifier and Implementer are fed by TWO
INDEPENDENTLY-EXTRACTED atom sets from two independent models, not one
shared list. Dafny's verifier is the actual arbiter -- if the
Implementer's independent reading missed something the Specifier's
independent reading required, `ensures eligible == MatchSpec(...)`
genuinely fails to verify, and the repair loop gets a real chance to
either fix it or genuinely fail to.
"""
from stage2_autoformalization import (
    run_dafny_verify, _check_type_fidelity, _strip_code_fences, VerificationResult,
)

K_MAX_ATTEMPTS = 1

SPECIFIER_SYSTEM_PROMPT = """You are Agent A (The Specifier).
Generate a formal declarative specification in Dafny based on the given atoms.
Output ONLY a valid Dafny `predicate` named `MatchSpec`.
Use `int` for integer-valued atoms, `real` for continuous-valued atoms.
Translate boundaries accurately (>=, <=, >, <, ==).
Return ONLY raw Dafny source for the predicate block. No commentary, no code fences.
"""

IMPLEMENTER_CROSS_SYSTEM_PROMPT = """You are Agent B (The Implementer), reasoning INDEPENDENTLY of Agent A.

You must produce a Dafny `method` named `CheckEligibility` with EXACTLY this
parameter signature (required so it type-checks against a specification you
have not seen):
{signature}

You do NOT know what thresholds Agent A's specification uses. Instead, use
YOUR OWN independent reading of the criteria below to decide what logic each
parameter should satisfy:
{own_atoms}

Rules:
- The method must return a boolean named `eligible`.
- Include `ensures eligible == MatchSpec({param_names})`.
- For any parameter in the required signature that your OWN reading above
  gives you no information about, do not fabricate a constraint for it --
  simply do not restrict on it in your logic (this is an honest reflection
  of what you independently found, not an error).
- Compute the result via assignment statements.
Return ONLY raw Dafny source for the method block. No commentary, no code fences.
"""

REPAIR_SYSTEM_PROMPT = """You are the Symbolic Repair Agent, fixing Agent B's implementation
so it matches Agent A's specification, based on the Dafny counterexample shown.
Do NOT change `predicate MatchSpec` -- it is ground truth.
If the counterexample reveals your implementation is missing a constraint
entirely (a parameter you had no independent information about), add the
correct constraint using the value shown in the counterexample IF it lets you
generalize a sensible rule; if the counterexample gives no generalizable
signal, say so plainly by returning the UNCHANGED method with a comment
explaining why it cannot be fixed from the given information, rather than
guessing.
Return ONLY raw Dafny source. No commentary, no code fences.
"""


def _atoms_to_spec_prompt(atoms, logic):
    lines = [f"  {a.var}: {a.op} {a.value}  [Dafny type: {'int' if a.is_integer else 'real'}]" for a in atoms]
    return (f"Atoms:\n" + "\n".join(lines) + f"\n\nCombine with logic: {logic}\n"
            f"Predicate name: MatchSpec\n")


def _param_signature(atoms):
    """One entry per UNIQUE variable name -- an atom list can contain
    two atoms sharing a var name for a range (age >= 18, age <= 75),
    which must become ONE parameter, not two, or Dafny rejects the
    method for a duplicate parameter name (and everything after the
    duplicate in the argument list shifts by one position, producing
    a cascade of unrelated-looking type-mismatch errors)."""
    seen = {}
    for a in atoms:
        seen[a.var] = "int" if a.is_integer else "real"  # last-seen type wins; ranges share one type anyway
    return list(seen.items())


def _atoms_to_own_reading_prompt(atoms_b):
    if not atoms_b:
        return "(Your own reading found no numeric criteria at all.)"
    lines = [f"  {a.var}: {a.op} {a.value}  [Dafny type: {'int' if a.is_integer else 'real'}]" for a in atoms_b]
    return "Atoms from YOUR OWN independent extraction:\n" + "\n".join(lines)


def autoformalize_cross(atoms_a, atoms_b, logic,
                         client_a, deployment_a, client_b, deployment_b):
    signature = _param_signature(atoms_a)
    sig_str = ", ".join(f"{name}: {typ}" for name, typ in signature)
    param_names = ", ".join(name for name, _ in signature)

    spec_resp = client_a.chat.completions.create(
        model=deployment_a, 
        #temperature=0.2,
        messages=[{"role": "system", "content": SPECIFIER_SYSTEM_PROMPT},
                  {"role": "user", "content": _atoms_to_spec_prompt(atoms_a, logic)}]
    )
    spec_code = _strip_code_fences(spec_resp.choices[0].message.content)
    # spec_code is now FROZEN -- never regenerated, never overwritten,
    # never sent to the repair step to "return alongside" the repair. It
    # is concatenated fresh into dafny_source on every single attempt.

    impl_prompt = IMPLEMENTER_CROSS_SYSTEM_PROMPT.format(
        signature=sig_str, own_atoms=_atoms_to_own_reading_prompt(atoms_b), param_names=param_names
    )
    impl_resp = client_b.chat.completions.create(
        model=deployment_b, temperature=0.2,
        messages=[{"role": "system", "content": impl_prompt},
                  {"role": "user", "content": "Write the CheckEligibility method now."}]
    )
    impl_code = _strip_code_fences(impl_resp.choices[0].message.content)

    last_stdout, last_stderr = "", ""
    for attempt in range(1, K_MAX_ATTEMPTS + 1):
        dafny_source = f"{spec_code}\n\n{impl_code}"  # reassembled EVERY attempt from frozen spec_code

        success, stdout, stderr = run_dafny_verify(dafny_source)
        if success:
            type_error = _check_type_fidelity(dafny_source, atoms_a)
            if type_error:
                success, stderr = False, type_error
            else:
                return VerificationResult(True, dafny_source, stdout, stderr, attempt)

        last_stdout, last_stderr = stdout, stderr
        if attempt == K_MAX_ATTEMPTS:
            break

        # Repair prompt now only ever shows/asks for the METHOD, never the
        # full file -- structurally impossible for it to drop the predicate,
        # since the predicate is never part of what it's asked to return.
        repair_prompt = (
            f"Specification (for reference ONLY, do not repeat it in your answer):\n{spec_code}\n\n"
            f"Your previous CheckEligibility method:\n{impl_code}\n\n"
            f"Counterexample / compiler output when checked against the specification above:\n"
            f"{last_stdout}\n{last_stderr}\n\n"
            f"Return ONLY the corrected `method CheckEligibility` block. Do NOT return the "
            f"predicate. Do NOT return anything except the method."
        )
        resp = client_b.chat.completions.create(
            model=deployment_b, temperature=0.2,
            messages=[{"role": "system", "content": REPAIR_SYSTEM_PROMPT},
                      {"role": "user", "content": repair_prompt}]
        )
        impl_code = _strip_code_fences(resp.choices[0].message.content)  # ONLY impl_code is replaced

    dafny_source = f"{spec_code}\n\n{impl_code}"
    return VerificationResult(False, dafny_source, last_stdout, last_stderr, K_MAX_ATTEMPTS)