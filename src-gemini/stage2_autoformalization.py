"""
Stage 2: Multi-Agent Adversarial Autoformalization and Loop Verification.
Splits code generation into two isolated agents:
1. Agent A (Specifier): Synthesizes the declarative logic (predicate).
2. Agent B (Implementer): Synthesizes the operational logic (method).
Uses Dafny as a symbolic arbiter to identify structural and logical disagreements.
"""






import os
from pathlib import Path
from dotenv import load_dotenv
import re
import subprocess
import tempfile
from dataclasses import dataclass
from typing import Optional
from openai import AzureOpenAI


# .env lives at the project root (one level above this file's directory,
# e.g. clamp/.env vs. clamp/code/stage2_autoformalization.py) -- adjust
# the parent count here if you relocate this file.
env_path = Path(__file__).resolve().parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

# Max internal consensus loops
K_MAX_ATTEMPTS = 15
DAFNY_SOLVER_PATH = os.environ.get(
    "DAFNY_SOLVER_PATH", r"C:\Users\admin\.conda\envs\cs7643-a3-nlp\bin\z3.EXE"
)

# =====================================================================
# MULTI-AGENT ADVERSARIAL PROMPTS
# =====================================================================

SPECIFIER_SYSTEM_PROMPT = """You are Agent A (The Specifier). 
Your sole task is to generate a formal declarative specification in Dafny based on clinical criteria atoms.
You must output ONLY a valid Dafny `predicate` named `MatchSpec`.

Rules:
- Use `int` for integer-valued atoms, `real` for continuous-valued atoms.
- Translate boundaries accurately (e.g. >=, <=, >, <, ==).
- Do NOT generate a method body, ensures clauses, or implementation logic.
- Return ONLY raw Dafny source code for the predicate block. No code fences, no markdown, no commentary.
"""

IMPLEMENTER_SYSTEM_PROMPT = """You are Agent B (The Implementer).
Your task is to generate the operational implementation for clinical eligibility matching.
You must output ONLY a valid Dafny `method` named `CheckEligibility`.

Rules:
- The method signature must match the parameter names and exact types provided.
- The method must return a boolean named `eligible`.
- Crucially, your method MUST include an `ensures` clause: `ensures eligible == MatchSpec(...)`
- Compute the boolean output step-by-step or dynamically using assignment statements (`:=`).
- Return ONLY raw Dafny source code for the method block. No code fences, no markdown, no commentary.
"""

REPAIR_SYSTEM_PROMPT = """You are the Symbolic Repair Agent.
You are fixing a disagreement caught by the Dafny compiler between Agent A's specification and Agent B's implementation.
You will be given the full merged Dafny code and the exact counterexample trace showing where they conflict.

Fix the program so it verifies. Do NOT change the `predicate MatchSpec` (Agent A's spec is ground truth). 
Fix the `method CheckEligibility` body or signature so it aligns perfectly with the spec.
Return ONLY the corrected raw Dafny source code. No code fences, no commentary.
"""

@dataclass
class VerificationResult:
    success: bool
    dafny_source: str
    stdout: str
    stderr: str
    attempts_used: int

# Keep your existing utility methods intact (_client, _strip_code_fences, _atoms_to_prompt, run_dafny_verify, _check_type_fidelity)
def _client() -> AzureOpenAI:
    return AzureOpenAI(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT_2"),
        api_key=os.getenv("AZURE_AI_FOUNDRY_API_KEY_2"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION_2"),
    )


def _strip_code_fences(text: str) -> str:
    text = text.strip()
    text = re.sub(r'^```[a-zA-Z]*\n', '', text)
    text = re.sub(r'\n```$', '', text)
    return text.strip()


def _atoms_to_prompt(atoms: list, logic: str) -> str:
    lines = [f"  {a.var}: {a.op} {a.value}  [Dafny type: {'int' if a.is_integer else 'real'}]"
             for a in atoms]
    return (
        f"Atoms:\n" + "\n".join(lines) +
        f"\n\nCombine with logic: {logic}\n"
        f"Predicate name: MatchSpec\n"
        f"Method name: CheckEligibility\n"
        f"IMPORTANT: use the exact Dafny type listed in brackets for each "
        f"parameter -- do NOT default everything to `real`. Using the wrong "
        f"type changes the logical meaning of the predicate at integer/real "
        f"boundary values.\n"
        f"Method should take each atom's variable as a parameter and return `eligible: bool`."
    )

def run_dafny_verify(dafny_source: str) -> tuple[bool, str, str]:
    """Writes source to a temp .dfy file and shells out to `dafny verify`."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".dfy", delete=False) as f:
        f.write(dafny_source)
        path = f.name
    try:
        proc = subprocess.run(
            ["dafny", "verify", "--solver-path", DAFNY_SOLVER_PATH, path],
            capture_output=True, text=True, timeout=60
        )
        success = proc.returncode == 0 and "0 errors" in proc.stdout
        return success, proc.stdout, proc.stderr
    except FileNotFoundError:
        raise RuntimeError(
            "Dafny CLI not found on PATH. Install Dafny and ensure "
            "`dafny --version` succeeds before running Stage 2."
        )
    finally:
        os.unlink(path)


def _check_type_fidelity(dafny_source: str, atoms: list) -> Optional[str]:
    """
    Confirms every atom's Dafny-side type matches its taxonomy-declared
    is_integer flag. Returns an error message (for repair feedback) if
    a mismatch is found, or None if all types are correct.

    This exists because Dafny will happily verify a `real`-typed
    predicate against a `real`-typed implementation even when the
    underlying clinical variable is integer-valued -- the mismatch is
    logically silent at Stage 2/3 and only surfaces as an inconsistency
    between what Dafny verified and what Stage 4's Python-side Z3 check
    (which correctly uses Int() per the taxonomy) actually compares.
    """
    mismatches = []
    for a in atoms:
        expected_type = "int" if a.is_integer else "real"
        pattern = re.compile(rf'\b{re.escape(a.var)}\s*:\s*(int|real)\b')
        m = pattern.search(dafny_source)
        if m and m.group(1) != expected_type:
            mismatches.append(f"{a.var} generated as `{m.group(1)}`, taxonomy requires `{expected_type}`")
    if mismatches:
        return "Type fidelity violation(s): " + "; ".join(mismatches)
    return None

def autoformalize(atoms: list, logic: str, client: Optional[AzureOpenAI] = None, deployment: Optional[str] = None) -> VerificationResult:
    """
    Runs the Multi-Agent Adversarial generate -> merge -> verify -> repair loop.
    """
    client = client or _client()
    deployment = deployment or os.getenv("AZURE_OPENAI_RESPONSES_DEPLOYMENT_NAME_2")
    user_prompt = _atoms_to_prompt(atoms, logic)
    
    dafny_source = None
    last_stdout, last_stderr = "", ""
    
    for attempt in range(1, K_MAX_ATTEMPTS + 1):
        if dafny_source is None:
            # ────────────── PASS 1: ISOLATED MULTI-AGENT GENERATION ──────────────
            # Call Agent A: The Specifier
            spec_resp = client.chat.completions.create(
                model=deployment, 
                temperature=0.2,
                messages=[{"role": "system", "content": SPECIFIER_SYSTEM_PROMPT}, 
                          {"role": "user", "content": user_prompt}]
            )
            spec_code = _strip_code_fences(spec_resp.choices[0].message.content)
            
            # Call Agent B: The Implementer
            impl_resp = client.chat.completions.create(
                model=deployment, 
                temperature=0.2,
                messages=[{"role": "system", "content": IMPLEMENTER_SYSTEM_PROMPT}, 
                          {"role": "user", "content": user_prompt}]
            )
            impl_code = _strip_code_fences(impl_resp.choices[0].message.content)
            
            # Interleave their outputs into a single cohesive file
            dafny_source = f"{spec_code}\n\n{impl_code}"
        else:
            # ────────────── PASS 2+: ADVERSARIAL REPAIR LOOP ──────────────
            repair_prompt = (
                f"Previous Joint Dafny source:\n{dafny_source}\n\n"
                f"Compiler/Arbitration mismatch output:\n{last_stdout}\n{last_stderr}\n\n"
                f"Identify why implementation disagrees with specification and repair the method."
            )
            resp = client.chat.completions.create(
                model=deployment, temperature=0.2,
                messages=[{"role": "system", "content": REPAIR_SYSTEM_PROMPT}, {"role": "user", "content": repair_prompt}]
            )
            dafny_source = _strip_code_fences(resp.choices[0].message.content)

        # ────────────── SYMBOLIC ARBITRATION ──────────────
        success, last_stdout, last_stderr = run_dafny_verify(dafny_source)
        
        if success:
            type_error = _check_type_fidelity(dafny_source, atoms)
            if type_error:
                # Force loop to keep going if types mismatch taxonomy standards
                last_stdout, last_stderr = "", type_error
                continue
            return VerificationResult(True, dafny_source, last_stdout, last_stderr, attempt)
            
    return VerificationResult(False, dafny_source, last_stdout, last_stderr, K_MAX_ATTEMPTS)
