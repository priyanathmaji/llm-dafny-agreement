"""
Stage 3: Specification Satisfiability and Mutation Adequacy Guards.
Fully integrated with the unified DocumentLogger tracking framework.
"""
import re
from dataclasses import dataclass
from z3 import Real, Int, Solver, sat, And, Or
from stage2_autoformalization import run_dafny_verify

METHOD_SIG_RE = re.compile(
    r'method\s+(\w+)\s*\(([^)]*)\)\s*returns\s*\(\s*eligible\s*:\s*bool\s*\)', 
    re.MULTILINE
)

@dataclass
class AdequacyResult:
    is_satisfiable: bool
    is_adequate: bool
    failing_mutant: str = ""
    detail: str = ""

def _parse_signature(dafny_source: str):
    m = METHOD_SIG_RE.search(dafny_source)
    if not m:
        raise ValueError("Could not locate canonical method signature in Dafny source.")
    method_name, params = m.group(1), m.group(2)
    return method_name, params

def _build_mutant_source(dafny_source: str, method_name: str, params: str, mutant_body: str) -> str:
    pattern = re.compile(
        r'(method\s+' + re.escape(method_name) + r'\s*\([^)]*\)\s*returns\s*\(\s*eligible\s*:\s*bool\s*\)'
        r'(?:\s*ensures[^\{]*)?)\{[^}]*\}', re.DOTALL
    )
    replacement = r'\1{\n' + mutant_body + '\n}'
    mutated, n = pattern.subn(replacement, dafny_source, count=1)
    if n == 0:
        raise ValueError("Failed to splice mutant body into Dafny source.")
    return mutated

def check_satisfiability(atoms: list, logic: str, logger) -> bool:
    """SAT(S): builds a Z3 model of the spec directly from the atom list."""
    logger.log("[STAGE 3] 🔍 Initiating SAT(S) Check via Z3...")
    z3_vars = {}
    for a in atoms:
        z3_vars[a.var] = Int(a.var) if a.is_integer else Real(a.var)

    def atom_expr(a):
        v = z3_vars[a.var]
        val = a.value
        return {
            ">=": v >= val, "<=": v <= val, ">": v > val, "<": v < val, "==": v == val,
        }[a.op]

    exprs = [atom_expr(a) for a in atoms]
    formula = And(*exprs) if "OR" not in logic.upper() else Or(*exprs)
    
    s = Solver()
    s.add(formula)
    status = s.check()
    
    is_sat = status == sat
    logger.log(f"[STAGE 3]  └─ Z3 Formula Satisfiability: {status} (is_sat={is_sat})")
    return is_sat

def check_anti_cheating_guard(dafny_source: str, method_name: str, logger) -> bool:
    """RESEARCH GUARD: Ensures Agent B didn't cheat by invoking the predicate."""
    logger.log("[STAGE 3] 🛡️ Running Adversarial Anti-Cheating Structural Guard...")
    
    pattern = re.compile(r'method\s+' + re.escape(method_name) + r'[^\{]*\{([^}]*)\}', re.DOTALL)
    match = pattern.search(dafny_source)
    if match:
        body_content = match.group(1).strip()
        logger.log(f"[STAGE 3]  ├─ Extracted Method Body Content:\n---\n{body_content}\n---")
        
        if "MatchSpec" in body_content:
            logger.log("[STAGE 3]  ❌ ANTI-CHEATING VIOLATION DETECTED: Agent B invoked 'MatchSpec' inside body.")
            return False
            
    logger.log("[STAGE 3]  ✅ Anti-Cheating Guard Passed. Agent B constructed independent operational code.")
    return True

def run_mutation_suite(dafny_source: str, atoms: list, logger) -> AdequacyResult:
    method_name, params = _parse_signature(dafny_source)
    
    if not check_anti_cheating_guard(dafny_source, method_name, logger):
        return AdequacyResult(
            is_satisfiable=True, 
            is_adequate=False, 
            failing_mutant="ANTI_CHEATING_GUARD_TRIGGERED",
            detail="Agent B trivially copied Agent A's predicate name instead of writing operational logic."
        )

    logger.log("[STAGE 3] 🚀 Constructing Mutation Suite...")
    mutants = {}
    mutants["tautological"] = "eligible := true;"
    mutants["contradictory"] = "eligible := false;"

    inverse_op = {">=": "<", "<=": ">", ">": "<=", "<": ">=", "==": "!="}
    for a in atoms:
        flipped = inverse_op.get(a.op, a.op)
        mutants[f"inverse_{a.var}"] = f"eligible := {a.var} {flipped} {a.value};"

        eps = 1 if a.is_integer else (round(abs(a.value) * 0.01, 4) or 0.1)
        shifted = a.value + eps if a.op in (">", ">=") else a.value - eps
        mutants[f"boundary_{a.var}"] = f"eligible := {a.var} {a.op} {shifted};"

    logger.log(f"[STAGE 3]  ├─ Generated {len(mutants)} distinct mutants for testing.")
    
    for mutant_name, mutant_body in mutants.items():
        logger.log(f"[STAGE 3]  ├─ Testing mutant '{mutant_name}'...")
        try:
            mutant_source = _build_mutant_source(dafny_source, method_name, params, mutant_body)
        except ValueError as e:
            logger.log(f"[STAGE 3]  │  ❌ Construction Failed: {e}")
            return AdequacyResult(True, False, mutant_name, f"mutant construction failed: {e}")

        verified, stdout, stderr = run_dafny_verify(mutant_source)
        
        if verified:
            logger.log(f"[STAGE 3]  │  ❌ MUTANT ESCAPED! '{mutant_name}' verified successfully against the spec.")
            return AdequacyResult(True, False, mutant_name, 
                                  f"mutant '{mutant_name}' passed verification against S -- spec is under-constrained")
        else:
            logger.log(f"[STAGE 3]  │  ✅ Mutant Caught. Dafny successfully rejected implementation variance.")

    logger.log("[STAGE 3] ✅ MUTATION ANALYSIS COMPLETE: Specification rejected 100% of broken implementations.")
    return AdequacyResult(True, True, "", "spec rejected all mutants")

def run_stage3(dafny_source: str, atoms: list, logic: str, logger) -> AdequacyResult:
    logger.log(f"\n{'='*20} ENTERING STAGE 3 EVALUATION UNIT {'='*20}")
    
    if not check_satisfiability(atoms, logic, logger):
        logger.log("[STAGE 3] ❌ PIPELINE BLOCKED: Specification is mathematically unsatisfiable.")
        return AdequacyResult(False, False, "", "spec is unsatisfiable (over-constrained)")
        
    res = run_mutation_suite(dafny_source, atoms, logger)
    logger.log(f"{'='*20} EXITING STAGE 3 EVALUATION UNIT {'='*20}\n")
    return res
