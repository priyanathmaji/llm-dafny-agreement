"""
Stage 4: SMT-Based Logical Equivalence Evaluation.
Fully integrated with the unified tracking framework to expose raw Z3 formulas.
Includes adaptive scaling defenses to eliminate human annotation unit drops.
"""
from dataclasses import dataclass
from z3 import Real, Int, Solver, sat, unsat, And, Or

@dataclass
class EquivalenceResult:
    is_match: bool
    is_unknown: bool
    detail: str = ""

def _build_z3_formula(atoms: list, logic: str, var_pool: dict):
    def var_for(a):
        if a.var not in var_pool:
            var_pool[a.var] = Int(a.var) if a.is_integer else Real(a.var)
        return var_pool[a.var]

    def atom_expr(a):
        v = var_for(a)

        ### DELETE THIS BLOCK: The following was a shortcut fix that caused a false positive in the pipeline ###
        # # ─── THE DEFINITIVE CELL-COUNT SCALE PATCH ───
        # # Normalize structural unit differences (e.g., 100 vs 100000 for platelets)
        # # when a base token scale enters the solver without its multiplier node.
        # target_value = a.value
        # if any(cell_kw in a.var for cell_kw in ["platelet", "wbc"]) and target_value <= 1000.0:
        #     target_value *= 1000.0
        # # ─────────────────────────────────────────────
        
        # return {
        #     ">=": v >= target_value,
        #     "<=": v <= target_value,
        #     ">": v > target_value,
        #     "<": v < target_value,
        #     "==": v == target_value,
        # }[a.op]

        return {
            ">=": v >= a.value,
            "<=": v <= a.value,
            ">": v > a.value,
            "<": v < a.value,
            "==": v == a.value,
        }[a.op]

    if not atoms:
        return True

    exprs = [atom_expr(a) for a in atoms]
    return Or(*exprs) if "OR" in logic.upper() and "AND" not in logic.upper() else And(*exprs)

def check_logical_match(ours_atoms: list, ours_logic: str, gold_atoms: list, gold_logic: str, logger,
                         label_ours: str = "Our Internal Logic Model",
                         label_theirs: str = "Ground Truth Gold Model") -> EquivalenceResult:
    """
    LogicalMatch <=> SAT(Phi_ours XOR Phi_gold) == unsat
    Dumps comprehensive solver constraints directly into the log track.

    label_ours / label_theirs let callers outside the original CHIA-gold
    use case (e.g. comparing two independently-formalized specs against
    each other, with no gold involved at all) print accurate labels
    instead of inherited "gold" language that no longer applies.
    """
    logger.log(f"\n[STAGE 4] 🧠 Initializing SMT Logical Equivalence Verification...")

    var_pool = {}
    phi_ours = _build_z3_formula(ours_atoms, ours_logic, var_pool)
    phi_gold = _build_z3_formula(gold_atoms, gold_logic, var_pool)

    logger.log(f"[STAGE 4]  ├─ SMT Variable Declarations: {list(var_pool.keys())}")
    logger.log(f"[STAGE 4]  ├─ {label_ours}:\n      {phi_ours}")
    logger.log(f"[STAGE 4]  ├─ {label_theirs}:\n      {phi_gold}")

    import z3
    xor = Or(And(phi_ours, z3.Not(phi_gold)), And(z3.Not(phi_ours), phi_gold))

    s = Solver()
    s.add(xor)

    logger.log(f"[STAGE 4]  ├─ Submitting Adversarial XOR Assertion Layer to Solver...")
    result = s.check()

    if result == unsat:
        logger.log(f"[STAGE 4]  ✅ VERIFICATION SUCCESS: Math models are logically equivalent across all assignments.")
        return EquivalenceResult(True, False, f"{label_ours} and {label_theirs} proven equivalent")
    elif result == sat:
        model = s.model()
        logger.log(f"[STAGE 4]  ❌ LOGICAL BREAK DETECTED: Counterexample trace generated.")
        logger.log(f"[STAGE 4]     └─ Divergent Assignment Vector: {model}")
        return EquivalenceResult(False, False, f"divergent under assignment: {model}")
    else:
        logger.log(f"[STAGE 4]  ❓ SOLVER EXCEPTION: SMT boundary calculation returned Unknown.")
        return EquivalenceResult(False, True, "solver returned unknown")