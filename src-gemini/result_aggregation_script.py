"""
Aggregate cross_agent_results.jsonl into paper-ready Results statistics.

Expected input:
    cross_agent_results.jsonl

Run separately for each of the four model configurations.

The script reports:
  1. Category breakdown over all attempted documents
  2. Semantic-evaluation denominator and agreement/failure rates
  3. Implementer coverage of Specifier variables
  4. Specifier atom-count complexity
  5. Agreement by complexity bucket
  6. Mutation adequacy among genuine agreements
  7. Structural verification-gaming audit
  8. Detailed contradiction diagnostics
  9. Candidate qualitative cases

Important denominator rules
---------------------------
Semantic evaluation excludes:
    GENERATION_FAILURE
    PIPELINE_ERROR
    EXTRACTION_DEGENERATE_REPETITION
    SPECIFIER_EMPTY

Coverage uses the same integrity exclusions, and additionally requires
a non-empty Specifier atom set.

Complexity analysis uses the same integrity exclusions and requires
a non-empty Specifier atom set.

Thus coverage and complexity denominators are explicitly defined and
reproducible rather than depending on incidental downstream fields.
"""

import json
import re
import sys
from collections import Counter, defaultdict


# ============================================================
# CONFIGURATION
# ============================================================

INTEGRITY_FAILURES = {
    "GENERATION_FAILURE",
    "PIPELINE_ERROR",
    "EXTRACTION_DEGENERATE_REPETITION",
    "SPECIFIER_EMPTY",
}

COMPLEXITY_BUCKETS = ("1-3", "4-7", "8+")


# ============================================================
# LOADING
# ============================================================

def load_results(path):
    records = []

    with open(path, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Invalid JSON on line {line_number}: {e}"
                ) from e

    return records


# ============================================================
# DAFNY PARSING
# ============================================================

def split_dafny_source(dafny_source: str):
    """
    Extract the MatchSpec predicate and CheckEligibility method
    from the generated Dafny source for qualitative inspection.
    """

    if not dafny_source:
        return None, None

    predicate_match = re.search(
        r"(predicate\s+MatchSpec.*?\n\})",
        dafny_source,
        re.DOTALL,
    )

    method_match = re.search(
        r"(method\s+CheckEligibility.*)",
        dafny_source,
        re.DOTALL,
    )

    predicate = predicate_match.group(1) if predicate_match else None
    method = method_match.group(1) if method_match else None

    return predicate, method


# ============================================================
# VERIFICATION-GAMING AUDIT
# ============================================================

def implementer_cheated(dafny_source: str) -> bool:
    """
    Detect direct delegation from CheckEligibility to MatchSpec.

    The audit specifically looks for:
        eligible := MatchSpec(...);
    or:
        return MatchSpec(...);

    This is intentionally narrower than merely checking whether
    the string "MatchSpec" occurs anywhere in the method body.
    """

    if not dafny_source:
        return False

    method_match = re.search(
        r"method\s+CheckEligibility[^{]*\{(.*?)\}",
        dafny_source,
        re.DOTALL,
    )

    if not method_match:
        return False

    body = method_match.group(1)

    direct_assignment = re.search(
        r"\beligible\s*:=\s*MatchSpec\s*\(",
        body,
    )

    direct_return = re.search(
        r"\breturn\s+MatchSpec\s*\(",
        body,
    )

    return bool(direct_assignment or direct_return)


# ============================================================
# ATOM HELPERS
# ============================================================

def atom_variables(atoms):
    """
    Return the set of variable names from atoms represented as:
        (variable, operator, value)
    """

    return {atom[0] for atom in atoms}


def atom_constraint_map(atoms):
    """
    Convert atoms into:
        variable -> set((operator, value))
    """

    result = defaultdict(set)

    for atom in atoms:
        if len(atom) < 3:
            continue

        variable, operator, value = atom[:3]
        result[variable].add((operator, value))

    return result


def complexity_bucket(atom_count):
    if atom_count <= 3:
        return "1-3"

    if atom_count <= 7:
        return "4-7"

    return "8+"


# ============================================================
# SEMANTIC / COVERAGE FILTERS
# ============================================================

def is_semantically_evaluated(record):
    """
    True when the record belongs in the primary semantic denominator.
    """

    return record.get("category") not in INTEGRITY_FAILURES


def has_valid_specifier_atoms(record):
    """
    True when the record has at least one Specifier atom.
    """

    return bool(record.get("specifier_atoms"))


def is_valid_coverage_record(record):
    """
    Coverage denominator:
      - not an integrity failure
      - has a non-empty Specifier atom set
    """

    return (
        is_semantically_evaluated(record)
        and has_valid_specifier_atoms(record)
    )


# ============================================================
# MAIN AGGREGATION
# ============================================================

def main(path):

    records = load_results(path)

    if not records:
        raise ValueError("No records found in input file.")

    total = len(records)

    spec_models = {
        r.get("specifier_model", "?")
        for r in records
        if r.get("specifier_model")
    }

    impl_models = {
        r.get("implementer_model", "?")
        for r in records
        if r.get("implementer_model")
    }

    spec_model = next(iter(spec_models), "?")
    impl_model = next(iter(impl_models), "?")

    print("=" * 72)
    print("CROSS-AGENT RESULTS AGGREGATION")
    print("=" * 72)

    print(f"Input file:          {path}")
    print(f"Specifier model:     {spec_model}")
    print(f"Implementer model:   {impl_model}")
    print(f"Total documents attempted: {total}")

    if len(spec_models) > 1:
        print(f"WARNING: multiple Specifier models found: {sorted(spec_models)}")

    if len(impl_models) > 1:
        print(f"WARNING: multiple Implementer models found: {sorted(impl_models)}")


    # ========================================================
    # 1. CATEGORY BREAKDOWN
    # ========================================================

    counts = Counter(
        r.get("category", "MISSING_CATEGORY")
        for r in records
    )

    print("\n" + "=" * 72)
    print("1. CATEGORY BREAKDOWN")
    print("=" * 72)

    for category, n in counts.most_common():
        print(
            f"{category:>40}: "
            f"{n:>3} / {total} "
            f"({n / total:.1%})"
        )


    # ========================================================
    # 2. PRIMARY SEMANTIC DENOMINATOR
    # ========================================================

    excluded_records = [
        r for r in records
        if r.get("category") in INTEGRITY_FAILURES
    ]

    semantic_records = [
        r for r in records
        if is_semantically_evaluated(r)
    ]

    n_excluded = len(excluded_records)
    n_semantic = len(semantic_records)

    print("\n" + "=" * 72)
    print("2. PRIMARY SEMANTIC EVALUATION DENOMINATOR")
    print("=" * 72)

    print(
        f"Excluded as non-semantic/integrity failures: "
        f"{n_excluded} / {total} ({n_excluded / total:.1%})"
    )

    print(
        f"Semantically evaluated: "
        f"{n_semantic} / {total} ({n_semantic / total:.1%})"
    )

    assert total == n_excluded + n_semantic, (
        "Denominator mismatch: "
        f"total={total}, excluded={n_excluded}, semantic={n_semantic}"
    )

    semantic_categories = (
        "GENUINE_AGREEMENT",
        "CERTIFIED",
        "REPAIRED",
        "NON_CONVERGENT_COVERAGE_GAP",
        "NON_CONVERGENT_REAL_DISAGREEMENT",
        "NON_CONVERGENT_DIRECT_CONTRADICTION",
    )

    print("\nSemantic outcomes:")

    for category in semantic_categories:
        n = counts.get(category, 0)

        if n:
            print(
                f"{category:>40}: "
                f"{n} / {n_semantic} "
                f"({n / n_semantic:.1%})"
            )


    # ========================================================
    # 3. AGREEMENT
    # ========================================================

    genuine_records = [
        r for r in semantic_records
        if r.get("category") == "GENUINE_AGREEMENT"
    ]

    n_genuine = len(genuine_records)

    raw_agreement = (
        n_genuine / n_semantic
        if n_semantic
        else 0.0
    )

    print("\n" + "=" * 72)
    print("3. AGREEMENT")
    print("=" * 72)

    print(
        f"Raw genuine agreement: "
        f"{n_genuine} / {n_semantic} "
        f"({raw_agreement:.1%})"
    )

    print(
        f"Raw agreement over all attempted documents: "
        f"{n_genuine} / {total} "
        f"({n_genuine / total:.1%})"
    )


    # ========================================================
    # 4. IMPLEMENTER COVERAGE
    # ========================================================

    coverage_records = [
        r for r in records
        if is_valid_coverage_record(r)
    ]

    coverage_ratios = []

    for r in coverage_records:

        spec_vars = atom_variables(
            r.get("specifier_atoms", [])
        )

        impl_vars = atom_variables(
            r.get("implementer_own_atoms", [])
        )

        if not spec_vars:
            continue

        ratio = len(spec_vars & impl_vars) / len(spec_vars)

        coverage_ratios.append(ratio)

    print("\n" + "=" * 72)
    print("4. IMPLEMENTER COVERAGE OF SPECIFIER VARIABLES")
    print("=" * 72)

    n_coverage = len(coverage_ratios)

    print(
        f"Coverage denominator: "
        f"{n_coverage} / {total} attempted documents"
    )

    print(
        f"Coverage exclusions: "
        f"{total - n_coverage} documents"
    )

    if coverage_ratios:

        mean_coverage = sum(coverage_ratios) / n_coverage

        full_coverage = sum(
            1 for c in coverage_ratios
            if c == 1.0
        )

        zero_coverage = sum(
            1 for c in coverage_ratios
            if c == 0.0
        )

        print(
            f"Mean coverage ratio: "
            f"{mean_coverage:.1%}"
        )

        print(
            f"Full coverage (100%): "
            f"{full_coverage} / {n_coverage} "
            f"({full_coverage / n_coverage:.1%})"
        )

        print(
            f"Zero coverage (0%):   "
            f"{zero_coverage} / {n_coverage} "
            f"({zero_coverage / n_coverage:.1%})"
        )

    else:
        print("No valid coverage records.")


    # ========================================================
    # 5. SPECIFIER COMPLEXITY
    # ========================================================

    complexity_records = [
        r for r in records
        if is_valid_coverage_record(r)
    ]

    atom_counts = [
        len(r.get("specifier_atoms", []))
        for r in complexity_records
        if r.get("specifier_atoms")
    ]

    print("\n" + "=" * 72)
    print("5. SPECIFIER COMPLEXITY")
    print("=" * 72)

    if atom_counts:

        mean_atoms = sum(atom_counts) / len(atom_counts)

        print(
            f"Complexity denominator: "
            f"{len(atom_counts)} documents"
        )

        print(f"Mean Specifier atom count: {mean_atoms:.1f}")
        print(f"Minimum: {min(atom_counts)}")
        print(f"Maximum: {max(atom_counts)}")

    else:
        print("No valid Specifier atom data.")


    # ========================================================
    # 6. OUTCOME BY COMPLEXITY BUCKET
    # ========================================================

    by_complexity = defaultdict(Counter)

    for r in complexity_records:

        n_atoms = len(r.get("specifier_atoms", []))

        if n_atoms == 0:
            continue

        bucket = complexity_bucket(n_atoms)

        by_complexity[bucket][
            r.get("category", "MISSING_CATEGORY")
        ] += 1

    print("\n" + "=" * 72)
    print("6. OUTCOME BY SPECIFIER ATOM-COUNT BUCKET")
    print("=" * 72)

    for bucket in COMPLEXITY_BUCKETS:

        counter = by_complexity[bucket]
        bucket_total = sum(counter.values())

        if bucket_total == 0:
            print(f"{bucket:>4} atoms: no records")
            continue

        n_agreement = counter.get(
            "GENUINE_AGREEMENT",
            0
        )

        print(
            f"\n{bucket:>4} atoms "
            f"({bucket_total} documents)"
        )

        print(
            f"  Agreement: "
            f"{n_agreement}/{bucket_total} "
            f"({n_agreement / bucket_total:.1%})"
        )

        for category, n in counter.most_common():
            print(
                f"  {category:>38}: "
                f"{n}/{bucket_total} "
                f"({n / bucket_total:.1%})"
            )


    # ========================================================
    # 7. MUTATION ADEQUACY
    # ========================================================

    mutation_records = [
        r for r in genuine_records
        if "mutation_is_adequate" in r
    ]

    print("\n" + "=" * 72)
    print("7. MUTATION ADEQUACY")
    print("=" * 72)

    print(
        f"Mutation denominator: "
        f"{len(mutation_records)} "
        f"genuinely agreed specifications"
    )

    if mutation_records:

        n_adequate = sum(
            1
            for r in mutation_records
            if r.get("mutation_is_adequate") is True
        )

        n_inadequate = (
            len(mutation_records) - n_adequate
        )

        print(
            f"Adequate:   "
            f"{n_adequate}/{len(mutation_records)} "
            f"({n_adequate / len(mutation_records):.1%})"
        )

        print(
            f"INADEQUATE: "
            f"{n_inadequate}/{len(mutation_records)} "
            f"({n_inadequate / len(mutation_records):.1%})"
        )

        if n_inadequate:

            print("\nInadequate documents:")

            for r in mutation_records:

                if not r.get("mutation_is_adequate"):

                    print(
                        f"  {r.get('doc_id')}: "
                        f"failing_mutant="
                        f"{r.get('mutation_failing_mutant')!r} "
                        f"detail="
                        f"{r.get('mutation_detail')!r}"
                    )

    else:
        print("No mutation data present for genuine agreements.")


    # ========================================================
    # 8. VERIFICATION-GAMING AUDIT
    # ========================================================

    cheat_docs = [
        r.get("doc_id")
        for r in genuine_records
        if implementer_cheated(
            r.get("dafny_source", "")
        )
    ]

    n_cheats = len(cheat_docs)

    corrected_agreement_count = (
        n_genuine - n_cheats
    )

    corrected_agreement = (
        corrected_agreement_count / n_semantic
        if n_semantic
        else 0.0
    )

    print("\n" + "=" * 72)
    print("8. VERIFICATION-GAMING AUDIT")
    print("=" * 72)

    print(
        f"Audited genuine-agreement cases: "
        f"{n_genuine}"
    )

    print(
        f"Direct MatchSpec invocation cases: "
        f"{n_cheats}/{n_genuine}"
    )

    if cheat_docs:
        print(f"Gaming documents: {cheat_docs}")

    print(
        f"Corrected genuine agreement: "
        f"{corrected_agreement_count}/{n_semantic} "
        f"({corrected_agreement:.1%})"
    )


    # ========================================================
    # 9. CONTRADICTION DIAGNOSTICS
    # ========================================================

    contradiction_records = [
        r for r in semantic_records
        if r.get("category")
        == "NON_CONVERGENT_DIRECT_CONTRADICTION"
    ]

    print("\n" + "=" * 72)
    print("9. CONTRADICTION DIAGNOSTICS")
    print("=" * 72)

    print(
        f"Direct contradiction records: "
        f"{len(contradiction_records)}"
    )

    for r in contradiction_records:

        a_map = atom_constraint_map(
            r.get("specifier_atoms", [])
        )

        b_map = atom_constraint_map(
            r.get("implementer_own_atoms", [])
        )

        shared = set(a_map) & set(b_map)

        matching = [
            v for v in shared
            if a_map[v] == b_map[v]
        ]

        contradictions = [
            v
            for v in shared
            if a_map[v] != b_map[v]
            and not (
                a_map[v] <= b_map[v]
                or b_map[v] <= a_map[v]
            )
        ]

        if not contradictions:
            continue

        print(
            f"\n--- {r.get('doc_id')} ---"
        )

        print(
            f"Shared variables:       {len(shared)}"
        )

        print(
            f"Exactly matching:       {len(matching)}"
        )

        print(
            f"Contradictory variables: {len(contradictions)}"
        )

        print(
            f"Contradiction variables: {contradictions}"
        )

        print(
            f"Specifier atoms: "
            f"{r.get('specifier_atoms')}"
        )

        print(
            f"Implementer atoms: "
            f"{r.get('implementer_own_atoms')}"
        )


    # ========================================================
    # 10. QUALITATIVE CONTRADICTION CASES
    # ========================================================

    print("\n" + "=" * 72)
    print("10. QUALITATIVE CONTRADICTION CASES WITH DAFNY")
    print("=" * 72)

    for r in contradiction_records:

        a_map = atom_constraint_map(
            r.get("specifier_atoms", [])
        )

        b_map = atom_constraint_map(
            r.get("implementer_own_atoms", [])
        )

        shared = set(a_map) & set(b_map)

        contradictions = [
            v
            for v in shared
            if a_map[v] != b_map[v]
            and not (
                a_map[v] <= b_map[v]
                or b_map[v] <= a_map[v]
            )
        ]

        if not contradictions:
            continue

        print(
            f"\n--- {r.get('doc_id')} "
            f"(contradiction on {contradictions}) ---"
        )

        print(
            f"Specifier atoms:   "
            f"{r.get('specifier_atoms')}"
        )

        print(
            f"Implementer atoms: "
            f"{r.get('implementer_own_atoms')}"
        )

        predicate, method = split_dafny_source(
            r.get("dafny_source", "")
        )

        if predicate:
            print(
                f"\n--- Specifier's Dafny predicate "
                f"({spec_model}) ---"
            )
            print(predicate)

        if method:
            print(
                f"\n--- Implementer's Dafny method "
                f"({impl_model}) ---"
            )
            print(method)


    # ========================================================
    # 11. FINAL PAPER-READY SUMMARY
    # ========================================================

    print("\n" + "=" * 72)
    print("11. PAPER-READY SUMMARY")
    print("=" * 72)

    print(
        f"Attempted documents: {total}"
    )

    print(
        f"Semantic denominator: {n_semantic}"
    )

    print(
        f"Genuine agreement: "
        f"{n_genuine}/{n_semantic} "
        f"({raw_agreement:.1%})"
    )

    print(
        f"Agreement over attempted: "
        f"{n_genuine}/{total} "
        f"({n_genuine / total:.1%})"
    )

    print(
        f"Corrected agreement: "
        f"{corrected_agreement_count}/{n_semantic} "
        f"({corrected_agreement:.1%})"
    )

    if coverage_ratios:

        print(
            f"Mean Implementer coverage: "
            f"{sum(coverage_ratios) / len(coverage_ratios):.1%} "
            f"(N={len(coverage_ratios)})"
        )

    print(
        f"Verification-gaming cases: "
        f"{n_cheats}"
    )

    if mutation_records:

        n_adequate = sum(
            1
            for r in mutation_records
            if r.get("mutation_is_adequate") is True
        )

        print(
            f"Mutation adequacy: "
            f"{n_adequate}/{len(mutation_records)} "
            f"({n_adequate / len(mutation_records):.1%})"
        )

    print("\nAggregation completed successfully.")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    path_arg = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "cross_agent_results.jsonl"
    )

    main(path_arg)