"""
Runs cross-agent formalization over the frozen subsample: Route A's
model is the Specifier (its own independently-extracted atoms), Route
B's model is the Implementer (told only the required signature, reasons
from ITS OWN independently-extracted atoms). Categorizes each document
by attempts_used, since that alone distinguishes trivial agreement from
recovered disagreement from genuine non-convergence.

v3: persists inclusion/exclusion source text and full Dafny source for
every document (not just failures), and runs mutation adequacy testing
on every verified spec -- so this run never needs to be repeated for
any of the post-hoc analysis.
"""
import json
from collections import defaultdict
from typing import Optional
from datasets import load_from_disk

from load_subsample import load_subsample_doc_ids
from run_pipeline import DocumentLogger
from run_dual_route_evaluation import (
    load_schema, _client_route_a, _client_route_b,
    extract_route_a, extract_route_b, normalize_against_schema,
    _truncate_for_display,
)
from cross_agent_formalize import autoformalize_cross
from stage3_mutation_guard import run_stage3

DATASET_PATH = "dataset\\chia_local_dataset"


def categorize(atoms_a, atoms_b, result):
    if result.attempts_used == 1 and result.success:
        return "GENUINE_AGREEMENT"

    stdout_lower = (result.stdout or "").lower()
    if not result.success and ("unresolved identifier" in stdout_lower
                                or "resolution/type errors" in stdout_lower):
        return "GENERATION_FAILURE"

    # FIXED: multiset per variable, not last-write-wins single value --
    # a variable with two atoms (a range) was previously collapsing to
    # one entry, causing a missing second bound to be mis-scored as a
    # contradiction instead of a coverage gap.
    by_var_a = defaultdict(set)
    by_var_b = defaultdict(set)
    for a in atoms_a:
        by_var_a[a.var].add((a.op, a.value))
    for a in atoms_b:
        by_var_b[a.var].add((a.op, a.value))

    shared = set(by_var_a) & set(by_var_b)
    contradicts = any(
        by_var_a[v] != by_var_b[v]
        and not (by_var_a[v] <= by_var_b[v] or by_var_b[v] <= by_var_a[v])
        for v in shared
    )

    if contradicts:
        return "NON_CONVERGENT_DIRECT_CONTRADICTION"
    return "NON_CONVERGENT_COVERAGE_GAP"


def main(subsample_path: str = "high_tier_subsample_96.jsonl",
         out_path: str = None,
         limit: Optional[int] = None,
         run_tag: str = None):
    vocab_names, is_integer_map = load_schema()
    allowed_names = set(vocab_names)
    print(f"Loaded schema: {len(vocab_names)} variables.\n")

    doc_ids = load_subsample_doc_ids(subsample_path)
    if limit:
        doc_ids = set(list(doc_ids)[:limit])
    print(f"Running on {len(doc_ids)} documents.\n")

    dataset = load_from_disk(DATASET_PATH)
    by_doc = defaultdict(dict)
    for row in dataset:
        doc_id = row.get("document_id")
        if doc_id not in doc_ids:
            continue
        rid = row.get("id", "")
        if rid.endswith("_inc"):
            by_doc[doc_id]["inc"] = row
        elif rid.endswith("_exc"):
            by_doc[doc_id]["exc"] = row

    client_a, deployment_a = _client_route_b()   # nano is now Specifier
    client_b, deployment_b = _client_route_a()   # mini is now Implementer
    print(f"Specifier (Route A): {deployment_a}")
    print(f"Implementer (Route B): {deployment_b}\n")

    run_tag = run_tag or f"{deployment_a}_spec__{deployment_b}_impl"
    out_path = out_path or f"cross_agent_results_{run_tag}.jsonl"

    counts = defaultdict(int)
    mutation_counts = defaultdict(int)
    results = []
    logic = "AND"

    with open(out_path, "w", encoding="utf-8") as outf:
        for i, doc_id in enumerate(sorted(doc_ids), 1):
            print(f"[{i}/{len(doc_ids)}] {doc_id}")
            rows = by_doc.get(doc_id, {})

            # ADDED: capture raw source text unconditionally
            inc_text = ""
            exc_text = ""
            if rows.get("inc") and rows["inc"].get("passages"):
                inc_text = rows["inc"]["passages"][0]["text"][0]
            if rows.get("exc") and rows["exc"].get("passages"):
                exc_text = rows["exc"]["passages"][0]["text"][0]
            parts = []
            if inc_text:
                parts.append("INCLUSION CRITERIA:\n" + inc_text)
            if exc_text:
                parts.append("EXCLUSION CRITERIA:\n" + exc_text)
            text = "\n\n".join(parts)

            record = {
                "doc_id": doc_id,
                "specifier_model": deployment_a,
                "implementer_model": deployment_b,
                "inclusion_text": inc_text,   # ADDED
                "exclusion_text": exc_text,   # ADDED
            }

            with DocumentLogger(f"crossagent_{run_tag}_{doc_id}") as logger:
                logger.log(f"[SOURCE] INCLUSION CRITERIA:\n{inc_text}")   # ADDED
                logger.log(f"[SOURCE] EXCLUSION CRITERIA:\n{exc_text}")   # ADDED

                try:
                    raw_a = extract_route_a(text, vocab_names, client_a, deployment_a)
                    raw_b = extract_route_b(text, vocab_names, client_b, deployment_b)
                    atoms_a, oov_a = normalize_against_schema(raw_a, allowed_names, is_integer_map)
                    atoms_b, oov_b = normalize_against_schema(raw_b, allowed_names, is_integer_map)
                    record["specifier_atoms"] = [(a.var, a.op, a.value) for a in atoms_a]
                    record["implementer_own_atoms"] = [(a.var, a.op, a.value) for a in atoms_b]

                    logger.log(f"[SPECIFIER] atoms: {record['specifier_atoms']}")
                    logger.log(f"[IMPLEMENTER] own atoms: {record['implementer_own_atoms']}")

                    if not atoms_a:
                        record["category"] = "SPECIFIER_EMPTY"
                    else:
                        result = autoformalize_cross(
                            atoms_a, atoms_b, logic, client_a, deployment_a, client_b, deployment_b
                        )
                        record["success"] = result.success
                        record["attempts_used"] = result.attempts_used
                        record["category"] = categorize(atoms_a, atoms_b, result)
                        record["dafny_source"] = result.dafny_source   # ADDED: unconditional, not just on failure

                        logger.log(f"[RESULT] success={result.success} attempts={result.attempts_used} "
                                   f"-> {record['category']}")
                        logger.log(f"[DAFNY SOURCE]\n{result.dafny_source}")   # ADDED

                        if not result.success:
                            logger.log(f"[RESULT] final stdout:\n{_truncate_for_display(result.stdout, 3000)}")
                            logger.log(f"[RESULT] final stderr:\n{_truncate_for_display(result.stderr, 3000)}")
                        else:
                            # ADDED: mutation adequacy on every verified spec
                            logger.log(f"\n[MUTATION] Running Stage 3 adequacy guard...")
                            adequacy = run_stage3(result.dafny_source, atoms_a, logic, logger)
                            record["mutation_is_adequate"] = adequacy.is_adequate
                            record["mutation_failing_mutant"] = adequacy.failing_mutant
                            record["mutation_detail"] = adequacy.detail
                            mutation_counts["adequate" if adequacy.is_adequate else "inadequate"] += 1
                            logger.log(f"[MUTATION] is_adequate={adequacy.is_adequate} "
                                       f"failing_mutant={adequacy.failing_mutant!r}")

                except Exception as e:
                    err_str = str(e)
                    logger.log(f"[PIPELINE_ERROR] {_truncate_for_display(err_str, 2000)}")
                    if "DEGENERATE_REPETITION" in err_str:
                        record["category"] = "EXTRACTION_DEGENERATE_REPETITION"
                    else:
                        record["category"] = "PIPELINE_ERROR"
                    record["error"] = _truncate_for_display(err_str, 500)

            print(f"  -> {record['category']}"
                  + (f" | mutation_adequate={record.get('mutation_is_adequate')}"
                     if "mutation_is_adequate" in record else ""))
            counts[record["category"]] += 1
            results.append(record)
            outf.write(json.dumps(record) + "\n")

    total = len(results)
    print("\n" + "=" * 70)
    print(f"CROSS-AGENT VERIFICATION SUMMARY (Specifier={deployment_a}, Implementer={deployment_b})")
    print("=" * 70)
    for cat, n in sorted(counts.items()):
        print(f"  {cat:>32}: {n} ({n/total:.1%})")

    # FIXED: reference the categories categorize() actually returns
    n_semantically_evaluated = (counts["GENUINE_AGREEMENT"]
                                 + counts["NON_CONVERGENT_COVERAGE_GAP"]
                                 + counts["NON_CONVERGENT_DIRECT_CONTRADICTION"])
    print(f"\n  (excluded from rates below: {counts['GENERATION_FAILURE']} GENERATION_FAILURE, "
          f"{counts['SPECIFIER_EMPTY']} SPECIFIER_EMPTY -- non-semantic outcomes)")
    if n_semantically_evaluated:
        print(f"  Genuine agreement:        {counts['GENUINE_AGREEMENT']}/{n_semantically_evaluated} "
              f"({counts['GENUINE_AGREEMENT']/n_semantically_evaluated:.1%})")
        print(f"  Coverage-gap divergence:  {counts['NON_CONVERGENT_COVERAGE_GAP']}/{n_semantically_evaluated} "
              f"({counts['NON_CONVERGENT_COVERAGE_GAP']/n_semantically_evaluated:.1%})")
        print(f"  Direct contradiction:     {counts['NON_CONVERGENT_DIRECT_CONTRADICTION']}/{n_semantically_evaluated} "
              f"({counts['NON_CONVERGENT_DIRECT_CONTRADICTION']/n_semantically_evaluated:.1%})")

    n_converged = counts["GENUINE_AGREEMENT"]  # only category where attempts_used==1 and success
    if n_converged:
        n_adequate = mutation_counts["adequate"]
        n_inadequate = mutation_counts["inadequate"]
        print(f"\n  Mutation adequacy among {n_converged} genuinely-agreed specs:")
        print(f"    Adequate:   {n_adequate}/{n_converged} ({n_adequate/n_converged:.1%})")
        print(f"    INADEQUATE: {n_inadequate}/{n_converged} ({n_inadequate/n_converged:.1%})")

    print(f"\nFull results (incl. Dafny source, mutation data, raw text) written to {out_path}")


if __name__ == "__main__":
    import sys
    path_arg = sys.argv[1] if len(sys.argv) > 1 else "high_tier_subsample_96.jsonl"
    limit_arg = int(sys.argv[2]) if len(sys.argv) > 2 else None
    tag_arg = sys.argv[3] if len(sys.argv) > 3 else None
    main(subsample_path=path_arg, limit=limit_arg, run_tag=tag_arg)