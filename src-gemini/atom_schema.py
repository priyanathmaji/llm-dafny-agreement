"""
Stage 1 -> Stage 2 bridge: normalizes raw LLM extraction output into a
structured atom schema (var, op, value) that Stage 2/3/4 can consume as
Z3 expressions, and flags anything that doesn't parse cleanly as
Out-of-Vocabulary (OOV) per the methodology's coverage policy.

This module intentionally does NOT try to guess at the semantics of
free-text fields like "unless treated and/or stable for six months" --
those are exactly the compound/negated clauses your taxonomy audit
found sit outside the fixed predicate vocabulary. They are recorded
as OOV, not silently dropped or guessed at.
"""
import re
from dataclasses import dataclass, field
from typing import Optional


# Matches things like ">= 18", "<1.5", "> 60", "<= 1", and now also
# tolerates a trailing unit descriptor with no parens, e.g. "<1.5 X ULN",
# ">= 6 months" -- real Stage 1 output legitimately varies on whether
# units are included, and the parser shouldn't silently OOV a value just
# because the LLM was more descriptive on one run than another. Compound/
# conditional clauses are still excluded upstream by the conditional-
# marker check (parens, "if", "unless", etc.) before this regex even runs.
INEQUALITY_RE = re.compile(r'^\s*(>=|<=|>|<|==)\s*([\d.]+)\s*(?:[a-zA-Z%][a-zA-Z0-9%/\s]*)?\s*$')

# Fixed taxonomy type declarations, centralized in taxonomy.py so
# Stage 1's LLM prompt and gold_extraction.py's alignment table can
# never silently drift apart from each other.
from taxonomy import TAXONOMY_INT_VARS, TAXONOMY_VAR_NAMES, TAXONOMY_ANCHOR_KEYWORDS


@dataclass
class Atom:
    var: str
    op: Optional[str] = None
    value: Optional[float] = None
    is_integer: bool = False
    raw_text: str = ""
    is_oov: bool = False
    oov_reason: str = ""


@dataclass
class NormalizedSchema:
    atoms: list = field(default_factory=list)
    oov_atoms: list = field(default_factory=list)

    def clean_atoms(self):
        return [a for a in self.atoms if not a.is_oov]

    def coverage(self):
        total = len(self.atoms)
        clean = len(self.clean_atoms())
        return clean / total if total else 0.0


def normalize_stage1_output(raw_dict: dict, source_text: str = None) -> NormalizedSchema:
    """
    Takes the flat {variable_name: constraint_string} dict produced by
    Stage 1 and converts each entry into an Atom. Simple inequality
    constraints (">= 18", "<1.5") parse cleanly. Anything containing
    compound logic, conditionals, or free text ("unless...", "except...",
    "(one month if...)") is flagged OOV rather than guessed at.

    source_text: if provided, enables the anchor-keyword hallucination
    check -- variables tied to a specific drug/procedure name (see
    taxonomy.TAXONOMY_ANCHOR_KEYWORDS) are rejected unless that term
    literally appears in the source text. Without this, real batch
    runs showed the LLM confidently fabricating drug-specific values
    (e.g. "sandostatin_stabilization_days") on documents that never
    mention that drug at all, by pattern-matching onto a structurally
    similar but unrelated criterion. These fabrications were otherwise
    invisible to evaluation, since gold has no comparable data to
    catch them against.
    """
    schema = NormalizedSchema()

    for var, value in raw_dict.items():
        # Reject any key not in the fixed taxonomy outright -- Stage 1
        # is instructed to use ONLY the given variable names, but
        # nothing previously enforced this, so an invented key (seen
        # in practice: "max_ast_alt_uln_liver_metastases") could
        # silently pass through as a "clean" atom if its value merely
        # happened to parse.
        if var not in TAXONOMY_VAR_NAMES:
            atom = Atom(var=var, raw_text=str(value), is_oov=True,
                        oov_reason=f"'{var}' is not in the fixed taxonomy -- "
                                    f"LLM invented a variable name outside the allowed list")
            schema.atoms.append(atom)
            schema.oov_atoms.append(atom)
            continue

        if value is None:
            atom = Atom(var=var, raw_text="null", is_oov=True,
                        oov_reason="value is null (LLM indicated this variable was not discussed "
                                    "in the passage) -- not a parse failure, no taxonomy extension needed")
            schema.atoms.append(atom)
            schema.oov_atoms.append(atom)
            continue

        value_str = str(value).strip()

        # Range check FIRST: Stage 1 sometimes emits a compound range
        # like ">= 18 and <= 55" for a single variable -- this must be
        # detected before the generic " and " conditional-marker
        # rejection below, or every range silently OOVs even though
        # it's perfectly parseable as two atoms sharing the same var.
        range_m = re.match(
            r'^\s*(>=|<=|>|<|==)\s*([\d.]+)\s+and\s+(>=|<=|>|<|==)\s*([\d.]+)\s*$',
            value_str, re.IGNORECASE
        )
        if range_m:
            op1, val1, op2, val2 = range_m.groups()
            is_int = var in TAXONOMY_INT_VARS
            for op, val in [(op1, float(val1)), (op2, float(val2))]:
                schema.atoms.append(Atom(var=var, op=op, value=val, is_integer=is_int,
                                          raw_text=f"{value_str} (range)"))
            continue

        # Reject anything that smells like compound/conditional logic --
        # these need real Stage-1 taxonomy extensions, not regex guessing.
        conditional_markers = ["unless", "except", "if ", "and/or", " and ", "(", "or "]
        if any(m in value_str.lower() for m in conditional_markers):
            atom = Atom(var=var, raw_text=value_str, is_oov=True,
                        oov_reason="compound/conditional clause outside fixed predicate grammar")
            schema.atoms.append(atom)
            schema.oov_atoms.append(atom)
            continue

        m = INEQUALITY_RE.match(value_str)
        if m:
            op, val = m.group(1), float(m.group(2))

            # Anchor-keyword check: does the source text actually
            # mention the specific drug/procedure this variable is
            # named after?
            if source_text is not None and var in TAXONOMY_ANCHOR_KEYWORDS:
                anchor = TAXONOMY_ANCHOR_KEYWORDS[var]
                if anchor.lower() not in source_text.lower():
                    atom = Atom(var=var, op=op, value=val, raw_text=value_str, is_oov=True,
                                oov_reason=f"HALLUCINATION SUSPECTED: '{anchor}' does not appear "
                                            f"anywhere in the source text, but Stage 1 produced a "
                                            f"value for '{var}' anyway")
                    schema.atoms.append(atom)
                    schema.oov_atoms.append(atom)
                    continue

            is_int = var in TAXONOMY_INT_VARS
            schema.atoms.append(Atom(var=var, op=op, value=val, is_integer=is_int, raw_text=value_str))
        else:
            atom = Atom(var=var, raw_text=value_str, is_oov=True,
                        oov_reason="did not match canonical inequality grammar")
            schema.atoms.append(atom)
            schema.oov_atoms.append(atom)

    return schema


if __name__ == "__main__":
    # Sanity check against your actual Stage 1 output
    stage1_output = {
        "brain_metastases_allowed": "unless treated and/or stable for at least six months prior to study start",
        "chemoembolization_exclusion_months": "6 (1 if other sites of measurable disease)",
        "diarrhea_grade_limit": "<= 1",
        "max_warfarin_mg": "1",
        "prior_malignancy_exclusion_years": "5",
        "max_total_bilirubin_uln": "<1.5",
        "max_ast_alt_uln": "<2.5 (<5 if liver metastases present)",
        "sandostatin_stabilization_days": "30",
        "min_life_expectancy_months": ">3",
        "min_karnofsky_status": ">60",
    }

    schema = normalize_stage1_output(stage1_output)
    print(f"Coverage: {schema.coverage():.1%} ({len(schema.clean_atoms())}/{len(schema.atoms)} atoms parsed cleanly)\n")
    print("--- Clean atoms (proceed to Stage 2) ---")
    for a in schema.clean_atoms():
        print(f"  {a.var} {a.op} {a.value}")
    print("\n--- OOV atoms (excluded, reported separately) ---")
    for a in schema.oov_atoms:
        print(f"  {a.var}: \"{a.raw_text}\"  [{a.oov_reason}]")