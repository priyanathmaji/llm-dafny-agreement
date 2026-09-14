"""
Single source of truth for the fixed clinical taxonomy.

Both Stage 1 (what the LLM is told it may extract) and gold_extraction.py
(what CHIA gold predicates get aligned to) must agree on the exact same
variable names and types, or you get silent mismatches that look like
extraction failures but are actually just bookkeeping drift between two
hand-maintained lists. This module is the one place that list is defined.

Merged from two sources: the original 10-13 variable taxonomy (validated
across every test in this project) plus a corpus-frequency-driven
expansion (via a discovery script scanning CHIA for common numeric
predicate phrases). Nothing from the original set was dropped -- doing
so would silently break every existing regression test that references
these variables by name.
"""

# var_name -> {is_integer, keywords, anchor_keyword (optional)}
# keywords: substrings (whole-word matched) that identify a gold CHIA
# entity's text as belonging to this variable. Keep entries specific
# enough to avoid false alignment (see gold_extraction.py's docstring
# for the "ast" vs "at least" collision this caused when too generic).
#
# anchor_keyword (optional): for variables named after a SPECIFIC
# drug/procedure/scoring-system (not a generic clinical concept), the
# source text must literally contain this term before Stage 1's
# extracted value is accepted. Without this, real batch runs showed the
# LLM confidently fabricating values for narrowly-named variables on
# documents that never discuss that specific thing at all.
#
# IMPORTANT: never bundle two distinct clinical scales under one
# variable (e.g. Child-Pugh and Gleason are both "scores" but are
# unrelated, differently-scaled measurements) -- this defeats the
# anchor-keyword defense entirely, since both would legitimately match
# a generic anchor like "score". Give each scale its own variable and
# its own specific anchor.
TAXONOMY = {
    # --- Original validated set --------------------------------------
    "age": {
        "is_integer": True, 
        "keywords": ["age", "aged", "years", "gestational age", "gestational"],
        "anchor_keyword": "age"
    },
    "diarrhea_grade_limit": {"is_integer": True, "keywords": ["diarrhea"], "anchor_keyword": "diarrhea"},
    "max_total_bilirubin_uln": {"is_integer": False, "keywords": ["bilirubin", "total bilirubin"], "anchor_keyword": "bilirubin"},
    "max_ast_alt_uln": {"is_integer": False, "keywords": ["ast", "sgot"], "anchor_keyword": "ast"},
    "max_alt_uln": {"is_integer": False, "keywords": ["alt", "sgpt"], "anchor_keyword": "alt"},
    "min_life_expectancy_months": {"is_integer": True, "keywords": ["life expectancy"], "anchor_keyword": "expectancy"},
    "min_karnofsky_status": {"is_integer": True, "keywords": ["karnofsky"], "anchor_keyword": "karnofsky"},
    "max_warfarin_mg": {"is_integer": True, "keywords": ["warfarin"], "anchor_keyword": "warfarin"},
    "chemoembolization_exclusion_months": {"is_integer": True, "keywords": ["chemoembolization"], "anchor_keyword": "chemoembolization"},
    "prior_malignancy_exclusion_years": {"is_integer": True, "keywords": ["malignancy"], "anchor_keyword": "malignancy"},
    "sandostatin_stabilization_days": {"is_integer": True, "keywords": ["sandostatin"], "anchor_keyword": "sandostatin"},
    "min_platelet_count": {"is_integer": True, "keywords": ["platelet count", "platelets", "plt"], "anchor_keyword": "platelet"},
    "max_serum_creatinine": {"is_integer": False, "keywords": ["serum creatinine", "creatinine"], "anchor_keyword": "creatinine"},

    "min_iq_score": {
        "is_integer": True, 
        "keywords": ["iq", "intelligence quotient", "iq score"], 
        "anchor_keyword": "iq"
    },
    "max_plasma_hiv_rna_copies_ml": {
        "is_integer": True, 
        "keywords": ["plasma hiv rna", "hiv rna", "viral load", "copies/ml", "copies / ml"], 
        "anchor_keyword": "rna"
    },

    # --- Corpus-frequency expansion, deduped against the above, units
    #     made explicit, clinical-score bundling split apart -----------
    "bmi": {"is_integer": False, "keywords": ["bmi", "body mass index"], "anchor_keyword": "bmi"},
    # NOTE: deliberately omitted -- source had a bare "weight" variable
    # with no unit. "weight >= 50" is dangerously ambiguous (kg vs lbs
    # is a very different clinical threshold). Decide the unit and add
    # back as weight_kg / weight_lbs before using.
    "min_hemoglobin_g_dl": {"is_integer": False, "keywords": ["hemoglobin", "hgb"], "anchor_keyword": "hemoglobin"},
    "min_wbc_count": {"is_integer": True, "keywords": ["wbc", "white blood cell count", "leukocytes"], "anchor_keyword": "wbc"},
    "min_creatinine_clearance_ml_min": {"is_integer": False, "keywords": ["creatinine clearance", "crcl"], "anchor_keyword": "clearance"},
    "min_egfr_ml_min": {"is_integer": False, "keywords": ["egfr", "estimated glomerular filtration rate"], "anchor_keyword": "egfr"},
    "max_systolic_blood_pressure": {"is_integer": True, "keywords": ["systolic blood pressure", "sbp"], "anchor_keyword": "systolic"},
    "min_left_ventricular_ejection_fraction": {"is_integer": False, "keywords": ["left ventricular ejection fraction", "lvef"], "anchor_keyword": "ejection"},
    "min_fvc_percent": {"is_integer": False, "keywords": ["fvc", "forced vital capacity"], "anchor_keyword": "fvc"},
    "max_body_temperature": {"is_integer": False, "keywords": ["temperature", "temp"], "anchor_keyword": "temperature"},
    "max_inr": {"is_integer": False, "keywords": ["inr", "international normalized ratio"], "anchor_keyword": "inr"},

    # Split apart from the source's bundled "max_clinical_score" -- two
    # unrelated scoring systems on different scales must not share a
    # variable, since a generic "score" anchor can't tell them apart.
    "max_child_pugh_score": {"is_integer": True, "keywords": ["child-pugh", "child pugh"], "anchor_keyword": "child-pugh"},
    "max_gleason_score": {"is_integer": True, "keywords": ["gleason"], "anchor_keyword": "gleason"},
}

TAXONOMY_INT_VARS = {var for var, meta in TAXONOMY.items() if meta["is_integer"]}
TAXONOMY_ALIGNMENT = {var: meta["keywords"] for var, meta in TAXONOMY.items()}
TAXONOMY_VAR_NAMES = list(TAXONOMY.keys())
TAXONOMY_ANCHOR_KEYWORDS = {var: meta["anchor_keyword"] for var, meta in TAXONOMY.items()
                             if "anchor_keyword" in meta}