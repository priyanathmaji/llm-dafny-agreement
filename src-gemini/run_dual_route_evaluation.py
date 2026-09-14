"""
Canonical extraction + normalization module for the dual-route /
cross-agent comparison work. This is the single source of truth for:
  - loading the shared vocabulary+type schema
  - Route A (literal) and Route B (interpretive) extraction prompts
  - turning raw LLM output into SimpleAtom objects, with OOV/malformed
    values reported explicitly rather than silently dropped

Everything else (run_dual_route_evaluation.py's old inline duplicates,
cross_agent_formalize.py, run_cross_agent_evaluation.py) should import
FROM this file, not redefine any of this.
"""
import json
import os
import re
from dataclasses import dataclass
from typing import Optional
from collections import Counter
from stage1_llm_extraction import _client as _base_client


# =====================================================================
# SCHEMA LOADING
# =====================================================================
SCHEMA_PATH = "final_unified_schema.json"


def load_schema(path: str = SCHEMA_PATH):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    names = data["complete_vocabulary_keys"]
    is_integer = data.get("variable_is_integer", {})
    return names, is_integer


def _format_vocab(names):
    return "\n".join(f"  - {v}" for v in sorted(names))


# =====================================================================
# SEPARATE CLIENTS -- Route A / Route B never share a client object
# =====================================================================
def _client_route_a():
    from openai import AzureOpenAI
    return AzureOpenAI(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT_2"),
        api_key=os.getenv("AZURE_AI_FOUNDRY_API_KEY_2"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION_2"),
    ), os.getenv("AZURE_OPENAI_RESPONSES_DEPLOYMENT_NAME_2")


def _client_route_b():
    from openai import AzureOpenAI
    return AzureOpenAI(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT_3"),
        api_key=os.getenv("AZURE_AI_FOUNDRY_API_KEY_3"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION_3"),
    ), os.getenv("AZURE_OPENAI_RESPONSES_DEPLOYMENT_NAME_3")


# =====================================================================
# PROMPTS -- shared format contract + per-route reasoning style
# =====================================================================
_SHARED_FORMAT_RULES = """
### FORMAT CONTRACT (applies regardless of your reasoning style below):
- EVERY value you emit MUST start with an explicit comparison operator
  (>=, <=, >, <, ==). Even for implicit-direction cases (e.g. "stable
  dose for 30 days" or "exclusion window of 3 months"), you MUST infer
  and emit the operator -- never emit a bare number or a units phrase
  with no operator. This is boundary transcription, not interpretation.
- If a concept in the passage matches the MEANING of one of the approved
  variable names below, you MUST use that exact approved name, even if
  your own preferred phrasing would differ. Do not invent a new
  descriptive name for a concept that already has an approved name in
  the list below.
- Only emit a variable if you can reduce it to a SINGLE numeric
  threshold (or "A and B" for a two-sided range). Do NOT emit purely
  categorical/qualitative criteria (e.g. "informed consent required",
  "pregnant: yes/no" with no numeric bound) -- skip these entirely.
- Do NOT paste descriptive English as a value. The value must be a
  clean comparison expression, e.g. "<= 3", ">= 40 and <= 70" --
  never a restated sentence fragment.
"""

ROUTE_A_SYSTEM_PROMPT_TEMPLATE = """You are a precision clinical data extraction engine performing a STRICT, LITERAL reading.

Extract numeric threshold criteria using ONLY these pre-approved variable names:
{vocab}
""" + _SHARED_FORMAT_RULES + """
Reasoning style -- LITERAL:
- Use the boundary EXACTLY as written. Preserve strict vs inclusive wording
  precisely ('greater than' = '>', never '>='; 'at least' = '>=').
- Do not infer a threshold the text doesn't state, beyond the minimal
  operator-direction inference required by the format contract above.
- For EXCLUSION CRITERIA, extract the eligibility-framed NEGATION of the
  literal condition.

Return ONLY raw JSON mapping variable name to constraint string. No markdown, no commentary.
"""

ROUTE_B_SYSTEM_PROMPT_TEMPLATE = """You are a clinical trial coordinator reasoning about what a patient
must actually satisfy to be eligible, based on the protocol's intent.

You may use these approved variable names when they capture a criterion you identify:
{vocab}
""" + _SHARED_FORMAT_RULES + """
Reasoning style -- INTERPRETIVE:
- Reason about clinical INTENT. If a criterion implies a threshold even when
  phrased descriptively, infer the specific numeric bound the intent requires.
- For EXCLUSION CRITERIA, reason through to the eligibility-framed requirement
  a patient must meet to NOT be excluded.
- You do not need to find a match for every approved name -- only the ones
  actually relevant here -- but you MUST use the approved name whenever the
  concept matches one, per the format contract above.

Return ONLY raw JSON mapping variable name to constraint string. No markdown, no commentary.
"""


def _strip_fences(raw: str) -> str:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()
    return raw

def _detect_degenerate_repetition(raw_content: str, threshold: int = 5) -> bool:
    """Checks whether the same 'key': value pair appears an excessive
    number of times -- a signature of the model getting stuck in a
    repetition loop rather than genuinely repeating a real duplicate key
    (which JSON wouldn't have anyway, since dict keys are unique)."""
    pairs = re.findall(r'"[^"]+"\s*:\s*"[^"]*"', raw_content)
    if not pairs:
        return False
    counts = Counter(pairs)
    most_common_pair, count = counts.most_common(1)[0]
    return count >= threshold


def _call_extraction(text: str, system_prompt: str, client, deployment, temperature: float, route_label: str) -> dict:
    resp = client.chat.completions.create(
        model=deployment,
        #temperature=temperature,
        #max_tokens=4000,  # fail faster/cheaper on runaway generation instead of burning the full context
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Isolate clinical values from this text passage:\n{text}"},
        ],
    )
    raw_content = resp.choices[0].message.content
    if not raw_content or not raw_content.strip():
        raise ValueError(f"[{route_label}] Empty response from model "
                          f"(finish_reason={resp.choices[0].finish_reason!r}).")

    if _detect_degenerate_repetition(raw_content):
        raise ValueError(f"[{route_label}] DEGENERATE_REPETITION: model got stuck repeating "
                          f"the same key/value pair. Truncated sample:\n"
                          f"{_truncate_for_display(raw_content, 500)}")

    try:
        return json.loads(_strip_fences(raw_content))
    except json.JSONDecodeError as e:
        raise ValueError(f"[{route_label}] Model returned non-JSON content. Truncated sample:\n"
                          f"{_truncate_for_display(raw_content, 2000)}")


def extract_route_a(text: str, vocab_names: list, client, deployment) -> dict:
    prompt = ROUTE_A_SYSTEM_PROMPT_TEMPLATE.format(vocab=_format_vocab(vocab_names))
    return _call_extraction(text, prompt, client, deployment, temperature=0.2, route_label="Route A")


def extract_route_b(text: str, vocab_names: list, client, deployment) -> dict:
    prompt = ROUTE_B_SYSTEM_PROMPT_TEMPLATE.format(vocab=_format_vocab(vocab_names))
    return _call_extraction(text, prompt, client, deployment, temperature=0.2, route_label="Route B")


# =====================================================================
# NORMALIZATION -- the ONLY version. Returns (atoms, oov_report) always.
# Enforces vocabulary membership STRICTLY against whatever `allowed_names`
# the caller passes in (i.e. exactly what's in final_unified_schema.json,
# nothing implied or assumed beyond that).
# =====================================================================
INEQUALITY_RE = re.compile(r'^\s*(>=|<=|>|<|==)\s*([\d,]+\.?\d*)\s*(?:[a-zA-Z%][a-zA-Z0-9%/\s]*)?\s*$')
RANGE_AND_RE = re.compile(
    r'^\s*(>=|<=|>|<|==)\s*([\d,]+\.?\d*)\s+and\s+(>=|<=|>|<|==)\s*([\d,]+\.?\d*)\s*$',
    re.IGNORECASE
)


@dataclass
class SimpleAtom:
    var: str
    op: Optional[str] = None
    value: Optional[float] = None
    is_integer: bool = False


def _to_float(numeric_str: str) -> float:
    return float(numeric_str.replace(",", ""))

def _truncate_for_display(text: str, max_len: int = 2000) -> str:
    if len(text) <= max_len:
        return text
    return text[:max_len] + f"\n... [TRUNCATED, {len(text)} total chars] ..."


def normalize_against_schema(raw_dict: dict, allowed_names: set, is_integer_map: dict):
    """
    THE canonical normalizer. Always returns (atoms, oov_report) -- a
    2-tuple, never a bare list. allowed_names is checked EXACTLY as
    passed in; if a name from final_unified_schema.json is rejected
    here, that means it genuinely is not in the `allowed_names` set the
    caller built from that file -- print/inspect `allowed_names` at the
    call site if that's ever in doubt, rather than assuming this
    function's logic is at fault.
    """
    atoms = []
    oov_report = []

    for var, value in raw_dict.items():
        if var not in allowed_names:
            oov_report.append({"var": var, "raw_value": str(value), "reason": "not in shared vocabulary"})
            continue
        if value is None:
            oov_report.append({"var": var, "raw_value": "null", "reason": "null value"})
            continue

        value_str = str(value).strip()

        range_m = RANGE_AND_RE.match(value_str)
        if range_m:
            op1, val1, op2, val2 = range_m.groups()
            is_int = is_integer_map.get(var, False)
            atoms.append(SimpleAtom(var=var, op=op1, value=_to_float(val1), is_integer=is_int))
            atoms.append(SimpleAtom(var=var, op=op2, value=_to_float(val2), is_integer=is_int))
            continue

        m = INEQUALITY_RE.match(value_str)
        if m:
            op, val = m.group(1), _to_float(m.group(2))
            atoms.append(SimpleAtom(var=var, op=op, value=val, is_integer=is_integer_map.get(var, False)))
        else:
            reason = ("compound/disjunctive clause not supported by this normalizer"
                      if " or " in value_str.lower()
                      else "did not match canonical inequality grammar (e.g. missing operator)")
            oov_report.append({"var": var, "raw_value": value_str, "reason": reason})

    return atoms, oov_report


if __name__ == "__main__":
    # Quick sanity check: confirms this file's own vocabulary enforcement
    # against final_unified_schema.json, standalone, no other script involved.
    names, types = load_schema()
    print(f"Loaded {len(names)} names from {SCHEMA_PATH}")
    test = {"age": ">= 18", "asa_physical_status_min": ">= 1", "min_gcs_score": ">= 15"}
    atoms, oov = normalize_against_schema(test, set(names), types)
    print("Atoms:", [(a.var, a.op, a.value) for a in atoms])
    print("OOV:", oov)