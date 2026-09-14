r"""
Stage 1: raw clinical text -> flat {variable: constraint_string} dict,
via Azure OpenAI. Generalized from the original one-off script into a
reusable function, with the allowed variable list pulled from
taxonomy.py instead of being hand-typed separately (see that module's
docstring for why keeping these in one place matters).
"""
import os
from pathlib import Path
from dotenv import load_dotenv
from openai import AzureOpenAI

from taxonomy import TAXONOMY_VAR_NAMES

env_path = Path(__file__).resolve().parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

SYSTEM_INSTRUCTION_TEMPLATE = (
    "You are a precision clinical data extraction engine.\n"
    "Your task is to parse unstructured text and isolate the values for specific parameters.\n"
    "You are restricted to evaluating ONLY these variables: {taxonomy}\n\n"
    "CRITICAL SYNTAX REQUIREMENTS:\n"
    "- Carefully evaluate negative operators (e.g., 'unless', 'except', 'no major impairment') "
    "to map the absolute condition scope accurately.\n"
    "- EVERY value MUST start with an explicit comparison operator (>=, <=, >, <, or ==), even "
    "when the source text states a bare number. E.g. if the text says 'stable dose for 30 days', "
    "extract this as '>= 30', not '30' -- infer the implied direction from context (a stabilization "
    "period is a minimum; an exclusion window is typically a maximum). If truly ambiguous, prefer "
    "'<=' for exclusion-style limits (mg, years, months as an upper bound) and '>=' for "
    "requirement-style minimums (days stabilized, life expectancy).\n"
    "- Preserve STRICT vs INCLUSIVE boundary wording exactly as written. 'greater than 3 months' "
    "means '> 3', NOT '>= 3' -- these are different eligibility thresholds and the distinction "
    "matters. Only use >=/<= when the source text says 'at least', 'or more', 'or equal to', "
    "'no more than', etc.\n"
    "- The passage below is divided into INCLUSION CRITERIA and EXCLUSION CRITERIA sections. "
    "For EXCLUSION CRITERIA, the text describes what DISQUALIFIES a patient -- you must extract "
    "the ELIGIBILITY-framed NEGATION of the literal condition, not the literal condition itself. "
    "E.g. if EXCLUSION CRITERIA says 'BMI less than 20', a patient is excluded when BMI < 20, so "
    "eligibility requires BMI >= 20 -- extract 'bmi': '>= 20', NOT '< 20'. For INCLUSION CRITERIA, "
    "extract the literal condition as written, no negation needed.\n"
    "- If a variable is mentioned with an inequality (e.g. >=, <=, >, <), include the symbol and number.\n"
    "- Handle nested conditionals carefully (e.g., secondary limits if liver metastases are present).\n"
    "- Return your output as a clean JSON dictionary mapping variable names to their extracted constraints.\n"
    "- Only include a key if that variable is actually discussed in the passage.\n"
    "- Return ONLY raw JSON text inside a valid JSON block. Do not add markdown backticks or conversational text."
)


def _client() -> AzureOpenAI:
    return AzureOpenAI(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT_2"),
        api_key=os.getenv("AZURE_AI_FOUNDRY_API_KEY_2"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION_2"),
    )


def extract_stage1(raw_text_passage: str, client=None, deployment=None) -> dict:
    """
    Runs the Stage 1 LLM extraction against a raw clinical text passage.
    Returns the parsed flat dict, or raises if the LLM response isn't
    valid JSON (caller should treat that as an Extraction Failure).
    """
    import json

    client = client or _client()
    deployment = deployment or os.getenv("AZURE_OPENAI_RESPONSES_DEPLOYMENT_NAME_2")

    system_instruction = SYSTEM_INSTRUCTION_TEMPLATE.format(taxonomy=TAXONOMY_VAR_NAMES)
    user_input = f"Isolate clinical values from this text passage:\n{raw_text_passage}"

    response = client.chat.completions.create(
        model=deployment,
        temperature=0.2,
        messages=[
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": user_input},
        ],
    )

    raw_output = response.choices[0].message.content.strip()
    # Defensive: strip markdown fences if the model adds them despite instructions
    if raw_output.startswith("```"):
        raw_output = raw_output.split("```")[1]
        if raw_output.startswith("json"):
            raw_output = raw_output[4:]
        raw_output = raw_output.strip()

    return json.loads(raw_output)


if __name__ == "__main__":
    print("Taxonomy variables Stage 1 will be told about:")
    for v in TAXONOMY_VAR_NAMES:
        print(f"  {v}")