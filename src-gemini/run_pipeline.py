"""
End-to-end orchestrator implementing an interleaved Verifier-Guided 
and Mutation-Guided Multi-Agent Cooperative Refinement Loop (Phase I).
Generates isolated, comprehensive artifact logs containing exact text inputs,
Stage 1 raw JSON input/outputs, and ground truth baselines.
"""
import os
import sys
from dataclasses import dataclass
from typing import Optional

from atom_schema import normalize_stage1_output, NormalizedSchema
from stage2_autoformalization import autoformalize, VerificationResult, _client, _strip_code_fences, REPAIR_SYSTEM_PROMPT
from stage3_mutation_guard import run_stage3, AdequacyResult
from stage4_equivalence import check_logical_match, EquivalenceResult

class FailureCategory:
    OOV_INCOMPLETE = "OOV_INCOMPLETE"
    EXTRACTION_FAILURE = "EXTRACTION_FAILURE"
    NON_CONVERGENCE_TYPE_A = "NON_CONVERGENCE_TYPE_A"
    NON_CONVERGENCE_TYPE_B = "NON_CONVERGENCE_TYPE_B"
    LOGICAL_DIVERGENCE = "LOGICAL_DIVERGENCE"
    CERTIFIED_MATCH = "CERTIFIED_MATCH"

@dataclass
class PipelineResult:
    category: str
    schema: Optional[NormalizedSchema] = None
    stage2: Optional[VerificationResult] = None
    stage3: Optional[AdequacyResult] = None
    stage4: Optional[EquivalenceResult] = None
    detail: str = ""

class DocumentLogger:
    """Manages dual logging to console and a document-specific log file."""
    def __init__(self, doc_id: str):
        self.doc_id = doc_id
        self.log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
        os.makedirs(self.log_dir, exist_ok=True)
        self.log_file_path = os.path.join(self.log_dir, f"{doc_id}.log")
        self.file = None

    def __enter__(self):
        self.file = open(self.log_file_path, "w", encoding="utf-8")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.file:
            self.file.close()

    def log(self, message: str, print_to_console: bool = True):
        """Writes to the artifact log file and optionally pipes to terminal."""
        if print_to_console:
            print(message)
        if self.file:
            self.file.write(message + "\n")
            self.file.flush()

def run_phase_1(stage1_raw_output: dict, logger: DocumentLogger) -> tuple:
    """Interleaved Multi-Agent Generation and Mutation Loop."""
    logger.log(f"\n{'#'*30} STARTING PHASE I: UNLEAKED AUTOMATED VERIFICATION {'#'*30}")
    schema = normalize_stage1_output(stage1_raw_output)
    clean_atoms = schema.clean_atoms()
    
    if not clean_atoms:
        logger.log("[ERROR] Phase I Aborted: No clean atoms parsed from Stage 1.")
        return PipelineResult(FailureCategory.OOV_INCOMPLETE, schema=schema, detail="No atoms parsed cleanly"), None

    logic = "AND"
    max_total_attempts = 15
    current_dafny_source = None
    last_stage2_result = None
    last_stage3_result = None
    
    client = _client()
    deployment = os.getenv("AZURE_OPENAI_RESPONSES_DEPLOYMENT_NAME_2")

    logger.log(f"[PIPELINE] Interleaved Repair Loop Initialized (Max Attempts: {max_total_attempts})")
    
    for attempt in range(1, max_total_attempts + 1):
        logger.log(f"\n[PIPELINE] ─── CRITERIA INTERLEAVED LOOP: ATTEMPT {attempt}/{max_total_attempts} ───")
        
        if current_dafny_source is None:
            logger.log("[STAGE 2] Triggering Initial Stage 2 Multi-Agent Code Generation...")
            stage2_result = autoformalize(clean_atoms, logic, client=client, deployment=deployment)
        else:
            logger.log("[STAGE 2] Repair Phase: Routing Mutation/Adequacy failure data back to LLM...")
            repair_prompt = (
                f"Previous Joint Dafny source:\n{current_dafny_source}\n\n"
                f"Mutation Guard Adequacy Mismatch:\n{last_stage3_result.detail if last_stage3_result else ''}\n"
                f"Failing condition indicator: {last_stage3_result.failing_mutant if last_stage3_result else ''}\n\n"
                f"CRITICAL REPAIR INSTRUCTION: Fix the implementation body. Do NOT invoke 'MatchSpec' inside "
                f"the body of the method, and ensure the rules match the specification strictly."
            )
            resp = client.chat.completions.create(
                model=deployment, temperature=0.2,
                messages=[{"role": "system", "content": REPAIR_SYSTEM_PROMPT}, {"role": "user", "content": repair_prompt}]
            )
            updated_code = _strip_code_fences(resp.choices.message.content)
            
            from stage2_autoformalization import run_dafny_verify, _check_type_fidelity
            success, stdout, stderr = run_dafny_verify(updated_code)
            type_err = _check_type_fidelity(updated_code, clean_atoms) if success else None
            if type_err:
                success, stderr = False, type_err
                
            stage2_result = VerificationResult(success, updated_code, stdout, stderr, attempt)

        last_stage2_result = stage2_result
        current_dafny_source = stage2_result.dafny_source

        logger.log(f"\n[RAW ARTIFACT] --- BEGIN RAW LLM GENERATED DAFNY SOURCE (ATTEMPT {attempt}) ---", print_to_console=False)
        logger.log(current_dafny_source, print_to_console=False)
        logger.log("[RAW ARTIFACT] --- END RAW LLM GENERATED DAFNY SOURCE ---\n", print_to_console=False)

        if not stage2_result.success:
            logger.log(f"[STAGE 2] ❌ Compiler Error: {stage2_result.stderr.strip()}")
            if attempt == max_total_attempts:
                return PipelineResult(FailureCategory.NON_CONVERGENCE_TYPE_A, schema=schema, stage2=stage2_result, detail="Dafny compilation failed"), None
            continue

        logger.log("[STAGE 2] ✅ Stage 2 Compiled and Formally Verified Successfully.")

        stage3_result = run_stage3(current_dafny_source, clean_atoms, logic, logger)
        last_stage3_result = stage3_result

        if not stage3_result.is_adequate:
            logger.log(f"[STAGE 3] ❌ Guard Triggered: {stage3_result.failing_mutant} - {stage3_result.detail}")
            if attempt == max_total_attempts:
                return PipelineResult(FailureCategory.NON_CONVERGENCE_TYPE_B, schema=schema, stage2=stage2_result, stage3=stage3_result, detail=stage3_result.detail), None
            continue

        logger.log(f"\n[PIPELINE] 🎉 PHASE I CERTIFICATION SUCCESSFUL AT ATTEMPT {attempt} 🎉")
        return None, (schema, clean_atoms, logic, stage2_result, stage3_result)

def run_phase_2(certified_state, gold_atoms: list, gold_logic: str, logger: DocumentLogger) -> PipelineResult:
    """Post-hoc SMT execution against external CHIA baseline."""
    logger.log(f"\n{'='*20} ENTERING PHASE II: OFFLINE CROSS-LANGUAGE SMT EVALUATION {'='*20}")
    schema, clean_atoms, logic, stage2_result, stage3_result = certified_state
    
    stage4_result = check_logical_match(clean_atoms, logic, gold_atoms, gold_logic, logger)
    category = FailureCategory.CERTIFIED_MATCH if stage4_result.is_match else FailureCategory.LOGICAL_DIVERGENCE
    
    logger.log(f"[PIPELINE] Phase II Evaluation Verdict: {category}")
    return PipelineResult(category, schema=schema, stage2=stage2_result, stage3=stage3_result, stage4=stage4_result, detail=stage4_result.detail)


def run_full_pipeline(stage1_raw_output: dict, gold_atoms: list, gold_logic: str = "AND", doc_id: str = "UNKNOWN_DOC", raw_text_input: str = "", raw_dataset_row: dict = None) -> PipelineResult:
    """Wrapper coordinating logging scopes and execution phases."""
    import json
    with DocumentLogger(doc_id) as doc_logger:
        doc_logger.log(f"================================================================================", print_to_console=False)
        doc_logger.log(f"DATA TRACE LINEAGE ARTIFACT: {doc_id}", print_to_console=False)
        doc_logger.log(f"================================================================================", print_to_console=False)
        
        doc_logger.log(f"\n[DATA TRACK] --- RAW STAGE 1 TEXT PASSAGE INPUT ---", print_to_console=False)
        doc_logger.log(raw_text_input.strip() if raw_text_input else "No text supplied.", print_to_console=False)
        
        # ────────────── NEW RAW DATASET ENTITY INSPECTION DUMP ──────────────
        if raw_dataset_row:
            doc_logger.log(f"\n[UNFILTERED GOLD GROUND TRUTH] --- ALL HUMAN ANNOTATED TEXT ENTITIES ---", print_to_console=True)
            raw_entities = raw_dataset_row.get("entities", [])
            for i, ent in enumerate(raw_entities):
                ent_text = " ".join(ent.get("text", []))
                ent_type = ent.get("type", "UNKNOWN")
                doc_logger.log(f"  ├─ Entity [{i}] [{ent_type}]: \"{ent_text}\"", print_to_console=True)
                
            doc_logger.log(f"\n[UNFILTERED GOLD GROUND TRUTH] --- ALL RAW RELATION PIPELINES ---", print_to_console=True)
            raw_relations = raw_dataset_row.get("relations", [])
            for i, rel in enumerate(raw_relations):
                doc_logger.log(f"  ├─ Relation [{i}]: Type='{rel.get('type')}' | Arg1='{rel.get('arg1_id')}' | Arg2='{rel.get('arg2_id')}'", print_to_console=True)
        # ────────────────────────────────────────────────────────────────────

        doc_logger.log(f"\n[DATA TRACK] --- STAGE 1 LLM OUTPUT DICTIONARY ---", print_to_console=False)
        doc_logger.log(json.dumps(stage1_raw_output, indent=2), print_to_console=False)
        
        doc_logger.log(f"\n[DATA TRACK] --- PARSED GROUND TRUTH GOLD ATOMS ---", print_to_console=True)
        for i, ga in enumerate(gold_atoms):
            doc_logger.log(f"  ├─ Gold Atom [{i}]: Variable='{ga.var}' | Operator='{ga.op}' | Value={ga.value}", print_to_console=True)
        doc_logger.log(f"================================================================================\n", print_to_console=True)

        
        # Proceed with normal running sequences
        early_result, certified_state = run_phase_1(stage1_raw_output, doc_logger)
        if early_result is not None:
            doc_logger.log(f"[PIPELINE SUMMARY] Execution Terminated in Phase I. Category: {early_result.category}")
            return early_result
        res = run_phase_2(certified_state, gold_atoms, gold_logic, doc_logger)
        doc_logger.log(f"[PIPELINE SUMMARY] Pipeline Completed. Category: {res.category} | Details: {res.detail}")
        return res
