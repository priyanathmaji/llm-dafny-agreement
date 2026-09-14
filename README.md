# Cross-Agent LLM-to-Dafny Formalization

This repository contains the code and experimental artifacts for the paper:

**When Verification Is Not Agreement: Cross-Agent LLM-to-Dafny Formalization**

The repository implements the dual-route evaluation protocol in which two independently prompted LLM agents formalize numeric clinical-trial eligibility criteria into Dafny. The resulting formalizations are checked using Dafny and compared at the normalized atom level to distinguish agreement, coverage gaps, and direct contradictions.

## Repository Overview

The main components are:

* **Route A / Specifier**: independently extracts and formalizes eligibility constraints.
* **Route B / Implementer**: independently extracts constraints and generates an operational Dafny implementation using the required parameter signature.
* **Dafny verification**: checks whether the Implementer's method satisfies the Specifier's formal predicate.
* **Atom-level comparison**: compares the independently extracted constraints to identify semantic agreement and non-convergence.
* **Mutation adequacy testing**: tests whether agreed specifications are sensitive to controlled specification mutations.
* **Structural audit**: detects verification shortcuts in which the Implementer directly invokes the Specifier's predicate.
* **Result aggregation**: produces the summary statistics reported in the paper.

The evaluation uses a 96-document subsample of the **Chia** clinical-trial eligibility corpus.

## Requirements

### Python

The experiments were developed and tested with **Python 3.12**.

Create a virtual environment and install the required dependencies:

```cmd
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Dafny

The Stage 2 formalization and verification pipeline requires **Dafny** and its associated Z3 solver.

Install Dafny according to the official Dafny installation instructions:

https://dafny.org/

After installation, identify the location of the Z3 executable used by Dafny.

The path to the solver must be configured in:

```text
src-gemini/stage2_autoformalization.py
```

The current configuration uses:

```python
DAFNY_SOLVER_PATH = os.environ.get(
    "DAFNY_SOLVER_PATH",
    r"C:\Users\admin\.conda\envs\<folder>\bin\z3.EXE"
)
```

Replace the example path with the actual path to your local `z3.EXE`.

Alternatively, set the `DAFNY_SOLVER_PATH` environment variable before running the experiments.

## LLM API Configuration

The experiments use externally hosted Azure OpenAI deployments.

API credentials are **not included in this repository**.

Create a local `.env` file based on:

```text
.env_sample
```

The sample file contains the following variables:

```text
AZURE_OPENAI_ENDPOINT_2=""
AZURE_OPENAI_RESPONSES_DEPLOYMENT_NAME_2=""
AZURE_OPENAI_API_VERSION_2=""
AZURE_AI_FOUNDRY_API_KEY_2=""

AZURE_OPENAI_ENDPOINT_3=""
AZURE_OPENAI_RESPONSES_DEPLOYMENT_NAME_3=""
AZURE_OPENAI_API_VERSION_3=""
AZURE_AI_FOUNDRY_API_KEY_3=""
```

Populate these variables with credentials for the corresponding Azure OpenAI deployments.

**Do not commit the `.env` file or API keys to the repository.**

The model deployment names used in the experiments correspond to the model configurations reported in the paper.

## Dataset

The experiments use the Chia clinical-trial eligibility corpus.

The dataset is not bundled with this repository. Obtain the dataset from its original distribution source and place the required local dataset files under:

```text
dataset\chia_local_dataset
```

The code expects the dataset path to be configured as:

```python
DATASET_PATH = "dataset\\chia_local_dataset"
```

The Chia dataset is described in:

Kury, F. S. et al. *Chia, a large annotated corpus of clinical trial eligibility criteria.* Scientific Data, 2020.

Dataset DOI:

`10.6084/m9.figshare.11855817`

See the paper and dataset record for the applicable licensing and distribution terms.

## Running the Evaluation

After installing the Python dependencies, configuring the Azure OpenAI credentials, installing Dafny, configuring `DAFNY_SOLVER_PATH`, and placing the dataset in the expected location, the cross-agent evaluation can be run with:

```cmd
python src-gemini/run_cross_agent_evaluation_swapped.py high_tier_subsample_96.jsonl
```

The evaluation script executes the configured cross-agent experiment and writes the resulting document-level records to JSONL output files.

The four model configurations evaluated in the paper are:

1. GPT-4.1-mini / GPT-4.1-mini
2. GPT-4.1-nano / GPT-4.1-mini
3. GPT-4.1-mini / GPT-4.1-nano
4. GPT-5.6-luna / GPT-5.4-mini

The first model in each pair is the **Specifier** and the second is the **Implementer**.

## Aggregating Results

After an evaluation has produced its JSONL results, use the result aggregation script to generate the corresponding summary.

### GPT-4.1-mini Specifier / GPT-4.1-nano Implementer

```cmd
python src-gemini/result_aggregation_script.py cross_agent_results_gpt-4.1-mini_spec__gpt-4.1-nano_impl.jsonl > original_run_summary.txt
```

### GPT-4.1-nano Specifier / GPT-4.1-mini Implementer

```cmd
python src-gemini/result_aggregation_script.py cross_agent_results_gpt-4.1-nano_spec__gpt-4.1-mini_impl.jsonl > swapped_run_summary.txt
```

### GPT-4.1-mini Specifier / GPT-4.1-mini Implementer

```cmd
python src-gemini/result_aggregation_script.py cross_agent_results_gpt-4.1-mini_spec__gpt-4.1-mini_impl.jsonl > original_run_summary_4.1mini.txt
```

### GPT-5.6-luna Specifier / GPT-5.4-mini Implementer

```cmd
python src-gemini/result_aggregation_script.py cross_agent_results_gpt-5.6-luna_spec__gpt-5.4-mini_impl.jsonl > original_run_summary_56luna_54mini.txt
```

The resulting text files contain the aggregated document-level statistics used to produce the reported experimental results.

## Reproducibility Notes

The evaluation is based on externally hosted LLM inference. Consequently, exact model-generated outputs may vary across API calls or model-service versions even when the same source documents and prompts are used.

The repository provides the experimental code, prompts, schemas, normalization logic, evaluation rules, mutation procedures, structural auditing, and result aggregation needed to reproduce the evaluation protocol.

Reproducing the LLM inference component requires access to the corresponding externally hosted model deployments.

The repository does not contain API credentials.

## Hardware and Compute

Local experiments were executed on a machine with:

* Intel Core i7 CPU, 1.8 GHz
* 16 GB RAM
* No GPU

Local computation was used for experiment orchestration, data processing, normalization, Dafny verification, mutation testing, structural auditing, and result analysis.

Across the reported experiments, approximately 3,494 hosted-model inference requests were made, corresponding to approximately 5.44 million total tokens:

| Model        |  Requests | Prompt Tokens | Completion Tokens | Total Tokens |
| ------------ | --------: | ------------: | ----------------: | -----------: |
| GPT-4.1-nano |     1,030 |         1.51M |            0.110M |        1.62M |
| GPT-4.1-mini |     2,080 |         2.94M |            0.252M |        3.19M |
| GPT-5.4-mini |       192 |        0.287M |            0.018M |       0.305M |
| GPT-5.6-luna |       192 |        0.272M |            0.048M |       0.320M |
| **Total**    | **3,494** |     **5.01M** |        **0.428M** |    **5.44M** |

The total hosted inference cost for the reported experiments was approximately **$5**.

## Repository Contents

The repository is organized around the experimental pipeline:

```text
.
├── .env_sample
├── requirements.txt
├── dataset/
│   └── chia_local_dataset/
├── src-gemini/
│   ├── stage1_llm_extraction.py
│   ├── stage2_autoformalization.py
│   ├── run_cross_agent_evaluation_swapped.py
│   ├── result_aggregation_script.py
│   └── ...
└── README.md
```

Additional files and analysis scripts supporting the evaluation are included in the repository.


## Citation

If you use this code or evaluation protocol, please cite:

```bibtex
@inproceedings{anonymous2026verification,
  title     = {When Verification Is Not Agreement: Cross-Agent LLM-to-Dafny Formalization},
  author    = {Anonymous},
  booktitle = {NeurIPS 2026 Workshop},
  year      = {2026}
}
```

The evaluation uses the Chia clinical-trial eligibility corpus. Please also cite the original Chia publication:

```bibtex
@article{kury2020chia,
  title   = {Chia, a large annotated corpus of clinical trial eligibility criteria},
  author  = {Kury, Fabrice S. and others},
  journal = {Scientific Data},
  volume  = {7},
  pages   = {1--11},
  year    = {2020},
  doi     = {10.1038/s41597-020-00620-0}
}
```

