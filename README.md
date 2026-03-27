# GreenPrompt
This project explores the intersection of prompt engineering and environmental sustainability in large language model (LLM)–based code generation. While prompt design is widely known to influence the accuracy and reasoning of LLM outputs, its impact on the carbon footprint of the generated code remains underexplored.

We hypothesize that different prompting strategies—such as zero-shot, few-shot, chain-of-thought (CoT), and self-reflection (e.g., Reflexion)—not only affect correctness but also influence code complexity, runtime efficiency, and energy consumption. The project aims to systematically evaluate these effects.

# GreenPrompt

> Investigating the Carbon Impact of Prompt Engineering in Code Generation

Honours Individual Project dissertation by Melissa Dorrian (University of Glasgow, 2026). Investigating how different LLM prompting strategies affect the carbon emissions of generated Python code, using the MBPP benchmark dataset.

---

## Project Structure

```
greenprompt/
├── data/                       # Dataset and results
│   ├── new_final_results.jsonl # Experimental results output
│   └── merged_dataset          # Merged MBPP dataset
├── main.py                     # Entry point — runs the full pipeline
├── llm_inference.py            # Handles OpenAI API calls
├── prompt_strategies.py        # Prompt formatting for each strategy
├── apply_strategy.py           # Applies strategy formatting to problems
├── execution_harness.py        # Worker-based sandbox execution harness
├── complexity_analysis.py      # Static code complexity analysis (Radon)
├── load_data.py                # Loads and preprocesses the MBPP dataset
├── name_enforce.py             # Enforces correct function naming
├── app.py                      # Streamlit interactive dashboard
├── README.md                   # Project overview
└── manual.md                   # Installation and usage guide
```

---

## Requirements

- Python 3.10 or later
- An OpenAI API key with access to `gpt-3.5-turbo`
- Windows 10/11 (results in the dissertation were collected on Windows 11 Home)

> For full installation and setup instructions, including dependency installation and API key configuration, see **`manual.md`**.

---

## Running the Pipeline

The pipeline loads the MBPP dataset, generates solutions under four prompting strategies, executes them in a sandbox, measures carbon emissions and other metrics, and saves results to a JSONL file.

Open a terminal (Command Prompt or PowerShell) in the project directory and run:

```bash
python pipeline.py
```

Results are saved incrementally to `new_final_results.jsonl` as each problem is processed. If the pipeline is interrupted, partial results are preserved.

**Configuration options** (set at the top of `pipeline.py`):

| Parameter | Default | Description |
|---|---|---|
| `MODEL` | `gpt-3.5-turbo` | OpenAI model to use |
| `MAX_TOKENS` | `1024` | Max tokens per LLM response |
| `TEMPERATURE` | `0.0` | Sampling temperature (0 = deterministic) |
| `TIMEOUT` | `15` | Execution timeout in seconds |

> **Note:** In this project all executions were ran on the same machine to ensure fair comparison of carbon emissions across strategies. Background processes on non-dedicated hardware will introduce minor noise. This is a known limitation discussed in the dissertation.

---


## Running the Dashboard

The interactive Streamlit dashboard allows visual exploration of the results across all metrics and strategies.

Open a terminal in the project directory and run:

```bash
streamlit run app.py
```

This will open the dashboard in your default browser automatically. If it does not open, navigate to `http://localhost:8501` manually.

The dashboard requires `new_final_results.jsonl` to be present in the same directory.

**Dashboard features:**

- Overview of summary statistics across all runs
- Per-problem analysis and strategy comparison
- Win rate visualisation across matched problems
- Trade-off explorer: execution carbon vs complexity metrics
- Explore generated code tab — view and compare solutions by problem ID

---

## Prompting Strategies

The pipeline evaluates four prompting strategies:

| Strategy | Description |
|---|---|
| Zero-shot | Natural language instruction only, no examples |
| Few-shot | Includes two worked input/output examples |
| Chain-of-thought | Instructs the model to reason step by step |
| Reflexion | Model generates a solution, reflects on it, then revises |

---

## Notes on Reproducibility

- Temperature is set to `0.0` for all API calls to produce deterministic outputs, ensuring variation in results is attributable to prompt strategy rather than sampling randomness.
- All code executions are run in a restricted sandbox environment. Only `math`, `re`, `itertools` and `random` modules are permitted, and a 15 second timeout is enforced.
- Carbon emissions are estimated using [CodeCarbon](https://github.com/mlco2/codecarbon), which derives CO₂ estimates from CPU thermal design power (TDP). Measurements are hardware-dependent and should be treated as estimates.

---

