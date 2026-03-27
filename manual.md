# GreenPrompt — User Manual

This manual covers dependency installation, API key configuration, and first-run instructions for the GreenPrompt experimental framework on Windows 10/11.

---

## Requirements

- Python 3.10 or later
- An OpenAI API key with access to `gpt-3.5-turbo`
- Windows 10/11

---

## Installing Dependencies

Navigate to the project directory in Command Prompt and run:

```bash
pip install -r requirements.txt
```

If a `requirements.txt` is not present, install the required packages manually:

```bash
pip install openai datasets codecarbon radon streamlit plotly pandas numpy matplotlib
```

---

## Setting Up Your OpenAI API Key  (For the current session only)

```bash
set OPENAI_API_KEY=your-key-here
```

This must be repeated each time a new terminal is opened.

Verify the key is set with:

```bash
echo %OPENAI_API_KEY%
```

---

## CodeCarbon

### What CodeCarbon does

CodeCarbon is a Python library that estimates the CO₂ equivalent emissions produced by code execution. It measures energy consumption by reading the CPU's thermal design power (TDP) and system-level power draw, then converts this to a carbon emissions estimate using a regional carbon intensity factor based on the machine's location. Emissions are reported in kilograms of CO₂ equivalent (kg CO₂e).

In this project, CodeCarbon is used in two places:

- **Inference emissions**: measured during each API call to GPT-3.5 turbo, capturing the local computational cost of sending and receiving the request
- **Execution emissions**: measured inside the sandbox worker during execution of each generated solution, capturing the carbon emissions of running the code

Both are recorded separately for each problem-strategy combination and stored in `data/new_final_results.jsonl`.

### Installation

CodeCarbon is included in the project dependencies and will be installed automatically via `pip install -r requirements.txt`. To install it individually:

```bash
pip install codecarbon
```

### Configuration in this project

CodeCarbon is initialised and stopped within the sandbox worker process. It is configured to track at the machine level, meaning it measures total system power draw rather than isolating a single process. The tracker is started immediately before `exec()` is called on the generated code and stopped as soon as execution terminates, limiting the measurement window as closely as possible to the generated code's execution.

### Known limitations

CodeCarbon provides **estimates**, not direct measurements. It does not use a hardware power meter — it derives energy consumption from the CPU's TDP rating, which represents maximum rated power rather than actual instantaneous draw. This means:

- Emissions may be slightly overestimated for low-intensity tasks
- Background system processes on Windows (such as antivirus or scheduled tasks) cannot be fully excluded from measurements
- Measurements are hardware-specific and should not be compared directly across different machines

These limitations are consistent with prior research using CodeCarbon in similar experimental settings (Cursaru et al. 2024) and are discussed further in the dissertation's limitations section.

---

## Running the Pipeline

```bash
python pipeline.py
```

Results are saved incrementally to `new_final_results.jsonl` after each problem is processed. If the pipeline is interrupted, completed results are preserved and the run can be restarted.

> **Note:** The full pipeline makes approximately 1,676 API calls across 419 problems and four strategies. You must have sufficient OpenAI API credit before starting!

### Configuration

The following parameters can be adjusted at the top of `pipeline.py`:

| Parameter | Default | Description |
|---|---|---|
| `MODEL` | `gpt-3.5-turbo` | OpenAI model to use |
| `MAX_TOKENS` | `1024` | Maximum tokens per API response |
| `TEMPERATURE` | `0.0` | Sampling temperature (0 = deterministic) |
| `TIMEOUT` | `15` | Execution timeout in seconds per solution |

---


## Running the Dashboard

```bash
streamlit run dashboard.py
```

Opens automatically at `http://localhost:8501`. Requires `new_final_results.jsonl` to be present in the same directory.
