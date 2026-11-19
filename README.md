# NLP Project I: Structured Text Classification & Entity Extraction

Local open-source LLM evaluation pipeline built with Python and **Ollama** (`v0.35.0`, zero cloud APIs). Evaluates Turkish text classification (`category`, `sentiment`) and entity extraction (`entities`) across **two experimental settings** (`prompt_only` vs. `schema_constrained`) and **four temperatures** ($T \in \{0.1, 0.4, 0.7, 1.0\}$).

## Reproducibility & Environment

- **Ollama Version:** `0.35.0` (`/api/chat`)
- **Models Evaluated (all `Q4_K_M` quantization):**
  - `llama3.1:latest` (8.0B, 4.9 GB)
  - `llama3.2:latest` (3.2B, 2.0 GB)
  - `qwen2.5:1.5b` (1.5B, 986 MB)
- **Fixed Parameters:** `seed = 42`, `num_ctx = 4096`, `num_predict = 256`
- **Dataset:** `data/benchmark_200.json` (curated from `winvoker/turkish-sentiment-analysis-dataset`; first 100 samples manually annotated with **130 gold entities** via `data/annotate_gold_100.py`)
- **Turkish Lowercasing Rule (`src/metrics.py`):** Replaces `'İ' -> 'i'` and `'I' -> 'ı'` *before* calling `.lower()` to prevent Unicode combining dot (`\u0307`) mismatches.

## Repository Structure

```text
├── config.py                     # Models, temperatures, SYSTEM_PROMPT, JSON_SCHEMA, seed=42
├── main.py                       # CLI orchestrator (--mode prompt_only | schema_constrained | both)
├── demo_live.py                  # Uncut live tests demo (settings, T=0 vs T=1, 5 unseen samples)
├── requirements.txt
├── data/
│   ├── fetch_sentiment_dataset.py
│   ├── annotate_gold_100.py      # 130 gold entity annotations for first 100 samples
│   └── benchmark_200.json
├── src/
│   ├── ollama_runner.py          # Local Ollama REST client with TPS/latency telemetry
│   ├── cleaner.py                # 3-stage try-except JSON parser & schema validator
│   ├── metrics.py                # Turkish-aware normalization, Macro F1, Exact/Partial Entity F1
│   ├── pipeline.py               # Evaluation loop & per-mode artifact export
│   ├── visualizer.py             # Per-model temperature curves & dashboards
│   ├── enrich_metrics.py         # Generates per-model Markdown tables & comparison plots
│   └── demo_defense_experiments.py # Live defense experiments (HF tokenizer, İ/I ablation, scale)
├── results/
│   ├── all_models_summary.csv    # Combined 24-row summary (3 models x 4 temps x 2 modes)
│   └── <model_slug>/
│       ├── prompt_only/          # CSV/JSON raw records & summaries
│       ├── schema_constrained/   # CSV/JSON raw records & summaries
│       ├── prompt_only_tables.md
│       └── schema_constrained_tables.md
└── reports/
    ├── NLP_Project_1_Report.pdf  # Final 4-page technical report
    ├── benchmark_report.md
    └── figures/
        ├── ortak_grafikler/      # Multi-model & Prompt vs Schema comparison charts
        └── <model_slug>/{prompt_only,schema_constrained}/
```

## Quick Start

```bash
# 1. Install dependencies
python3 -m venv venv
./venv/bin/pip install -r requirements.txt

# 2. Ensure Ollama is running with required models
ollama serve
ollama pull qwen2.5:1.5b
ollama pull llama3.2:latest
ollama pull llama3.1:latest

# 3. Run evaluation (100 annotated samples)
./venv/bin/python main.py --mode both --samples 100
./venv/bin/python src/enrich_metrics.py

# 4. Run live tests & defense experiments
./venv/bin/python demo_live.py
./venv/bin/python src/demo_defense_experiments.py
```

## Key Results Summary (T = 0.1, 100 Samples)

| Model | Setting | Raw JSON | Cleaned JSON | Schema Valid | Cat Acc (Parsed / Macro F1) | Sent Acc (Parsed / Macro F1) | Entity Exact F1 | Entity Partial F1 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `llama3.1:latest` | Prompt-Only | 100.0% | 100.0% | 100.0% | 90.0% / 67.8% | 81.0% / 81.0% | **50.4%** | **64.4%** |
| `llama3.1:latest` | Schema-Constrained | 100.0% | 100.0% | 100.0% | **92.0% / 91.7%** | **82.0% / 82.1%** | 46.1% | 61.5% |
| `llama3.2:latest` | Prompt-Only | 0.0% | 100.0% | 100.0% | 32.0% / 4.2% | **69.0% / 53.2%** | **30.8%** | **42.4%** |
| `llama3.2:latest` | Schema-Constrained | **98.0%** | 98.0% | 98.0% | **58.2% / 49.7%** | 64.3% / 49.8% | 27.9% | 39.0% |
| `qwen2.5:1.5b` | Prompt-Only | 9.0% | 97.0% | 97.0% | 55.7% / 54.9% | 75.3% / 75.2% | 23.4% | 41.1% |
| `qwen2.5:1.5b` | Schema-Constrained | **98.0%** | **98.0%** | **98.0%** | **56.1%** / 54.7% | **76.5% / 76.0%** | **27.0%** | **44.3%** |

See [reports/NLP_Project_1_Report.pdf](reports/NLP_Project_1_Report.pdf) for full methodology, temperature sweeps, error analysis, and discussion.

## Project Development Timeline

- **2026-10-21 – 2026-10-22:** Environment setup, dataset curation scripts, and initial 70-sample experiments.
- **2026-10-26:** Initial benchmark report and HTML presentation slides.
- **2026-11-13 – 2026-11-17:** Strict JSON cleaner, Ollama telemetry runner, 100-sample manual gold entity annotation, and Turkish-aware Entity/Macro F1 metrics.
- **2026-11-19:** Full `schema_constrained` benchmark across all 3 models, live demo & defense experiment scripts, symmetrical per-model tables/charts, and final 4-page technical report PDF.
