# NLP Project I: Structured Text Classification & Information Extraction with Local LLMs

Local open-source LLM evaluation project implemented in Python using **Ollama** (zero cloud APIs). The system forces local instruction-tuned LLMs to emit strictly valid JSON payloads (`category`, `sentiment`, `entities`) from unstructured Turkish texts and compares **Prompt-Only** generation against **Schema-Constrained** decoding (`format=JSON_SCHEMA`) across multiple model scales and temperatures.

---

## 1. Reproducibility & Experimental Setup

- **Runtime:** Local Ollama `v0.35.0` via REST endpoint `http://localhost:11434/api/chat`
- **Evaluated Models (verified with `ollama show`):**
  - `qwen2.5:1.5b` — 1.5B parameters, `Q4_K_M` quantization (986 MB)
  - `llama3.2:latest` — 3.2B parameters, `Q4_K_M` quantization (2.0 GB)
  - `llama3.1:latest` — 8.0B parameters, `Q4_K_M` quantization (4.9 GB)
- **Fixed Generation Parameters (`config.py` & `src/ollama_runner.py`):**
  - `seed = 42`
  - `num_ctx = 4096` (context length)
  - `num_predict = 256` (max generated tokens)
  - `temperature` sweep: `[0.1, 0.4, 0.7, 1.0]`
- **Evaluation Scope:** First **100 annotated samples** from `data/benchmark_200.json` $\times$ 4 temperatures $\times$ 3 models $\times$ 2 settings = **2,400 total inferences**.

---

## 2. Dataset, Gold Entity Annotation & Turkish Matching Rules

- **Source Dataset (`data/fetch_sentiment_dataset.py`):** Curated from HuggingFace `winvoker/turkish-sentiment-analysis-dataset` across 3 categories (`urun`, `sinema`, `ansiklopedi`) and 3 sentiments (`olumlu`, `olumsuz`, `notr`), filtered to 60–320 characters (`data/benchmark_200.json`).
- **Gold Entity Annotation (`data/annotate_gold_100.py`):** Because the upstream dataset only includes class/sentiment labels, the first 100 evaluation samples were manually annotated with **130 gold entities** (95 non-empty samples; 5 samples legitimately have `[]`). Target entity types follow the system prompt: *Person*, *Organization/Brand*, *Location*, *Product/Device/Model*, and *Film/Work*.
- **Turkish Lowercasing Rule (`src/metrics.py -> turkish_lower`):** Standard Python `'İSTANBUL'.lower()` produces `'i\u0307stanbul'` (with a Unicode combining dot) and `'IŞIK'.lower()` produces `'işik'`. Our pipeline replaces `'İ' -> 'i'` and `'I' -> 'ı'` **before** calling `.lower()`, then normalizes diacritics (`ıöüçşğ -> ioucsg`) and punctuation.
- **Entity Matching Rules (`src/metrics.py -> is_entity_match`):**
  - **Exact Match:** Normalized predicted and gold entity strings are identical.
  - **Partial Match:** Substring containment (`p in g or g in p`) or non-empty word-set overlap (`p_words & g_words`), preventing unfair penalties on Turkish agglutinative suffixes.

---

## 3. Repository Structure & What Every File Does

```text
├── config.py                        # Models, temperatures, SYSTEM_PROMPT, JSON_SCHEMA, seed=42, num_ctx=4096
├── main.py                          # Main CLI pipeline orchestrator (--mode, --model, --samples, --skip-eval)
├── demo_live.py                     # Uncut live-test script (Prompt-Only vs Schema, T=0 vs T=1, 5 unseen samples)
├── requirements.txt                 # Python package dependencies
│
├── data/
│   ├── fetch_sentiment_dataset.py   # Downloads & curates 200 Turkish samples from HuggingFace
│   ├── annotate_gold_100.py         # Writes 130 manual gold_entities onto the first 100 benchmark samples
│   ├── benchmark_200.json           # Active 200-sample dataset (first 100 include gold_entities)
│   ├── fetch_dataset.py             # Legacy script from early news-classification prototype
│   └── benchmark_200_old_news.json  # Archived backup of the earlier news dataset
│
├── src/
│   ├── ollama_runner.py             # OllamaRunner: calls /api/chat with optional `format` schema & captures TPS/latency
│   ├── cleaner.py                   # 3-stage try-except parser: Raw JSON -> Cleaned JSON -> Schema Validation
│   ├── metrics.py                   # turkish_lower(), normalize(), Accuracy, Macro F1, Exact & Partial Entity F1
│   ├── pipeline.py                  # EvaluationPipeline: runs inference loop, computes all metrics, exports CSV/JSON/plots
│   ├── visualizer.py                # Generates per-model & per-mode PNG curves and 3-panel dashboards
│   ├── enrich_metrics.py            # Recomputes Entity/Macro F1 from raw logs, builds *_tables.md & comparison charts
│   └── demo_defense_experiments.py  # Live defense script (HF BPE tokenization of "kitaplarımızdan", İ/I ablation, 1.5B vs 8B)
│
├── results/
│   ├── all_models_summary.csv       # Master 24-row summary table (3 models x 4 temps x 2 settings)
│   ├── <model_slug>/                # llama3.1_latest, llama3.2_latest, qwen2.5_1.5b
│   │   ├── prompt_only/             # detailed_records.csv, evaluation_summary.csv, evaluation_summary.json
│   │   ├── schema_constrained/      # detailed_records.csv, evaluation_summary.csv, evaluation_summary.json
│   │   ├── prompt_only_tables.md    # Human-readable Markdown tables for Prompt-Only
│   │   └── schema_constrained_tables.md # Human-readable Markdown tables for Schema-Constrained
│   └── _legacy_70samples/           # Archived early 70-sample qwen2.5:1.5b prototype outputs
│
├── reports/
│   ├── NLP_Project_1_Report.pdf     # Final 4-page technical report (methodology, results, discussion)
│   ├── benchmark_report.md          # Detailed Markdown summary report with links to all per-model tables
│   ├── slaytlar.html                # Interactive HTML presentation slides (early Prompt-Only showcase)
│   └── figures/
│       ├── ortak_grafikler/         # Multi-model Entity F1 & Prompt-vs-Schema comparison PNGs
│       ├── <model_slug>/{prompt_only,schema_constrained}/ # 5 charts per model per setting
│       └── _legacy_70samples/       # Archived charts from the early 70-sample run
```

---

## 4. How to Install & Run on Any Machine

### Step 1: Prerequisites (Python & Ollama)
1. Install **Python 3.10+** and **Ollama** (`https://ollama.com`).
2. Start the Ollama server and pull the exact models used in this project:
   ```bash
   ollama serve
   ollama pull qwen2.5:1.5b
   ollama pull llama3.2:latest
   ollama pull llama3.1:latest
   ```
   *(Optional: verify quantization and parameters via `ollama show llama3.1:latest`.)*

### Step 2: Create Virtual Environment & Install Dependencies
```bash
git clone https://github.com/yigit-bora/nlp_project_1.git
cd nlp_project_1
python3 -m venv venv
./venv/bin/pip install -r requirements.txt
```
*(Note: To rebuild the PDF report from HTML, `weasyprint` and `pypdf` were used; they are not required to run the LLM evaluation or demo scripts.)*

### Step 3: Run Evaluations (`main.py`)
> **Important:** Gold entity labels exist on the **first 100 samples** of `data/benchmark_200.json`, so always pass `--samples 100` for full benchmark reproduction.

- **Run both settings (`prompt_only` + `schema_constrained`) for all 3 models (100 samples each):**
  ```bash
  ./venv/bin/python main.py --mode both --samples 100
  ```
- **Run a single setting or single model:**
  ```bash
  ./venv/bin/python main.py --mode schema_constrained --samples 100
  ./venv/bin/python main.py --model llama3.1:latest --mode both --samples 100
  ```
- **Quick 2-sample smoke test:**
  ```bash
  ./venv/bin/python main.py --model qwen2.5:1.5b --mode both --samples 2
  ```
- **Regenerate all Markdown tables (`*_tables.md`), `all_models_summary.csv`, and all 35 PNG charts from existing CSV logs (without calling Ollama):**
  ```bash
  ./venv/bin/python src/enrich_metrics.py
  ```

### Step 4: Run Live Video & Defense Demos
- **Live Tests Demo (`demo_live.py`)** — runs Prompt-Only vs. Schema-Constrained on identical samples, `T=0.0` vs. `T=1.0`, and 5 unseen samples (`NEW-1`..`NEW-5`, including an adversarial sarcasm case):
  ```bash
  ./venv/bin/python demo_live.py                        # default: qwen2.5:1.5b (shows markdown-fence & T paradox)
  ./venv/bin/python demo_live.py --model llama3.1:latest # 8B model (high accuracy + sarcasm corner case)
  ./venv/bin/python demo_live.py --model llama3.2:latest # 3B model
  ```
- **Defense Experiments Demo (`src/demo_defense_experiments.py`)** — live HuggingFace tokenizer split of `"kitaplarımızdan"` (Qwen & Llama 3), Turkish `'İ'`/`'I'` combining-dot ablation test (`src/metrics.py` line 14), and 1.5B vs. 8B scale comparison:
  ```bash
  ./venv/bin/python src/demo_defense_experiments.py
  ```
  *(Requires internet access on first run to cache HuggingFace tokenizer files for `Qwen/Qwen2.5-1.5B-Instruct` and `unsloth/llama-3-8b-Instruct`.)*

### Step 5: View Reports & Presentation Slides
- **4-Page Technical Report PDF:** Open [reports/NLP_Project_1_Report.pdf](reports/NLP_Project_1_Report.pdf)
- **Markdown Benchmark Report:** Open [reports/benchmark_report.md](reports/benchmark_report.md)
- **HTML Slides:** Open [reports/slaytlar.html](reports/slaytlar.html) in a browser or serve locally:
  ```bash
  python3 -m http.server 8000
  # then visit http://localhost:8000/reports/slaytlar.html
  ```

---

## 5. Key Results & Discussion Summary (T = 0.1, 100 Samples)

| Model | Setting | Raw JSON | Cleaned JSON | Schema Valid | Cat Acc (Strict / Parsed) | Cat Macro F1 | Sent Acc (Strict / Parsed) | Sent Macro F1 | Entity Exact F1 | Entity Partial F1 | Mean TPS |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `llama3.1:latest` | Prompt-Only | 100.0% | 100.0% | 100.0% | 90.0% / 90.0% | 67.8% | 81.0% / 81.0% | 81.0% | **50.4%** | **64.4%** | 28.0 |
| `llama3.1:latest` | Schema-Constrained | 100.0% | 100.0% | 100.0% | **92.0% / 92.0%** | **91.7%** | **82.0% / 82.0%** | **82.1%** | 46.1% | 61.5% | 26.1 |
| `llama3.2:latest` | Prompt-Only | 0.0% | 100.0% | 100.0% | 0.0% / 32.0% | 4.2% | 0.0% / **69.0%** | **53.2%** | **30.8%** | **42.4%** | 76.0 |
| `llama3.2:latest` | Schema-Constrained | **98.0%** | 98.0% | 98.0% | **57.0% / 58.2%** | **49.7%** | **63.0%** / 64.3% | 49.8% | 27.9% | 39.0% | 73.5 |
| `qwen2.5:1.5b` | Prompt-Only | 9.0% | 97.0% | 97.0% | 6.0% / 55.7% | **54.9%** | 6.0% / 75.3% | 75.2% | 23.4% | 41.1% | 145.7 |
| `qwen2.5:1.5b` | Schema-Constrained | **98.0%** | **98.0%** | **98.0%** | **55.0% / 56.1%** | 54.7% | **75.0% / 76.5%** | **76.0%** | **27.0%** | **44.3%** | **163.8** |

**Main Takeaways:**
1. **Format Discipline:** Schema-constrained decoding fixes markdown fences (`qwen2.5:1.5b`: 9% $\to$ 98%) and chat preambles (`llama3.2:latest`: 0% $\to$ 98%). Remaining 1%–4% small-model failures occur when unbounded `entities` arrays enter repetition loops and hit `num_predict=256`.
2. **Task Accuracy Impact (Closed-Set vs. Open-Ended):** Schema constraints strongly improve **closed-set `enum` classification** (`llama3.1` Category Macro F1: 67.8% $\to$ 91.7%; `llama3.2`: 4.2% $\to$ 49.7%), but slightly degrade **open-ended `entities` F1** on 3B/8B models (e.g., `llama3.1` Exact F1: 50.4% $\to$ 46.1%) due to autoregressive conditioning shifts and GBNF logit-masking perturbations.
3. **Temperature Paradox (Prompt-Only):** Higher temperature broke `qwen2.5:1.5b`'s markdown-fence reflex (Raw JSON 9% $\to$ 31% at $T=0.7$), whereas it provoked verbosity and 256-token truncation in `llama3.2:latest` (Schema Validity 100% $\to$ 88%).
