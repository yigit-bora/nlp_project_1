#!/usr/bin/env python3
"""
NLP Project I - Live Tests Demo Script (Uncut Video Recording)

Covers all three mandatory Live Test items from the specification:
1. Same samples in both settings: Prompt-Only vs. Schema-Constrained (`format=JSON_SCHEMA`)
2. Same samples with two different temperatures: `T = 0.0` vs. `T = 1.0`
3. 5 brand-new samples NOT in the dataset (including 1 adversarial/sarcastic sample expected to fail)

Usage:
  ./venv/bin/python demo_live.py
  ./venv/bin/python demo_live.py --model llama3.2:latest
  ./venv/bin/python demo_live.py --model qwen2.5:1.5b
"""

import argparse
import json
import sys
from pathlib import Path

from config import (
    DATASET_PATH,
    DEFAULT_CONTEXT_LENGTH,
    DEFAULT_SEED,
    JSON_SCHEMA,
    SYSTEM_PROMPT,
    USER_PROMPT_TEMPLATE,
)
from src.cleaner import parse_llm_json
from src.ollama_runner import OllamaRunner

# Test 1 & 2 için aynı örnekler (veri setinden 2 örnek)
SAME_SAMPLES = [
    {
        "label": "Sample A (Product / Positive)",
        "text": "Samsung Galaxy S24 telefonun kamerası ve ekran parlaklığı mükemmel, Hepsiburada üzerinden sipariş verdim ertesi gün sorunsuz geldi.",
    },
    {
        "label": "Sample B (Encyclopedia / Neutral)",
        "text": "Orta Doğu Teknik Üniversitesi, 1956 yılında Ankara'da kurulmuş olup Türkiye'nin önde gelen devlet üniversitelerinden biridir.",
    },
]

# Test 3: Veri setinde OLMAYAN 5 yeni örnek (5. örnek kasıtlı tuzak/ironi içerir)
NEW_SAMPLES_5 = [
    {
        "id": "NEW-1",
        "description": "Clear Positive Product Review",
        "expected_category": "urun",
        "expected_sentiment": "olumlu",
        "expected_entities": ["Sony WH-1000XM5", "Amazon"],
        "text": "Sony WH-1000XM5 kulaklığın gürültü engelleme özelliği olağanüstü, Amazon üzerinden ertesi gün elime ulaştı.",
    },
    {
        "id": "NEW-2",
        "description": "Clear Negative Cinema Review",
        "expected_category": "sinema",
        "expected_sentiment": "olumsuz",
        "expected_entities": ["Zeki Demirkubuz", "İzmir"],
        "text": "Zeki Demirkubuz imzalı bu yeni filmi İzmir'deki galada izledim ancak senaryosu çok dağınık ve son derece sıkıcıydı.",
    },
    {
        "id": "NEW-3",
        "description": "Neutral Encyclopedic Sentence (with Turkish uppercase İ/I)",
        "expected_category": "ansiklopedi",
        "expected_sentiment": "notr",
        "expected_entities": ["İSTANBUL", "Boğaziçi Üniversitesi", "Rumeli Hisarı"],
        "text": "İSTANBUL ilindeki Boğaziçi Üniversitesi, 1863 yılında Robert Kolej olarak Rumeli Hisarı sırtlarında kurulmuştur.",
    },
    {
        "id": "NEW-4",
        "description": "Mixed Aspect Review (Good product, terrible shipping)",
        "expected_category": "urun",
        "expected_sentiment": "olumsuz",
        "expected_entities": ["Philips", "Aras Kargo"],
        "text": "Philips kahve makinesi gerçekten harika çalışıyor fakat Aras Kargo paketi tamamen ezmiş ve teslimatı 6 gün geciktirmiş.",
    },
    {
        "id": "NEW-5",
        "description": "EXPECTED TO FAIL (Heavy Sarcasm / Irony Corner Case)",
        "expected_category": "urun",
        "expected_sentiment": "olumsuz",
        "expected_entities": ["Dyson", "Yurtiçi Kargo"],
        "text": "Gerçekten muhteşem bir deneyim! Dyson süpürge daha ilk çalıştırmada bozuldu, Yurtiçi Kargo da kutuyu paramparça getirerek günüme neşe kattı, bayıldım!",
        "note": "Sarcastic positive words ('muhteşem', 'neşe kattı', 'bayıldım') often trick small/medium LLMs into predicting 'olumlu' instead of 'olumsuz'.",
    },
]


def print_banner(title: str) -> None:
    print("\n" + "=" * 78)
    print(f"  {title}")
    print("=" * 78)


def run_single(runner: OllamaRunner, model: str, text: str, temperature: float, use_schema: bool) -> None:
    user_prompt = USER_PROMPT_TEMPLATE.format(text=text)
    schema = JSON_SCHEMA if use_schema else None
    res_api = runner.generate(
        model=model,
        system_prompt=SYSTEM_PROMPT,
        user_prompt=user_prompt,
        temperature=temperature,
        format_schema=schema,
        seed=DEFAULT_SEED,
        context_length=DEFAULT_CONTEXT_LENGTH,
    )
    parsed = parse_llm_json(res_api["raw_text"])

    setting_label = "SCHEMA-CONSTRAINED (format=JSON_SCHEMA)" if use_schema else "PROMPT-ONLY (no format param)"
    print(f"  [{setting_label} | T={temperature} | TPS={res_api['tps']} | Latency={res_api['latency_sec']}s]")
    print(f"  Raw Output    : {res_api['raw_text']!r}")
    print(
        f"  Parse Metrics : pure_json={parsed.is_pure_json} | "
        f"valid_syntax={parsed.is_valid_syntax} | "
        f"schema_valid={parsed.is_schema_valid} | "
        f"format_type={parsed.format_type}"
    )
    if parsed.data:
        print(f"  Parsed JSON   : {json.dumps(parsed.data, ensure_ascii=False)}")
    else:
        print(f"  Parse Error   : {parsed.error}")


def verify_new_samples_not_in_dataset() -> None:
    if not DATASET_PATH.exists():
        return
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        dataset_texts = {item.get("text", "").strip() for item in json.load(f)}
    for s in NEW_SAMPLES_5:
        assert s["text"].strip() not in dataset_texts, f"Sample {s['id']} is already in dataset!"


def main() -> None:
    parser = argparse.ArgumentParser(description="Live Tests Demo for NLP Project I Video")
    parser.add_argument("--model", type=str, default="qwen2.5:1.5b", help="Ollama model tag to test live")
    args = parser.parse_args()

    verify_new_samples_not_in_dataset()

    runner = OllamaRunner()
    if not runner.is_healthy():
        print("ERROR: Ollama is not running. Start it with `ollama serve`.")
        sys.exit(1)

    print_banner(f"NLP PROJECT I - LIVE TESTS DEMO (Model: {args.model}, Seed: {DEFAULT_SEED})")
    print("Warming up model into VRAM...")
    runner.warmup(args.model)

    # 1. Prompt-Only vs Schema-Constrained on the SAME samples
    print_banner("PART 1: SAME SAMPLES -> PROMPT-ONLY vs. SCHEMA-CONSTRAINED (T=0.1)")
    for sample in SAME_SAMPLES:
        print(f"\n--- {sample['label']} ---")
        print(f"Text: {sample['text']}")
        run_single(runner, args.model, sample["text"], temperature=0.1, use_schema=False)
        run_single(runner, args.model, sample["text"], temperature=0.1, use_schema=True)

    # 2. Temperature 0.0 vs 1.0 on the SAME samples
    print_banner("PART 2: SAME SAMPLES -> TEMPERATURE T=0.0 vs. T=1.0 (Prompt-Only)")
    for sample in SAME_SAMPLES:
        print(f"\n--- {sample['label']} ---")
        print(f"Text: {sample['text']}")
        run_single(runner, args.model, sample["text"], temperature=0.0, use_schema=False)
        run_single(runner, args.model, sample["text"], temperature=1.0, use_schema=False)

    # 3. 5 Brand-New Samples NOT in the dataset
    print_banner("PART 3: 5 NEW UNSEEN SAMPLES (Schema-Constrained, T=0.0)")
    for sample in NEW_SAMPLES_5:
        print(f"\n--- {sample['id']}: {sample['description']} ---")
        print(f"Text: {sample['text']}")
        print(
            f"Expected -> category={sample['expected_category']}, "
            f"sentiment={sample['expected_sentiment']}, "
            f"entities={sample['expected_entities']}"
        )
        if "note" in sample:
            print(f"Note     -> {sample['note']}")
        run_single(runner, args.model, sample["text"], temperature=0.0, use_schema=True)

    print_banner("LIVE TESTS COMPLETED")


if __name__ == "__main__":
    main()
