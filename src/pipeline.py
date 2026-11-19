"""
Değerlendirme Pipeline Orkestratörü.
Tek model, çoklu sıcaklık üzerinden stres testi.
JSON otomatik kurtarma YOK — bozuk çıktı = başarısızlık.
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd
from tqdm import tqdm

from config import (
    SYSTEM_PROMPT, USER_PROMPT_TEMPLATE,
    DATASET_PATH, RESULTS_DIR, FIGURES_DIR, MODEL, TEMPERATURES,
    VALID_CATEGORIES, JSON_SCHEMA, DEFAULT_SEED, get_model_slug
)
from src.ollama_runner import OllamaRunner
from src.cleaner import parse_llm_json
from src.metrics import calculate_accuracy, calculate_entity_f1, calculate_macro_f1, normalize


class EvaluationPipeline:
    def __init__(
        self,
        model: str = MODEL,
        temperatures: Optional[List[float]] = None,
        sample_limit: Optional[int] = None,
        results_dir: Optional[Path] = None,
        figures_dir: Optional[Path] = None,
        mode: str = "prompt_only",  # "prompt_only" veya "schema_constrained"
        seed: int = DEFAULT_SEED,
    ):
        self.model = model
        self.model_slug = get_model_slug(self.model)
        self.mode = mode
        self.seed = seed
        self.temperatures = temperatures or TEMPERATURES
        self.sample_limit = sample_limit
        self.runner = OllamaRunner()
        self.dataset = self._load_dataset()
        self.detailed_records: List[Dict[str, Any]] = []
        self.summary_records: List[Dict[str, Any]] = []

        # Modele ve ortama özel dizinler
        base_res = results_dir or RESULTS_DIR
        base_fig = figures_dir or FIGURES_DIR
        
        self.model_results_dir = base_res / self.model_slug
        self.model_figures_dir = base_fig / self.model_slug
        self.mode_results_dir = self.model_results_dir / self.mode
        self.mode_figures_dir = self.model_figures_dir / self.mode
        
        self.model_results_dir.mkdir(parents=True, exist_ok=True)
        self.model_figures_dir.mkdir(parents=True, exist_ok=True)
        self.mode_results_dir.mkdir(parents=True, exist_ok=True)
        self.mode_figures_dir.mkdir(parents=True, exist_ok=True)

    def _load_dataset(self) -> List[Dict[str, Any]]:
        with open(DATASET_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        if self.sample_limit:
            data = data[:self.sample_limit]
        return data

    def run(self) -> pd.DataFrame:
        print("=" * 65)
        print("🚀 LLM DEĞERLENDİRME PIPELINE'I BAŞLATILIYOR")
        print(f"   Model: {self.model}")
        print(f"   Ortam (Setting): {self.mode.upper()} " + 
              ("['format' parametresi KAPALI]" if self.mode == "prompt_only" else "['format' = JSON Schema]"))
        print(f"   Sıcaklıklar: {self.temperatures}")
        print(f"   Toplam Örnek: {len(self.dataset)}")
        print(f"   Seed: {self.seed}")
        print("=" * 65)

        print("\n⏳ Model VRAM'e yükleniyor (ısınma adımı)...")
        self.runner.warmup(self.model)
        print("✅ Model hazır.")

        for temp in self.temperatures:
            print(f"\n🌡️  Sıcaklık Testi: T = {temp}")
            self._run_temperature_step(temp)

        self._save_results()
        return pd.DataFrame(self.summary_records)

    def _run_temperature_step(self, temp: float):
        pure_json_ok = 0
        syntax_ok = 0
        schema_valid_ok = 0
        cat_preds_overall = []
        cat_preds_parsed = []
        cat_gts_overall = []
        cat_gts_parsed = []
        sent_preds_overall = []
        sent_preds_parsed = []
        sent_gts_overall = []
        sent_gts_parsed = []
        pred_entities_list = []
        gt_entities_list = []
        entity_counts = []
        tps_list = []

        # Format şeması seçimi (Deneysel Ortam 1: None vs Deneysel Ortam 2: JSON_SCHEMA)
        format_schema = JSON_SCHEMA if self.mode == "schema_constrained" else None

        pbar = tqdm(self.dataset, desc=f"{self.model} [{self.mode} T={temp}]", leave=True)
        for item in pbar:
            sid = item["id"]
            text = item["text"]
            gt_cat = item["category"]
            gt_sent = item.get("sentiment", "notr")
            gt_entities = item.get("gold_entities", [])

            # 1. Inference
            user_prompt = USER_PROMPT_TEMPLATE.format(text=text)
            telemetry = self.runner.generate(
                model=self.model,
                system_prompt=SYSTEM_PROMPT,
                user_prompt=user_prompt,
                temperature=temp,
                format_schema=format_schema,
                seed=self.seed
            )

            raw = telemetry["raw_text"]
            tps = telemetry["tps"]
            tps_list.append(tps)

            # 2. Çok metrikli bağımsız JSON ayrıştırma
            res = parse_llm_json(raw)

            if res.is_pure_json:
                pure_json_ok += 1
            if res.is_valid_syntax:
                syntax_ok += 1
            if res.is_schema_valid:
                schema_valid_ok += 1

            pred_cat = None
            pred_sent = None
            pred_entities = []
            cat_correct = False
            sent_correct = False

            if res.data:
                pred_cat = str(res.data.get("category", "")).strip()
                pred_sent = str(res.data.get("sentiment", "")).strip()
                raw_ents = res.data.get("entities", [])
                if isinstance(raw_ents, list):
                    pred_entities = [str(e) for e in raw_ents if isinstance(e, str)]

                cat_correct = (normalize(pred_cat) == normalize(gt_cat))
                sent_correct = (normalize(pred_sent) == normalize(gt_sent))

                cat_preds_parsed.append(pred_cat)
                cat_gts_parsed.append(gt_cat)
                sent_preds_parsed.append(pred_sent)
                sent_gts_parsed.append(gt_sent)

                # Katman 2: Strict / Uçtan Uca (Kurtarma olmadan, saf JSON kuralına uymayanlar FAIL)
                if res.is_pure_json:
                    cat_preds_overall.append(pred_cat)
                    sent_preds_overall.append(pred_sent)
                else:
                    cat_preds_overall.append("__JSON_FAIL__")
                    sent_preds_overall.append("__JSON_FAIL__")
            else:
                cat_preds_overall.append("__JSON_FAIL__")
                sent_preds_overall.append("__JSON_FAIL__")

            cat_gts_overall.append(gt_cat)
            sent_gts_overall.append(gt_sent)

            pred_entities_list.append(pred_entities)
            gt_entities_list.append(gt_entities if isinstance(gt_entities, list) else [])
            entity_counts.append(len(pred_entities))

            # Per-sample kayıt
            self.detailed_records.append({
                "model": self.model,
                "setting": self.mode,
                "mode": self.mode,
                "temperature": temp,
                "sample_id": sid,
                "gt_category": gt_cat,
                "pred_category": pred_cat,
                "strict_category_correct": (res.is_pure_json and cat_correct),
                "parsed_category_correct": cat_correct,
                "category_correct": (res.is_pure_json and cat_correct),
                "gt_sentiment": gt_sent,
                "pred_sentiment": pred_sent,
                "strict_sentiment_correct": (res.is_pure_json and sent_correct),
                "parsed_sentiment_correct": sent_correct,
                "sentiment_correct": (res.is_pure_json and sent_correct),
                "entity_count": len(pred_entities),
                "is_pure_json": res.is_pure_json,
                "is_valid_syntax": res.is_valid_syntax,
                "is_schema_valid": res.is_schema_valid,
                "format_type": res.format_type,
                "json_error": res.error,
                "tps": tps,
                "latency_sec": telemetry["latency_sec"],
                "eval_tokens": telemetry["eval_count"],
                "raw_output_preview": raw[:800] if raw else ""
            })

            pbar.set_postfix({
                "PureJSON": f"{(pure_json_ok / len(self.dataset))*100:.0f}%",
                "SchemaValid": f"{(schema_valid_ok / len(self.dataset))*100:.0f}%",
                "StrictCat": f"{(sum(1 for p, g in zip(cat_preds_overall, cat_gts_overall) if p != '__JSON_FAIL__' and normalize(p) == normalize(g)) / len(self.dataset) * 100):.0f}%"
            })

        # 3 Katmanlı Özet Hesaplama
        total = len(self.dataset)
        pure_json_rate = (pure_json_ok / total) * 100
        syntax_valid_rate = (syntax_ok / total) * 100
        schema_valid_rate = (schema_valid_ok / total) * 100

        # Başarı sayıları
        cat_parsed_correct = sum(1 for p, g in zip(cat_preds_parsed, cat_gts_parsed) if normalize(p) == normalize(g))
        cat_overall_correct = sum(1 for p, g in zip(cat_preds_overall, cat_gts_overall) if p != "__JSON_FAIL__" and normalize(p) == normalize(g))

        # 1. Parsed Doğruluk (Format bozulsa bile kurtarılanlar içindeki oran)
        parsed_cat_acc = (cat_parsed_correct / len(cat_preds_parsed) * 100) if cat_preds_parsed else 0.0

        # 2. Strict / Overall Doğruluk (Format bozulması = Başarısızlık sayılan oran)
        overall_cat_acc = (cat_overall_correct / total * 100) if total > 0 else 0.0

        sent_parsed_correct = sum(1 for p, g in zip(sent_preds_parsed, sent_gts_parsed) if normalize(p) == normalize(g))
        sent_overall_correct = sum(1 for p, g in zip(sent_preds_overall, sent_gts_overall) if p != "__JSON_FAIL__" and normalize(p) == normalize(g))

        parsed_sent_acc = (sent_parsed_correct / len(sent_preds_parsed) * 100) if sent_preds_parsed else 0.0
        overall_sent_acc = (sent_overall_correct / total * 100) if total > 0 else 0.0

        avg_tps = sum(tps_list) / len(tps_list) if tps_list else 0
        avg_entities = sum(entity_counts) / total
        ent_res = calculate_entity_f1(pred_entities_list, gt_entities_list)
        cat_macro_f1 = calculate_macro_f1(cat_preds_parsed, cat_gts_parsed)
        sent_macro_f1 = calculate_macro_f1(sent_preds_parsed, sent_gts_parsed)

        summary = {
            "model": self.model,
            "setting": self.mode,
            "mode": self.mode,
            "temperature": temp,
            "total_samples": total,
            "pure_json_rate": round(pure_json_rate, 2),
            "syntax_valid_rate": round(syntax_valid_rate, 2),
            "schema_valid_rate": round(schema_valid_rate, 2),
            "strict_category_accuracy": round(overall_cat_acc, 2),
            "parsed_category_accuracy": round(parsed_cat_acc, 2),
            "strict_sentiment_accuracy": round(overall_sent_acc, 2),
            "parsed_sentiment_accuracy": round(parsed_sent_acc, 2),
            # Geriye dönük uyumluluk alanları:
            "overall_category_accuracy": round(overall_cat_acc, 2),
            "overall_sentiment_accuracy": round(overall_sent_acc, 2),
            "category_accuracy": round(overall_cat_acc, 2),
            "sentiment_accuracy": round(overall_sent_acc, 2),
            "avg_entities_per_sample": round(avg_entities, 2),
            "mean_tps": round(avg_tps, 2),
            "exact_f1": ent_res["exact_f1"],
            "exact_precision": ent_res["exact_precision"],
            "exact_recall": ent_res["exact_recall"],
            "partial_f1": ent_res["partial_f1"],
            "partial_precision": ent_res["partial_precision"],
            "partial_recall": ent_res["partial_recall"],
            "category_macro_f1": cat_macro_f1,
            "sentiment_macro_f1": sent_macro_f1,
        }
        self.summary_records.append(summary)

        print(
            f"  📊 Sonuçlar [{self.mode} T={temp}]: "
            f"Saf JSON: %{pure_json_rate:.1f} | "
            f"Şema Geçerli: %{schema_valid_rate:.1f} | "
            f"Strict Kategori: %{overall_cat_acc:.1f} (Parsed: %{parsed_cat_acc:.1f}) | "
            f"Strict Duygu: %{overall_sent_acc:.1f} (Parsed: %{parsed_sent_acc:.1f}) | "
            f"Entity F1 (Exact/Partial): %{ent_res['exact_f1']:.1f}/%{ent_res['partial_f1']:.1f} | "
            f"TPS: {avg_tps:.0f}"
        )

    def _save_results(self):
        detailed_df = pd.DataFrame(self.detailed_records)
        summary_df = pd.DataFrame(self.summary_records)

        # 1. Ortama özel dizin (results/{model_slug}/{mode}/)
        detailed_path = self.mode_results_dir / "detailed_records.csv"
        summary_csv_path = self.mode_results_dir / "evaluation_summary.csv"
        summary_json_path = self.mode_results_dir / "evaluation_summary.json"

        detailed_df.to_csv(detailed_path, index=False, encoding="utf-8")
        summary_df.to_csv(summary_csv_path, index=False, encoding="utf-8")

        with open(summary_json_path, "w", encoding="utf-8") as f:
            json.dump(self.summary_records, f, ensure_ascii=False, indent=2)

        print(f"\n💾 [{self.model} - {self.mode}] Detaylı kayıtlar: {detailed_path}")
        print(f"💾 [{self.model} - {self.mode}] Özet metrikler:   {summary_csv_path}")

        # Otomatik olarak ortama özel grafikleri üret
        print(f"🎨 [{self.model} - {self.mode}] Grafikler oluşturuluyor...")
        from src.visualizer import generate_all_visualizations
        generate_all_visualizations(summary_csv=summary_csv_path, output_dir=self.mode_figures_dir)
        print(f"📊 [{self.model} - {self.mode}] Grafikler kaydedildi: {self.mode_figures_dir}")
