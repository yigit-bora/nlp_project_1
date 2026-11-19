"""
Ana CLI giriş noktası.
Çoklu model ve sıcaklık stres testi pipeline'ı.

Kullanım:
  # Tüm modeller için 2 örnekle hızlı doğrulama testi:
  python main.py --samples 2

  # Tek bir model için 2 örnekle test:
  python main.py --model qwen2.5:1.5b --samples 2

  # Tüm modeller, tam veri seti (200 örnek):
  python main.py

  # Sadece grafikleri yeniden üret (model çalıştırmadan):
  python main.py --skip-eval
"""

import argparse
import sys
from pathlib import Path
from typing import List
import pandas as pd

from config import MODELS, MODEL, RESULTS_DIR, FIGURES_DIR, get_model_slug
from src.ollama_runner import OllamaRunner
from src.pipeline import EvaluationPipeline
from src.visualizer import generate_all_visualizations


def update_master_summary() -> None:
    """Tüm modellerin ve ortamların özet CSV dosyalarını tek bir ana dosyada birleştirir."""
    dfs = []
    for m in MODELS:
        slug = get_model_slug(m)
        # Ortama özel klasörleri tara; yoksa geriye dönük kök klasörden prompt_only olarak al
        found_mode = False
        for mode_name in ["prompt_only", "schema_constrained"]:
            sf = RESULTS_DIR / slug / mode_name / "evaluation_summary.csv"
            if sf.exists():
                try:
                    df = pd.read_csv(sf)
                    if not df.empty:
                        if "mode" not in df.columns:
                            df["mode"] = mode_name
                        if "setting" not in df.columns:
                            df["setting"] = mode_name
                        dfs.append(df)
                        found_mode = True
                except Exception:
                    pass

        legacy_sf = RESULTS_DIR / slug / "evaluation_summary.csv"
        if legacy_sf.exists():
            # Eğer prompt_only alt klasörü yoksa legacy dosyasını prompt_only olarak ekle
            prompt_sf = RESULTS_DIR / slug / "prompt_only" / "evaluation_summary.csv"
            if not prompt_sf.exists():
                try:
                    df = pd.read_csv(legacy_sf)
                    if not df.empty:
                        if "mode" not in df.columns:
                            df["mode"] = "prompt_only"
                        if "setting" not in df.columns:
                            df["setting"] = "prompt_only"
                        dfs.append(df)
                except Exception:
                    pass

    if dfs:
        master_df = pd.concat(dfs, ignore_index=True).drop_duplicates()
        master_path = RESULTS_DIR / "all_models_summary.csv"
        master_df.to_csv(master_path, index=False, encoding="utf-8")
        print(f"📊 Birleşik özet tablosu güncellendi: {master_path}")


def main():
    parser = argparse.ArgumentParser(description="Çoklu LLM Değerlendirme Pipeline'ı")
    parser.add_argument("--samples", type=int, default=None, help="Test örnek sayısı (örn: 2, None=tümü)")
    parser.add_argument("--models", nargs="+", default=None, help="Çalıştırılacak model listesi (örn: qwen2.5:1.5b llama3.2:latest)")
    parser.add_argument("--model", type=str, default=None, help="Tek bir model çalıştırmak için kısayol")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["prompt_only", "schema_constrained", "both"],
        default="prompt_only",
        help="Deneysel ortam: 'prompt_only' (format parametresiz), 'schema_constrained' (format=JSON_SCHEMA), veya 'both'"
    )
    parser.add_argument("--skip-eval", action="store_true", help="Model çalıştırmayı atla, sadece mevcut sonuçlardan grafikleri üret")
    args = parser.parse_args()

    print("╔══════════════════════════════════════════════════════════╗")
    print("║     ÇOKLU LLM YAPILANDIRILMIŞ ÇIKTI DEĞERLENDİRME        ║")
    print("╚══════════════════════════════════════════════════════════╝")

    runner = OllamaRunner()
    if not runner.is_healthy():
        print("❌ Ollama çalışmıyor! 'ollama serve' komutuyla başlatın.")
        sys.exit(1)

    installed_models = runner.list_models()
    print(f"✅ Ollama aktif. Yüklü modeller: {installed_models}")

    # Hedef modelleri belirle
    if args.model:
        target_models = [args.model]
    elif args.models:
        target_models = args.models
    else:
        target_models = MODELS

    modes = ["prompt_only", "schema_constrained"] if args.mode == "both" else [args.mode]

    print(f"🎯 Hedef modeller: {target_models}")
    print(f"🧪 Deneysel Ortam(lar): {modes}")
    if args.samples:
        print(f"⚡ Örnek sınırı: Sıcaklık adımı başına {args.samples} örnek")

    # 1. Değerlendirme adımı
    if not args.skip_eval:
        for idx, m in enumerate(target_models, 1):
            for current_mode in modes:
                print("\n" + "=" * 65)
                print(f"🔄 [{idx}/{len(target_models)}] MODEL: {m} | ORTAM: {current_mode.upper()}")
                print("=" * 65)
                
                pipeline = EvaluationPipeline(model=m, sample_limit=args.samples, mode=current_mode)
                pipeline.run()

    # 2. Grafik üretimi
    print("\n" + "=" * 65)
    print("🎨 GRAFİKLER KONTROL EDİLİYOR VE GÜNCELLENİYOR")
    print("=" * 65)
    for m in target_models:
        slug = get_model_slug(m)
        for current_mode in modes:
            model_csv = RESULTS_DIR / slug / current_mode / "evaluation_summary.csv"
            model_fig_dir = FIGURES_DIR / slug / current_mode
            if not model_csv.exists() and current_mode == "prompt_only":
                model_csv = RESULTS_DIR / slug / "evaluation_summary.csv"
                model_fig_dir = FIGURES_DIR / slug
            if model_csv.exists():
                print(f"📈 [{m} - {current_mode}] Grafikler üretiliyor -> {model_fig_dir}")
                generate_all_visualizations(summary_csv=model_csv, output_dir=model_fig_dir)
            else:
                print(f"⚠️ [{m} - {current_mode}] İçin özet CSV henüz bulunamadı ({model_csv})")

    # 3. Master birleşik özet
    update_master_summary()

    print("\n" + "═" * 65)
    print("✨ TÜM İŞLEMLER TAMAMLANDI!")
    print(f"📁 Sonuçlar ana dizini: {RESULTS_DIR}/")
    print(f"📊 Grafikler ana dizini: {FIGURES_DIR}/")
    print("═" * 65)


if __name__ == "__main__":
    main()
