"""
Yeni Duygu ve Kategori Benchmark Veri Seti Oluşturucu.
HuggingFace 'winvoker/turkish-sentiment-analysis-dataset' üzerinden
200 adet yüksek kaliteli, net duygu ve kategori etiketli örnek çeker.

Kategoriler:
- urun: E-ticaret ürün değerlendirmeleri (Hepsiburada vb.)
- sinema: Film ve sinema eleştirileri (HUMIR)
- ansiklopedi: Vikipedi ansiklopedik bilgilendirici cümleleri

Duygu Etiketleri:
- olumlu: Net pozitif kullanıcı görüşleri
- olumsuz: Net negatif eleştiri ve şikayetler
- notr: Duygu içermeyen objektif ansiklopedi metinleri
"""

import json
import random
import re
import shutil
from pathlib import Path
from datasets import load_dataset
from collections import Counter

DATA_DIR = Path(__file__).resolve().parent
BENCHMARK_PATH = DATA_DIR / "benchmark_200.json"
BACKUP_PATH = DATA_DIR / "benchmark_200_old_news.json"


def clean_text(text: str) -> str:
    """Metni gereksiz boşluklardan ve satır sonlarından arındırır."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def build_benchmark(seed: int = 42):
    random.seed(seed)

    # 1. Eski veri setini yedekle
    if BENCHMARK_PATH.exists() and not BACKUP_PATH.exists():
        shutil.copy2(BENCHMARK_PATH, BACKUP_PATH)
        print(f"📦 Eski haber veri seti yedeklendi: {BACKUP_PATH.name}")

    print("⏳ HuggingFace'ten 'winvoker/turkish-sentiment-analysis-dataset' yükleniyor...")
    ds = load_dataset("winvoker/turkish-sentiment-analysis-dataset", split="train")

    pool_urun_pos = []
    pool_urun_neg = []
    pool_sinema_pos = []
    pool_sinema_neg = []
    pool_wiki_notr = []

    print("🔍 Örnekler filtreleniyor ve havuzlar oluşturuluyor...")
    for item in ds:
        raw_text = item.get("text", "")
        text = clean_text(raw_text)
        length = len(text)
        
        # Çok kısa veya aşırı uzun olmayan metinleri seç (60-320 karakter)
        if not (60 <= length <= 320):
            continue

        dataset_src = item.get("dataset")
        label = item.get("label")

        if dataset_src == "urun_yorumlari":
            if label == "Positive":
                pool_urun_pos.append(text)
            elif label == "Negative":
                pool_urun_neg.append(text)
        elif dataset_src == "HUMIR":
            if label == "Positive":
                pool_sinema_pos.append(text)
            elif label == "Negative":
                pool_sinema_neg.append(text)
        elif dataset_src == "wiki":
            if label == "Notr":
                pool_wiki_notr.append(text)

    # Havuzları karıştır
    random.shuffle(pool_urun_pos)
    random.shuffle(pool_urun_neg)
    random.shuffle(pool_sinema_pos)
    random.shuffle(pool_sinema_neg)
    random.shuffle(pool_wiki_notr)

    # Hedef örnek sayıları:
    # Toplam 200 örnek
    # Kategori: 70 urun, 70 sinema, 60 ansiklopedi
    # Duygu: 70 olumlu, 70 olumsuz, 60 notr
    selected_samples = []

    # 35 Ürün Olumlu
    for text in pool_urun_pos[:35]:
        selected_samples.append({
            "text": text,
            "category": "urun",
            "sentiment": "olumlu"
        })

    # 35 Ürün Olumsuz
    for text in pool_urun_neg[:35]:
        selected_samples.append({
            "text": text,
            "category": "urun",
            "sentiment": "olumsuz"
        })

    # 35 Sinema Olumlu
    for text in pool_sinema_pos[:35]:
        selected_samples.append({
            "text": text,
            "category": "sinema",
            "sentiment": "olumlu"
        })

    # 35 Sinema Olumsuz
    for text in pool_sinema_neg[:35]:
        selected_samples.append({
            "text": text,
            "category": "sinema",
            "sentiment": "olumsuz"
        })

    # 60 Ansiklopedi Nötr
    for text in pool_wiki_notr[:60]:
        selected_samples.append({
            "text": text,
            "category": "ansiklopedi",
            "sentiment": "notr"
        })

    # Tüm örnekleri rastgele sırala
    random.shuffle(selected_samples)

    # ID'leri ata
    for idx, item in enumerate(selected_samples, 1):
        item["id"] = idx
        # Alan sırasını standartlaştır
        selected_samples[idx - 1] = {
            "id": idx,
            "text": item["text"],
            "category": item["category"],
            "sentiment": item["sentiment"]
        }

    # JSON'a kaydet
    with open(BENCHMARK_PATH, "w", encoding="utf-8") as f:
        json.dump(selected_samples, f, ensure_ascii=False, indent=2)

    cat_counts = Counter(x["category"] for x in selected_samples)
    sent_counts = Counter(x["sentiment"] for x in selected_samples)

    print("\n" + "=" * 50)
    print("✅ YENİ BENCHMARK VERİ SETİ BAŞARIYLA OLUŞTURULDU!")
    print(f"📁 Dosya: {BENCHMARK_PATH}")
    print(f"📊 Toplam Örnek: {len(selected_samples)}")
    print(f"🏷️  Kategori Dağılımı: {dict(cat_counts)}")
    print(f"🎭 Duygu Dağılımı:    {dict(sent_counts)}")
    print("=" * 50)


if __name__ == "__main__":
    build_benchmark()
