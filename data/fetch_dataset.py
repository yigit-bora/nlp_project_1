"""
Veri Seti Çekme Scripti.
HuggingFace savasy/ttc4900 Türkçe haber veri setinden
her kategoriden dengeli şekilde 200 örnek çeker ve JSON olarak kaydeder.

Metinler çok uzun olduğu için LLM'e gönderilecek kısmı max 500 karakter ile kırpılır.
Ancak orijinal metin de saklanır (analiz amaçlı).
"""

import json
import random
from pathlib import Path
from datasets import load_dataset


CATEGORY_MAP = {
    0: "siyaset",
    1: "dunya",
    2: "ekonomi",
    3: "kultur",
    4: "saglik",
    5: "spor",
    6: "teknoloji"
}

MAX_TEXT_LENGTH = 500  # LLM'e gönderilecek metnin max karakter uzunluğu
SAMPLES_PER_CATEGORY = 29  # 7 * 29 = 203, yuvarlak 200 civarı
TOTAL_TARGET = 200


def fetch_and_save(output_path: str = "data/benchmark_200.json", seed: int = 42):
    random.seed(seed)
    
    print("HuggingFace'ten savasy/ttc4900 veri seti çekiliyor...")
    ds = load_dataset("savasy/ttc4900", split="train")
    
    # Her kategoriden dengeli örnekleme
    category_buckets = {i: [] for i in range(7)}
    for item in ds:
        category_buckets[item["category"]].append(item)
    
    selected = []
    for cat_id in range(7):
        bucket = category_buckets[cat_id]
        random.shuffle(bucket)
        sampled = bucket[:SAMPLES_PER_CATEGORY]
        for item in sampled:
            text_full = item["text"].strip()
            # LLM prompt'u için kırpılmış metin
            text_short = text_full[:MAX_TEXT_LENGTH]
            if len(text_full) > MAX_TEXT_LENGTH:
                # Son kelimenin ortasından kesmemek için son boşlukta kes
                last_space = text_short.rfind(" ")
                if last_space > MAX_TEXT_LENGTH * 0.7:
                    text_short = text_short[:last_space]
            
            selected.append({
                "id": len(selected) + 1,
                "text": text_short,
                "category": CATEGORY_MAP[cat_id],
            })
    
    # Fazla olanları kes
    random.shuffle(selected)
    selected = selected[:TOTAL_TARGET]
    
    # ID'leri yeniden ata
    for i, item in enumerate(selected):
        item["id"] = i + 1
    
    # Kaydet
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(selected, f, ensure_ascii=False, indent=2)
    
    # Dağılım raporu
    from collections import Counter
    dist = Counter(item["category"] for item in selected)
    print(f"\n✅ {len(selected)} örnekli benchmark veri seti oluşturuldu: {output_path}")
    print(f"📊 Kategori dağılımı: {dict(dist)}")
    print(f"📏 Ortalama metin uzunluğu: {sum(len(x['text']) for x in selected) / len(selected):.0f} karakter")


if __name__ == "__main__":
    fetch_and_save()
