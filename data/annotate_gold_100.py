"""
100 Değerlendirme Örneği İçin Gold Entity (Altın Varlık) Etiketleyici.
Kural: Modelin sistem promptundaki 6 varlık sınıfı esas alınmıştır:
- Kişi (Kişi adları, aktörler, tarihi figürler)
- Kurum / Şirket / Marka (Üniversite, spor kulübü, e-ticaret sitesi, marka)
- Yer / Lokasyon (Şehir, ülke, cadde, tiyatro)
- Ürün / Cihaz / Model (Ütü, matkap, tost makinesi, telefon, model no)
- Film / Eser (Film adı, dizi, roman, şarkı)
"""

import json
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parent / "benchmark_200.json"

GOLD_ANNOTATIONS = {
    1: ["kargo"],
    2: ["cilt"],
    3: ["koltuk"],
    4: ["film"],
    5: ["film"],
    6: ["dikiş makinesi"],
    7: ["film"],
    8: ["Trabzon", "Karadeniz Teknik Üniversitesi"],
    9: ["De Niro"],
    10: ["doğal gaz"],
    11: ["ütü"],
    12: ["dizi", "film"],
    13: ["bebek arabası"],
    14: ["parfüm"],
    15: ["sinema salonu"],
    16: ["Denzel", "Russel Crowe"],
    17: ["Yarından Sonra"],
    18: ["telefon"],
    19: ["film"],
    20: ["film"],
    21: ["cihaz"],
    22: ["İngiltere", "Her Majesty's Theatre"],
    23: ["Oscar"],
    24: ["matkap"],
    25: ["Uma Thurman"],
    26: ["film"],
    27: ["İstanbul", "İzmir"],
    28: ["tost makinesi"],
    29: ["İngiltere", "The Avengers", "Tara"],
    30: ["film"],
    31: ["roman", "hepsiburada.com"],
    32: ["film"],
    33: [],
    34: ["ayakkabı"],
    35: ["deri ceket"],
    36: ["Sırp", "Kosova", "kolo"],
    37: ["801 tk a+"],
    38: ["termometre"],
    39: ["Domain Name Sistemi", "İnternet"],
    40: ["kitap"],
    41: ["süpürge"],
    42: ["Sid", "Diego"],
    43: ["hepsiburada"],
    44: ["sapan"],
    45: ["Muhammed Ali", "Vietnam", "ABD"],
    46: ["Brezilya", "pasaport"],
    47: [],
    48: ["mikatur"],
    49: ["cihaz", "lazer"],
    50: ["duş başlığı", "hortum"],
    51: [],
    52: ["bisiklet", "pil"],
    53: ["film"],
    54: ["ürün"],
    55: ["davlumbaz"],
    56: ["araba", "usb"],
    57: ["oyun hamuru"],
    58: ["Romero", "film"],
    59: ["mouse", "bloody", "v3 gun 3"],
    60: ["film"],
    61: ["film"],
    62: ["otel"],
    63: ["Suriye Caddesi"],
    64: ["film"],
    65: ["telefon"],
    66: ["film"],
    67: ["yağmur sensörü", "silecek"],
    68: [],
    69: ["telefon", "kamera"],
    70: ["ayakkabı"],
    71: ["platin elektrot"],
    72: ["müze"],
    73: ["pilates topu", "el pompası", "hepsiburada"],
    74: ["Osmanlı İmparatorluğu"],
    75: ["Gençlerbirliği", "Engin Baytar"],
    76: ["Hacettepe Üniversitesi", "Ankara Devlet Konservatuvarı"],
    77: ["Kardemir Karabükspor"],
    78: ["sinema", "film"],
    79: ["Edirne", "Kırklareli"],
    80: ["Kolordu Adli Müşavirliği"],
    81: ["Dünya Savaşı"],
    82: ["saat", "kargo"],
    83: [],
    84: ["yalıtım süngeri"],
    85: ["penye"],
    86: ["hepsiburada"],
    87: ["korku filmi"],
    88: ["Karayip Korsanları", "Türkiye"],
    89: ["apartman"],
    90: ["film"],
    91: ["otel"],
    92: ["tencere", "plastik"],
    93: ["palmiye ağacı"],
    94: ["Hristiyan"],
    95: ["film"],
    96: ["ıslak mendil", "can bebe"],
    97: ["kulaklık"],
    98: ["biberon"],
    99: ["okul korosu", "kilise korosu"],
    100: ["köpek maması"]
}


def apply_annotations():
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    annotated_count = 0
    total_entities = 0

    for item in dataset:
        sid = item["id"]
        if sid in GOLD_ANNOTATIONS:
            item["gold_entities"] = GOLD_ANNOTATIONS[sid]
            annotated_count += 1
            total_entities += len(GOLD_ANNOTATIONS[sid])
        else:
            if "gold_entities" not in item:
                item["gold_entities"] = []

    with open(DATA_PATH, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    print(f"✅ {annotated_count} örnek başarıyla etiketlendi!")
    print(f"📊 Toplam Altın Varlık (Gold Entity) Sayısı: {total_entities}")
    print(f"📏 Örnek Başına Ortalama Varlık: {total_entities / annotated_count:.2f}")


if __name__ == "__main__":
    apply_annotations()
