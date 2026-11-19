"""
Pipeline Konfigürasyonu.
Model, sıcaklık ızgarası, yollar ve katı sistem komutu.
"""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# Modeller
MODELS = ["qwen2.5:1.5b", "llama3.2:latest", "llama3.1:latest"]
MODEL = "qwen2.5:1.5b"  # Varsayılan model


def get_model_slug(model_name: str) -> str:
    """Model adını klasör ve dosya isimleri için güvenli hale getirir (örn: qwen2.5:1.5b -> qwen2.5_1.5b)."""
    return model_name.replace(":", "_").replace("/", "_")


# Sıcaklık stres adımları
TEMPERATURES = [0.1, 0.4, 0.7, 1.0]

# Yollar
DATASET_PATH = BASE_DIR / "data" / "benchmark_200.json"
RESULTS_DIR = BASE_DIR / "results"
REPORTS_DIR = BASE_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Sabit parametreler (Tekrarlanabilirlik / Reproducibility için)
DEFAULT_SEED = 42
DEFAULT_CONTEXT_LENGTH = 4096

# Geçerli kategori ve duygu listeleri
VALID_CATEGORIES = ["urun", "sinema", "ansiklopedi"]
VALID_SENTIMENTS = ["olumlu", "olumsuz", "notr"]

# Deneysel Ortam 2 için JSON Şeması (Ollama format parametresi)
JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "category": {
            "type": "string",
            "enum": VALID_CATEGORIES,
            "description": "Metnin ana kategorisi (urun, sinema, ansiklopedi)"
        },
        "sentiment": {
            "type": "string",
            "enum": VALID_SENTIMENTS,
            "description": "Metnin genel duygu tonu (olumlu, olumsuz, notr)"
        },
        "entities": {
            "type": "array",
            "items": {
                "type": "string"
            },
            "description": "Metinde geçen önemli varlıkların (ürün, marka, film, kişi, kurum, yer) listesi"
        }
    },
    "required": ["category", "sentiment", "entities"]
}

# Katı Sistem Komutu — modeli JSON formatına zorlayan prompt
SYSTEM_PROMPT = """Sen bir metin analiz yapay zekasısın. Görevin verilen Türkçe metni analiz etmektir.

MUTLAK KURALLAR:
1. Yanıtın SADECE ve SADECE geçerli bir JSON nesnesi olmalıdır.
2. JSON'un başında veya sonunda ASLA açıklama, selamlama, markdown etiketi veya yorum OLMAMALIDIR.
3. Yanıtın doğrudan { karakteri ile başlamalı ve } karakteri ile bitmelidir.

Üretmen gereken JSON formatı:
{
  "category": "kategorilerden biri",
  "sentiment": "olumlu veya olumsuz veya notr",
  "entities": ["varlık1", "varlık2"]
}

ALAN AÇIKLAMALARI:
- "category": Metnin türü. SADECE şunlardan biri olabilir: "urun", "sinema", "ansiklopedi"
- "sentiment": Metnin genel duygu tonu. SADECE şunlardan biri: "olumlu", "olumsuz", "notr"
- "entities": Metinde geçen önemli varlıkların (ürün, marka, film, kişi, kurum, yer) listesi. String dizisi olmalı.

ÖNEMLİ: Sadece JSON üret. Başka HİÇBİR ŞEY yazma. Yanıtına ASLA ```json veya markdown kod bloğu ekleme. Açıklama metni yazma. Yanıt doğrudan '{' karakteri ile başlamalı ve '}' karakteri ile bitmelidir."""

USER_PROMPT_TEMPLATE = """Aşağıdaki Türkçe metni analiz et:

{text}"""
