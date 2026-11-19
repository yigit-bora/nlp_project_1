# qwen2.5:1.5b — Prompt-Only (Ollama `format` parametresi KAPALI)

- **Model Slug:** `qwen2.5_1.5b` | **Quantization:** `Q4_K_M` | **Seed:** `42` | **Num Ctx:** `4096`
- **Ortam:** `prompt_only` | **Örneklem:** Sıcaklık başına `100` örnek (Toplam `400` çıkarım)
- **Ham Veri:** [prompt_only/evaluation_summary.csv](./prompt_only/evaluation_summary.csv) | [prompt_only/detailed_records.csv](./prompt_only/detailed_records.csv)

## 1. Format Disiplini, Şema Geçerliliği ve Hız

| Sıcaklık ($T$) | Ham JSON (`pure_json_rate`) | Temizlenmiş JSON (`syntax_valid_rate`) | Şema Geçerliliği (`schema_valid_rate`) | Ort. Entity / Örnek | Ort. TPS |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 0.1 | %9.00 | %97.00 | %97.00 | 2.54 | 145.71 |
| 0.4 | %18.00 | %99.00 | %99.00 | 2.65 | 143.70 |
| 0.7 | %31.00 | %99.00 | %99.00 | 2.41 | 144.05 |
| 1.0 | %27.00 | %99.00 | %99.00 | 2.46 | 143.11 |

## 2. Sınıflandırma Doğruluğu (Accuracy) ve Macro F1

| Sıcaklık ($T$) | Strict Kategori Acc | Parsed Kategori Acc | Kategori Macro F1 | Strict Duygu Acc | Parsed Duygu Acc | Duygu Macro F1 |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0.1 | %6.00 | %55.67 | %54.87 | %6.00 | %75.26 | %75.20 |
| 0.4 | %11.00 | %57.58 | %56.28 | %15.00 | %75.76 | %76.03 |
| 0.7 | %19.00 | %57.58 | %56.07 | %25.00 | %71.72 | %71.40 |
| 1.0 | %13.00 | %56.57 | %55.34 | %18.00 | %60.61 | %59.86 |

## 3. Varlık Çıkarımı (Entity Extraction) — Exact & Partial Match

| Sıcaklık ($T$) | Exact Precision | Exact Recall | Exact F1 | Partial Precision | Partial Recall | Partial F1 |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0.1 | %17.72 | %34.62 | **%23.44** | %31.10 | %60.77 | **%41.15** |
| 0.4 | %20.00 | %40.77 | **%26.84** | %31.32 | %63.85 | **%42.03** |
| 0.7 | %19.09 | %35.38 | **%24.80** | %30.71 | %56.92 | **%39.89** |
| 1.0 | %17.48 | %33.08 | **%22.87** | %30.49 | %57.69 | **%39.89** |
