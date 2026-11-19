# qwen2.5:1.5b — Schema-Constrained (Ollama `format = JSON_SCHEMA`)

- **Model Slug:** `qwen2.5_1.5b` | **Quantization:** `Q4_K_M` | **Seed:** `42` | **Num Ctx:** `4096`
- **Ortam:** `schema_constrained` | **Örneklem:** Sıcaklık başına `100` örnek (Toplam `400` çıkarım)
- **Ham Veri:** [schema_constrained/evaluation_summary.csv](./schema_constrained/evaluation_summary.csv) | [schema_constrained/detailed_records.csv](./schema_constrained/detailed_records.csv)

## 1. Format Disiplini, Şema Geçerliliği ve Hız

| Sıcaklık ($T$) | Ham JSON (`pure_json_rate`) | Temizlenmiş JSON (`syntax_valid_rate`) | Şema Geçerliliği (`schema_valid_rate`) | Ort. Entity / Örnek | Ort. TPS |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 0.1 | %98.00 | %98.00 | %98.00 | 2.40 | 163.78 |
| 0.4 | %96.00 | %96.00 | %96.00 | 2.42 | 150.66 |
| 0.7 | %98.00 | %98.00 | %98.00 | 2.19 | 151.85 |
| 1.0 | %100.00 | %100.00 | %100.00 | 2.20 | 154.47 |

## 2. Sınıflandırma Doğruluğu (Accuracy) ve Macro F1

| Sıcaklık ($T$) | Strict Kategori Acc | Parsed Kategori Acc | Kategori Macro F1 | Strict Duygu Acc | Parsed Duygu Acc | Duygu Macro F1 |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0.1 | %55.00 | %56.12 | %54.70 | %75.00 | %76.53 | %75.96 |
| 0.4 | %54.00 | %56.25 | %54.97 | %74.00 | %77.08 | %76.67 |
| 0.7 | %54.00 | %55.10 | %53.85 | %72.00 | %73.47 | %73.46 |
| 1.0 | %57.00 | %57.00 | %55.80 | %72.00 | %72.00 | %71.88 |

## 3. Varlık Çıkarımı (Entity Extraction) — Exact & Partial Match

| Sıcaklık ($T$) | Exact Precision | Exact Recall | Exact F1 | Partial Precision | Partial Recall | Partial F1 |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0.1 | %20.83 | %38.46 | **%27.03** | %34.17 | %63.08 | **%44.32** |
| 0.4 | %20.25 | %37.69 | **%26.34** | %32.64 | %60.77 | **%42.47** |
| 0.7 | %21.00 | %35.38 | **%26.36** | %32.88 | %55.38 | **%41.26** |
| 1.0 | %20.91 | %35.38 | **%26.29** | %32.27 | %54.62 | **%40.57** |
