# llama3.2:latest — Schema-Constrained (Ollama `format = JSON_SCHEMA`)

- **Model Slug:** `llama3.2_latest` | **Quantization:** `Q4_K_M` | **Seed:** `42` | **Num Ctx:** `4096`
- **Ortam:** `schema_constrained` | **Örneklem:** Sıcaklık başına `100` örnek (Toplam `400` çıkarım)
- **Ham Veri:** [schema_constrained/evaluation_summary.csv](./schema_constrained/evaluation_summary.csv) | [schema_constrained/detailed_records.csv](./schema_constrained/detailed_records.csv)

## 1. Format Disiplini, Şema Geçerliliği ve Hız

| Sıcaklık ($T$) | Ham JSON (`pure_json_rate`) | Temizlenmiş JSON (`syntax_valid_rate`) | Şema Geçerliliği (`schema_valid_rate`) | Ort. Entity / Örnek | Ort. TPS |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 0.1 | %98.00 | %98.00 | %98.00 | 3.57 | 73.51 |
| 0.4 | %100.00 | %100.00 | %100.00 | 3.47 | 72.91 |
| 0.7 | %99.00 | %99.00 | %99.00 | 3.32 | 72.02 |
| 1.0 | %99.00 | %99.00 | %99.00 | 3.09 | 71.34 |

## 2. Sınıflandırma Doğruluğu (Accuracy) ve Macro F1

| Sıcaklık ($T$) | Strict Kategori Acc | Parsed Kategori Acc | Kategori Macro F1 | Strict Duygu Acc | Parsed Duygu Acc | Duygu Macro F1 |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0.1 | %57.00 | %58.16 | %49.66 | %63.00 | %64.29 | %49.79 |
| 0.4 | %59.00 | %59.00 | %50.16 | %63.00 | %63.00 | %48.72 |
| 0.7 | %59.00 | %59.60 | %48.87 | %61.00 | %61.62 | %47.28 |
| 1.0 | %59.00 | %59.60 | %50.81 | %62.00 | %62.63 | %48.34 |

## 3. Varlık Çıkarımı (Entity Extraction) — Exact & Partial Match

| Sıcaklık ($T$) | Exact Precision | Exact Recall | Exact F1 | Partial Precision | Partial Recall | Partial F1 |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0.1 | %19.05 | %52.31 | **%27.93** | %26.61 | %73.08 | **%39.01** |
| 0.4 | %19.88 | %53.08 | **%28.93** | %27.38 | %73.08 | **%39.83** |
| 0.7 | %20.48 | %52.31 | **%29.44** | %27.71 | %70.77 | **%39.83** |
| 1.0 | %20.71 | %49.23 | **%29.16** | %29.45 | %70.00 | **%41.46** |
