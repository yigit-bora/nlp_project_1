# llama3.1:latest — Schema-Constrained (Ollama `format = JSON_SCHEMA`)

- **Model Slug:** `llama3.1_latest` | **Quantization:** `Q4_K_M` | **Seed:** `42` | **Num Ctx:** `4096`
- **Ortam:** `schema_constrained` | **Örneklem:** Sıcaklık başına `100` örnek (Toplam `400` çıkarım)
- **Ham Veri:** [schema_constrained/evaluation_summary.csv](./schema_constrained/evaluation_summary.csv) | [schema_constrained/detailed_records.csv](./schema_constrained/detailed_records.csv)

## 1. Format Disiplini, Şema Geçerliliği ve Hız

| Sıcaklık ($T$) | Ham JSON (`pure_json_rate`) | Temizlenmiş JSON (`syntax_valid_rate`) | Şema Geçerliliği (`schema_valid_rate`) | Ort. Entity / Örnek | Ort. TPS |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 0.1 | %100.00 | %100.00 | %100.00 | 1.43 | 26.13 |
| 0.4 | %100.00 | %100.00 | %100.00 | 1.41 | 24.49 |
| 0.7 | %100.00 | %100.00 | %100.00 | 1.45 | 24.55 |
| 1.0 | %100.00 | %100.00 | %100.00 | 1.49 | 24.55 |

## 2. Sınıflandırma Doğruluğu (Accuracy) ve Macro F1

| Sıcaklık ($T$) | Strict Kategori Acc | Parsed Kategori Acc | Kategori Macro F1 | Strict Duygu Acc | Parsed Duygu Acc | Duygu Macro F1 |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0.1 | %92.00 | %92.00 | %91.67 | %82.00 | %82.00 | %82.14 |
| 0.4 | %92.00 | %92.00 | %91.67 | %82.00 | %82.00 | %82.14 |
| 0.7 | %92.00 | %92.00 | %91.67 | %83.00 | %83.00 | %83.01 |
| 1.0 | %92.00 | %92.00 | %91.77 | %83.00 | %83.00 | %83.01 |

## 3. Varlık Çıkarımı (Entity Extraction) — Exact & Partial Match

| Sıcaklık ($T$) | Exact Precision | Exact Recall | Exact F1 | Partial Precision | Partial Recall | Partial F1 |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0.1 | %44.06 | %48.46 | **%46.15** | %58.74 | %64.62 | **%61.54** |
| 0.4 | %43.26 | %46.92 | **%45.02** | %56.74 | %61.54 | **%59.04** |
| 0.7 | %44.14 | %49.23 | **%46.55** | %54.48 | %60.77 | **%57.45** |
| 1.0 | %42.28 | %48.46 | **%45.16** | %52.35 | %60.00 | **%55.91** |
