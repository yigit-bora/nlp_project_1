# llama3.2:latest — Prompt-Only (Ollama `format` parametresi KAPALI)

- **Model Slug:** `llama3.2_latest` | **Quantization:** `Q4_K_M` | **Seed:** `42` | **Num Ctx:** `4096`
- **Ortam:** `prompt_only` | **Örneklem:** Sıcaklık başına `100` örnek (Toplam `400` çıkarım)
- **Ham Veri:** [prompt_only/evaluation_summary.csv](./prompt_only/evaluation_summary.csv) | [prompt_only/detailed_records.csv](./prompt_only/detailed_records.csv)

## 1. Format Disiplini, Şema Geçerliliği ve Hız

| Sıcaklık ($T$) | Ham JSON (`pure_json_rate`) | Temizlenmiş JSON (`syntax_valid_rate`) | Şema Geçerliliği (`schema_valid_rate`) | Ort. Entity / Örnek | Ort. TPS |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 0.1 | %0.00 | %100.00 | %100.00 | 2.85 | 76.03 |
| 0.4 | %1.00 | %93.00 | %93.00 | 2.59 | 75.14 |
| 0.7 | %3.00 | %91.00 | %89.00 | 2.63 | 74.26 |
| 1.0 | %2.00 | %89.00 | %88.00 | 2.20 | 73.26 |

## 2. Sınıflandırma Doğruluğu (Accuracy) ve Macro F1

| Sıcaklık ($T$) | Strict Kategori Acc | Parsed Kategori Acc | Kategori Macro F1 | Strict Duygu Acc | Parsed Duygu Acc | Duygu Macro F1 |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0.1 | %0.00 | %32.00 | %4.24 | %0.00 | %69.00 | %53.18 |
| 0.4 | %1.00 | %35.48 | %4.22 | %1.00 | %65.59 | %50.98 |
| 0.7 | %1.00 | %32.22 | %3.43 | %0.00 | %62.22 | %36.18 |
| 1.0 | %0.00 | %32.58 | %3.08 | %1.00 | %62.92 | %36.60 |

## 3. Varlık Çıkarımı (Entity Extraction) — Exact & Partial Match

| Sıcaklık ($T$) | Exact Precision | Exact Recall | Exact F1 | Partial Precision | Partial Recall | Partial F1 |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0.1 | %22.46 | %49.23 | **%30.84** | %30.88 | %67.69 | **%42.41** |
| 0.4 | %23.94 | %47.69 | **%31.88** | %32.05 | %63.85 | **%42.67** |
| 0.7 | %23.19 | %46.92 | **%31.04** | %30.80 | %62.31 | **%41.22** |
| 1.0 | %21.36 | %36.15 | **%26.86** | %31.82 | %53.85 | **%40.00** |
