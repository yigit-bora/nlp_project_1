# llama3.1:latest — Prompt-Only (Ollama `format` parametresi KAPALI)

- **Model Slug:** `llama3.1_latest` | **Quantization:** `Q4_K_M` | **Seed:** `42` | **Num Ctx:** `4096`
- **Ortam:** `prompt_only` | **Örneklem:** Sıcaklık başına `100` örnek (Toplam `400` çıkarım)
- **Ham Veri:** [prompt_only/evaluation_summary.csv](./prompt_only/evaluation_summary.csv) | [prompt_only/detailed_records.csv](./prompt_only/detailed_records.csv)

## 1. Format Disiplini, Şema Geçerliliği ve Hız

| Sıcaklık ($T$) | Ham JSON (`pure_json_rate`) | Temizlenmiş JSON (`syntax_valid_rate`) | Şema Geçerliliği (`schema_valid_rate`) | Ort. Entity / Örnek | Ort. TPS |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 0.1 | %100.00 | %100.00 | %100.00 | 1.40 | 28.03 |
| 0.4 | %100.00 | %100.00 | %100.00 | 1.43 | 27.90 |
| 0.7 | %100.00 | %100.00 | %100.00 | 1.64 | 28.03 |
| 1.0 | %100.00 | %100.00 | %100.00 | 1.42 | 27.39 |

## 2. Sınıflandırma Doğruluğu (Accuracy) ve Macro F1

| Sıcaklık ($T$) | Strict Kategori Acc | Parsed Kategori Acc | Kategori Macro F1 | Strict Duygu Acc | Parsed Duygu Acc | Duygu Macro F1 |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0.1 | %90.00 | %90.00 | %67.84 | %81.00 | %81.00 | %80.95 |
| 0.4 | %90.00 | %90.00 | %67.77 | %80.00 | %80.00 | %79.73 |
| 0.7 | %89.00 | %89.00 | %45.18 | %81.00 | %81.00 | %81.03 |
| 1.0 | %86.00 | %86.00 | %43.79 | %77.00 | %77.00 | %75.97 |

## 3. Varlık Çıkarımı (Entity Extraction) — Exact & Partial Match

| Sıcaklık ($T$) | Exact Precision | Exact Recall | Exact F1 | Partial Precision | Partial Recall | Partial F1 |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0.1 | %48.57 | %52.31 | **%50.37** | %62.14 | %66.92 | **%64.44** |
| 0.4 | %44.76 | %49.23 | **%46.89** | %60.84 | %66.92 | **%63.74** |
| 0.7 | %42.07 | %53.08 | **%46.94** | %51.83 | %65.38 | **%57.82** |
| 1.0 | %47.18 | %51.54 | **%49.26** | %59.15 | %64.62 | **%61.76** |
