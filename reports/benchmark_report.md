# 3 Model Karşılaştırmalı LLM Yapılandırılmış Çıktı ve Şema Kısıtlı Çıkarım Raporu

**Güncelleme Tarihi:** 19 Kasım 2026  
**Ollama Sürümü:** `0.35.0` | **Seed:** `42` | **Context (`num_ctx`):** `4096` | **Max Tokens (`num_predict`):** `256`  
**Test Edilen Modeller (Tüm Modeller `Q4_K_M` Kuantizasyon):**
1. `llama3.1:latest` (8.0B parametre, `Q4_K_M`)
2. `llama3.2:latest` (3.2B parametre, `Q4_K_M`)
3. `qwen2.5:1.5b` (1.5B parametre, `Q4_K_M`)

**Veri Seti:** HuggingFace `winvoker/turkish-sentiment-analysis-dataset` (`data/benchmark_200.json`, ilk 100 örnekte toplam 130 altın varlık etiketi)  
**Deneysel Tasarım:** 2 Ortam (`prompt_only` vs. `schema_constrained`) $\times$ 3 Model $\times$ 4 Sıcaklık ($T \in \{0.1, 0.4, 0.7, 1.0\}$) $\times$ 100 Örnek = **2.400 çıkarım**.

---

## 1. Model Bazlı Ayrı Tablolar (İlk Bakışta Erişim)

Her modelin altında `prompt_only` ve `schema_constrained` için ayrı klasörler ve okunabilir özet tablo dosyaları yer almaktadır:

* **`llama3.1:latest` (8B):**
  * Prompt-Only Tabloları: [results/llama3.1_latest/prompt_only_tables.md](../results/llama3.1_latest/prompt_only_tables.md)
  * Schema-Constrained Tabloları: [results/llama3.1_latest/schema_constrained_tables.md](../results/llama3.1_latest/schema_constrained_tables.md)
* **`llama3.2:latest` (3B):**
  * Prompt-Only Tabloları: [results/llama3.2_latest/prompt_only_tables.md](../results/llama3.2_latest/prompt_only_tables.md)
  * Schema-Constrained Tabloları: [results/llama3.2_latest/schema_constrained_tables.md](../results/llama3.2_latest/schema_constrained_tables.md)
* **`qwen2.5:1.5b` (1.5B):**
  * Prompt-Only Tabloları: [results/qwen2.5_1.5b/prompt_only_tables.md](../results/qwen2.5_1.5b/prompt_only_tables.md)
  * Schema-Constrained Tabloları: [results/qwen2.5_1.5b/schema_constrained_tables.md](../results/qwen2.5_1.5b/schema_constrained_tables.md)
* **Birleşik CSV (24 Satır):** [results/all_models_summary.csv](../results/all_models_summary.csv)

---

## 2. Deneysel Ortam 1: Prompt-Only (Ollama `format` Parametresi Kapalı)

| Model | $T$ | Ham JSON | Temizlenmiş JSON | Şema Geçerli | Parsed Kat. Acc (Macro F1) | Parsed Duygu Acc (Macro F1) | Entity Exact F1 | Entity Partial F1 | Ort. TPS |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`llama3.1:latest`** | 0.1 | %100.0 | %100.0 | %100.0 | %90.0 (%67.8) | %81.0 (%81.0) | **%50.4** | **%64.4** | 28.0 |
| | 0.4 | %100.0 | %100.0 | %100.0 | %90.0 (%67.8) | %80.0 (%79.7) | %46.9 | %63.7 | 27.9 |
| | 0.7 | %100.0 | %100.0 | %100.0 | %89.0 (%45.2) | %81.0 (%81.0) | %46.9 | %57.8 | 28.0 |
| | 1.0 | %100.0 | %100.0 | %100.0 | %86.0 (%43.8) | %77.0 (%76.0) | %49.3 | %61.8 | 27.4 |
| **`qwen2.5:1.5b`** | 0.1 | %9.0 | %97.0 | %97.0 | %55.7 (%54.9) | %75.3 (%75.2) | %23.4 | %41.2 | **145.7** |
| | 0.4 | %18.0 | %99.0 | %99.0 | %57.6 (%56.3) | %75.8 (%76.0) | %26.8 | %42.0 | **143.7** |
| | 0.7 | %31.0 | %99.0 | %99.0 | %57.6 (%56.1) | %71.7 (%71.4) | %24.8 | %39.9 | **144.1** |
| | 1.0 | %27.0 | %99.0 | %99.0 | %56.6 (%55.3) | %60.6 (%59.9) | %22.9 | %39.9 | **143.1** |
| **`llama3.2:latest`** | 0.1 | %0.0 | %100.0 | %100.0 | %32.0 (%4.2) | %69.0 (%53.2) | %30.8 | %42.4 | 76.0 |
| | 0.4 | %1.0 | %93.0 | %93.0 | %35.5 (%4.2) | %65.6 (%51.0) | %31.9 | %42.7 | 75.1 |
| | 0.7 | %3.0 | %91.0 | %89.0 | %32.2 (%3.4) | %62.2 (%36.2) | %31.0 | %41.2 | 74.3 |
| | 1.0 | %2.0 | %89.0 | %88.0 | %32.6 (%3.1) | %62.9 (%36.6) | %26.9 | %40.0 | 73.3 |

---

## 3. Deneysel Ortam 2: Schema-Constrained (Ollama `format = JSON_SCHEMA`)

| Model | $T$ | Ham JSON | Temizlenmiş JSON | Şema Geçerli | Parsed Kat. Acc (Macro F1) | Parsed Duygu Acc (Macro F1) | Entity Exact F1 | Entity Partial F1 | Ort. TPS |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`llama3.1:latest`** | 0.1 | %100.0 | %100.0 | %100.0 | **%92.0 (%91.7)** | %82.0 (%82.1) | %46.2 | %61.5 | 26.1 |
| | 0.4 | %100.0 | %100.0 | %100.0 | **%92.0 (%91.7)** | %82.0 (%82.1) | %45.0 | %59.0 | 24.5 |
| | 0.7 | %100.0 | %100.0 | %100.0 | **%92.0 (%91.7)** | **%83.0 (%83.0)** | %46.6 | %57.5 | 24.6 |
| | 1.0 | %100.0 | %100.0 | %100.0 | **%92.0 (%91.8)** | **%83.0 (%83.0)** | %45.2 | %55.9 | 24.6 |
| **`qwen2.5:1.5b`** | 0.1 | %98.0 | %98.0 | %98.0 | %56.1 (%54.7) | %76.5 (%76.0) | %27.0 | %44.3 | **163.8** |
| | 0.4 | %96.0 | %96.0 | %96.0 | %56.3 (%55.0) | %77.1 (%76.7) | %26.3 | %42.5 | **150.7** |
| | 0.7 | %98.0 | %98.0 | %98.0 | %55.1 (%53.9) | %73.5 (%73.5) | %26.4 | %41.3 | **151.9** |
| | 1.0 | **%100.0** | **%100.0** | **%100.0** | %57.0 (%55.8) | %72.0 (%71.9) | %26.3 | %40.6 | **154.5** |
| **`llama3.2:latest`** | 0.1 | %98.0 | %98.0 | %98.0 | %58.2 (%49.7) | %64.3 (%49.8) | %27.9 | %39.0 | 73.5 |
| | 0.4 | **%100.0** | **%100.0** | **%100.0** | %59.0 (%50.2) | %63.0 (%48.7) | %28.9 | %39.8 | 72.9 |
| | 0.7 | %99.0 | %99.0 | %99.0 | %59.6 (%48.9) | %61.6 (%47.3) | %29.4 | %39.8 | 72.0 |
| | 1.0 | %99.0 | %99.0 | %99.0 | %59.6 (%50.8) | %62.6 (%48.3) | %29.2 | %41.5 | 71.3 |

---

## 4. Tartışma: Schema-Constrained Decoding Görev Doğruluğunu Değiştirir mi?

**Evet — yalnızca format geçerliliğini değil, görev doğruluğunu da (hem olumlu hem olumsuz yönde) belirgin biçimde değiştirir:**

1. **Enum Kısıtının Pozitif Etkisi (`category` ve `sentiment`):**
   * `llama3.1:latest` modelinde `enum` kısıtı sayesinde şema dışı etiket uydurma ortadan kalkmış; Kategori Macro F1 skoru **%43.8–%67.8** bandından **%91.7–%91.8** seviyesine sıçramıştır.
   * `llama3.2:latest` modelinde serbest metin kategori üretimi engellendiği için Parsed Kategori Doğruluğu **%32–%35.5**'ten **%58.2–%59.6**'ya, Macro F1 ise **%3.1–%4.2**'den **%48.9–%50.8**'e çıkmıştır.
2. **Logit Maskelemenin Yan Etkisi (Entity F1 ve Küçük Model Duygusu):**
   * `llama3.1:latest`'te Entity Exact F1 (%50.4 $\rightarrow$ %46.2) ve Partial F1 (%64.4 $\rightarrow$ %61.5), `llama3.2:latest`'te ise Duygu Doğruluğu (%69.0 $\rightarrow$ %64.3 @ $T=0.1$) hafifçe düşmüştür. Zorunlu şema sırası ve GBNF token maskeleme, modelin doğal otoregresif olasılık dağılımını daraltarak özellikle açık uçlu varlık çıkarımı sınırlarında sapmaya yol açabilmektedir.
3. **Küçük Modellerde Neden %100 Yerine %96–%99 Çıktı?**
   * Şemadaki `entities` dizisinde `maxItems` kısıtı bulunmadığından, `qwen2.5:1.5b` (8 kayıt) ve `llama3.2:latest` (4 kayıt) tekrar döngüsüne girerek `num_predict=256` sınırında kesilmiş ve JSON'u kapatamamıştır.

---

## 5. Grafik Klasör Yapısı

* **Ortak Karşılaştırma Grafikleri (`reports/figures/ortak_grafikler/`):**
  * Prompt-Only vs Schema-Constrained Özeti: `reports/figures/ortak_grafikler/prompt_vs_schema_comparison.png`
  * Prompt-Only Çoklu Model F1: `reports/figures/ortak_grafikler/multi_model_exact_f1.png`, `reports/figures/ortak_grafikler/multi_model_partial_f1.png`
  * Schema-Constrained Çoklu Model F1: `reports/figures/ortak_grafikler/multi_model_exact_f1_schema_constrained.png`, `reports/figures/ortak_grafikler/multi_model_partial_f1_schema_constrained.png`
* **Model Bazlı Simetrik Klasörler (`dashboard.png`, `json_success_vs_temperature.png`, `accuracy_vs_temperature.png`, `entity_f1_exact_vs_temperature.png`, `entity_f1_partial_vs_temperature.png`):**
  * `reports/figures/<model_slug>/prompt_only/`
  * `reports/figures/<model_slug>/schema_constrained/`
