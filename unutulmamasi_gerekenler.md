# 📌 NLP Proje I: Unutulmaması Gerekenler ve Yapılacaklar Listesi (To-Do)

Bu doküman, hocanın proje şartnamesinde (**NLP Project I: Structured Text Classification & Information Extraction**) zorunlu kıldığı, teslim ve savunmada doğrudan değerlendirilecek tüm eksik maddeleri ve çalışma adımlarını içerir.

---

## 🎯 1. Deneysel Ortam 2: Schema-Constrained Tam Testlerin Koşulması
> **Durum:** ✅ **TAMAMLANDI!** Tüm 3 model (`qwen2.5:1.5b`, `llama3.2:latest`, `llama3.1:latest`) için `schema_constrained` ortamında 100 örnek × 4 sıcaklık (1.200 çıkarım) koşuldu ve `results/all_models_summary.csv` + karşılaştırma grafikleri (`prompt_vs_schema_comparison.png`) üretildi.

* [x] **Tüm Modeller İçin Schema-Constrained Benchmark'ının Koşulması:**
  * Modeller: `llama3.1:latest`, `llama3.2:latest`, `qwen2.5:1.5b`
  * Sıcaklıklar: $T \in [0.1, 0.4, 0.7, 1.0]$
  * Örnek sayısı: Sıcaklık başına 100 örnek (Model başına 400 çıkarım, toplam 1.200 çıkarım)
  * Komut:
    ```bash
    ./venv/bin/python main.py --mode schema_constrained --samples 100
    ```
* [x] **İki Ortamın Karşılaştırma Tablolarının Çıkarılması:**
  * `results/all_models_summary.csv` içinde 24 satırlık tam tablo (3 model × 4 sıcaklık × 2 ortam) oluşturuldu ve `reports/figures/ortak_grafikler/prompt_vs_schema_comparison.png` çizildi.

---

## 🏷️ 2. Entity Gold Labels (Varlık Altın Etiketleri) ve Entity F1 Skoru
> **Durum:** ✅ **TAMAMLANDI!** 100 örnek etiketlendi, Exact ve Partial F1 metrikleri kodlandı ve tüm modeller için ayrı ayrı grafikler üretildi.

* [x] **100 Değerlendirme Örneğinde Elle Entity Etiketleme (Manual Annotation):**
  * `data/benchmark_200.json` içindeki ilk 100 örneğe toplam **130 altın varlık** (`gold_entities`) eklendi.
* [x] **Etiketleme Kurallarının (Annotation Guidelines) Yazılması:**
  * Model promptundaki 6 varlık sınıfı (`ürün, marka, film, kişi, kurum, yer`) esas alındı.
* [x] **Türkçe Karakter Uyumlu Exact ve Partial Entity F1 Hesaplama:**
  * `src/metrics.py` içine `is_entity_match()`, `calculate_entity_metrics()` ve `calculate_entity_f1()` eklendi.
  * Hem Exact Match hem de Partial Match Precision, Recall ve F1 skorları hesaplandı.
* [x] **Ayrı Ayrı Grafikler Üretildi:**
  * Her model için ayrı `entity_f1_exact_vs_temperature.png` ve `entity_f1_partial_vs_temperature.png` çizildi.
  * 3 modeli kıyaslayan `multi_model_exact_f1.png` ve `multi_model_partial_f1.png` oluşturuldu.

---

## 📊 3. Sınıflandırma Metriklerine F1 Skoru (Macro F1) Eklenmesi
> **Durum:** ✅ **TAMAMLANDI!**

* [x] `src/metrics.py` içine `calculate_macro_f1` (Scikit-Learn tabanlı) eklendi.
* [x] Kategori ve Duygu için `category_macro_f1` ve `sentiment_macro_f1` değerleri özet tablolara kaydedildi.

---

## 💡 4. Rapor ve Sunumdaki Kritik Tartışma (Discussion) Sorusu
> **Durum:** ✅ **VERİ VE ARGÜMAN HAZIR!** ([sunum_notlari.md](file:///home/yigit/Desktop/NLP%20project/sunum_notlari.md) Soru A4 güncellendi, 4. adımda PDF rapora eklenecek.)
Hocanın şartnamesindeki kritik soru:
> *"Does schema-constrained decoding change task accuracy, not just format validity? Explain why or why not."*
> *(Şema kısıtlı decoding, sadece format geçerliliğini mi düzeltir yoksa görev doğruluğunu da değiştirir mi? Neden?)*

* [x] **Argüman Hazırlığı (Gerçek Deney Verileriyle):**
  * **Evet, görev doğruluğunu da değiştirir — hem olumlu hem olumsuz yönlerde:**
    * `enum` kısıtı sayesinde `llama3.1` Category Macro F1 **%43.8–67.8 → %91.7–91.8**'e, `llama3.2` ise **%3.1–4.2 → %48.9–50.8**'e yükseldi.
    * Ancak logit maskeleme serbest otoregresif üretimi kısıtladığı için `llama3.1` Entity Partial F1 hafifçe düştü (**%64.4 → %61.5** @ T=0.1) ve `llama3.2` Sentiment Accuracy geriledi (**%69.0 → %64.3** @ T=0.1).

---

## 🎬 5. Canlı Test (Live Test) Senaryosu ve 5 Yeni Örnek Hazırlığı
> **Durum:** ✅ **TAMAMLANDI!** `demo_live.py` yazıldı ve hem `qwen2.5:1.5b` hem `llama3.1:latest` ile doğrulandı.
Videoda ekranda kesintisiz (uncut) gösterilecek canlı test scripti (`demo_live.py`):

* [x] **Ortam Karşılaştırması:** Aynı örnek üzerinde Prompt-only vs Schema-constrained çıktısı canlı gösteriliyor.
* [x] **Sıcaklık Karşılaştırması:** Aynı örnek üzerinde `T=0.0` ve `T=1.0` farkı canlı gösteriliyor.
* [x] **5 Yeni Örnek (Veri setinde olmayan):** `NEW-1`..`NEW-5` (`NEW-5` ağır ironi/sarkazm tuzağı; `llama3.1:latest` ve `qwen2.5:1.5b` gerçekten `"olumlu"` diyerek yanılıyor).

---

## 🎤 6. Savunma Soruları (10 Soru) ve Canlı Deney
> **Durum:** ✅ **TAMAMLANDI!** 10 soru seçildi, derin bilimsel yanıtları ve İngilizce sunum metinleri [sunum_notlari.md](file:///home/yigit/Desktop/NLP%20project/sunum_notlari.md) içine yazıldı. Canlı deney scripti [demo_defense_experiments.py](file:///home/yigit/Desktop/NLP%20project/src/demo_defense_experiments.py) kodlandı ve başarıyla test edildi!

* [x] **Bölüm A (4 Soru - Zorunlu):**
  * A1: Kod satırı silme analizi (`metrics.py:38` Türkçe 'İ'/'I' Unicode combining dot ablation testi).
  * A2: AI önerisi reddi (Heuristic JSON auto-repair / `json_repair` bilimsel benchmark'ı bozduğu için reddedildi).
  * A3: Şaşırtıcı sonuç (Qwen 2.5'te sıcaklık artışının markdown ezberini kırıp saf JSON'ı %9'dan %31'e çıkarması).
  * A4: Format parametreli vs parametresiz sonuç farkları (Prompt-only vs Logit-maskelenmiş GBNF gramer decoding).
* [x] **Bölüm B-G (Kullanıcının Seçtiği 6 Soru):**
  * **B1 (Bölüm B):** Bir sonraki token seçimi: N-gram (Markov zinciri, sabit pencere, n-1 geçmiş) vs Transformer (Global self-attention, bağlamsal vektör uzayı ve softmax logit dağılımı).
  * **C1 (Bölüm C):** Sıcaklık artarsa parse başarısı ve accuracy nasıl değişir? (Entropi artışı, Llama 3.2'nin gevezelikle 256 token limitini aşıp %88'e düşmesi, Qwen'in ezberi kırması).
  * **D4 (Bölüm D):** Daha küçük (1B/1.5B) veya daha büyük (8B) model kullanılsaydı sonuçlar nasıl değişirdi? (Bizim projedeki Qwen 2.5 1.5B vs Llama 3.1 8B somut verileri: format disiplini %9'dan %100'e, F1 skoru %42'den %64'e yükseldi).
  * **E1 (Bölüm E - Canlı Deney):** Bir metnin Türkçe versiyonu neden İngilizcesinden daha fazla token içerir? (BPE sözlüklerinin İngilizce ağırlıklı olması, köklerin parçalanması ve eklemeli Türkçe morfolojisi - `demo_defense_experiments.py` ile canlı ispatlı).
  * **F2 (Bölüm F):** Few-shot örneklerin sırasını değiştirmek sonucu değiştirir mi? (Evet: Recency Bias / Yakınlık Önyargısı, son örneğin dikkat ağırlığını domine etmesi ve sınıf dengesizliği etkisi).
  * **G2 (Bölüm G):** Yalnızca 10 etiketli örnek olsaydı hangisi daha iyi performans gösterirdi: Klasik model (TF-IDF/BoW) mi yoksa Few-shot LLM mi? (Few-shot LLM açık ara kazanır: Klasik model 10 örnekte aşırı öğrenip [overfitting] sıfıra yakın genelleme yaparken, LLM devasa pre-training dünya bilgisi ve in-context learning ile geneller).
* [x] **CANLI DENEY (Videoda Gösterilecek Script Hazır!):**
  * `src/demo_defense_experiments.py` çalıştırılarak terminalde hem E1/E2 (Türkçe vs İngilizce token sayısı ve BPE parçalanması), hem A1 ('İSTANBUL' Unicode combining dot hatası), hem de D4/D1 model boyutu ve bellek hesabı canlı gösterilmektedir.


---

## 📄 7. Teslim Dokümanları
* [x] **Teknik Rapor (PDF, Max 4 Sayfa):**
  * [NLP_Project_1_Report.pdf](file:///home/yigit/Desktop/NLP%20project/reports/NLP_Project_1_Report.pdf) olarak 4 sayfa halinde üretildi ve görsel/sayfa kontrolü yapıldı (Ollama `0.35.0`, `Q4_K_M`, `seed=42`, annotation & Türkçe matching kuralları, 2 ortamın 4 zorunlu metriği, grafikler ve Discussion bölümü dahil).
* [ ] **Jupyter Notebook (`nlp_project_1.ipynb`):**
  * Tüm adımları baştan sona anlatan, açıklamalı, temiz ve çalıştırılabilir notebook.
* [ ] **YouTube Video Kaydı:**
  * Max 30 dakika, "Unlisted" (Gizli değil, Liste Dışı).
  * Açıklamada 3 bölüm zaman damgası (`00:00 Code walkthrough`, `... Live tests`, `... Defense questions`).
  * Yüz en az başta görünür, ses ve ekran net, canlı testler kesintisiz.
