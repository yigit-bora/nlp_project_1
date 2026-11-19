# 🎤 Sunum ve Savunma Notları: Entity Extraction & ABSA

Bu doküman, video kaydı sırasında veya 15. haftadaki sözlü savunma sınavında Entity (Varlık Çıkarımı) ve Duygu Analizi konularında kullanabileceğin kritik argümanları ve mühendislik vizyonunu içerir.

---

## 1. Şirketler Neden Sadece Sınıflandırma ile Yetinmeyip Entity Çıkarır?

Sınıflandırma (Classification) bir metnin **durumunu** (örn: "olumsuz" veya "urun") söyler; ancak işletmeler için **tek başına değersizdir**. Çünkü durum aksiyon aldırmaz. 
**Entity (Varlık Çıkarımı)** ise aksiyonun **kime/neye** yöneltileceğini söyler.

### Gerçek Hayat Örnekleri:
* **E-Ticaret / Perakende (Kök Neden Analizi):**
  * *"Kulaklığın sesi harika ama Aras Kargo paketi ezmiş, 4 gün şubede bekletti."*
  * Sadece sınıflandırma yaparsan: Yorum = `Olumsuz`. (Üreticiye haksız ceza kesilir).
  * Entity çıkarırsan: `["Kulaklık", "Aras Kargo"]`. Sorunun üründe değil, lojistikte olduğu anlaşılır.
* **Akıllı Destek Yönlendirme (Smart Ticket Routing):**
  * *"Kadıköy şubesindeki ATM'den para çekerken kartım sıkıştı."*
  * Entity: `["Kadıköy", "ATM", "Kart Sıkışması"]` $\rightarrow$ Talep çağrı merkezinde beklemeden doğrudan Kadıköy ATM saha ekibine atanır.
* **Sağlık & Medikal Raporlar (Knowledge Graph):**
  * Serbest epikriz metninden `Hasta`, `Teşhis`, `İlaç` ve `Dozaj` entity'leri çekilerek ilişkisel veritabanına bağlanır.

---

## 2. Kritik İkilem: Bir Metinde İki Varlık da Eleştirilirse Suçlu Nasıl Anlaşılır?

### Problem (Düz Şemanın Sınırı):
Eğer müşteri hem kulaklıktan hem de kargodan şikayetçiyse (veya biri iyi, biri kötüyse), düz JSON şeması yetersiz kalır:
```json
{
  "sentiment": "olumsuz",
  "entities": ["Kulaklık", "Aras Kargo"]
}
```
Burada bir **"Atfetme (Attribution) Problemi"** doğar; tek bir global duygu değeri her iki entity'ye de yapışır.

### Çözüm Yolları:
1. **Aspect-Based Sentiment Analysis (ABSA) — İleri Düzey Şema:**
   Şema düz bir liste yerine iç içe (nested) nesnelere dönüştürülür:
   ```json
   {
     "category": "urun",
     "aspects": [
       {
         "entity": "Kulaklık",
         "target_feature": "Ses / Malzeme",
         "sentiment": "olumsuz",
         "reason": "Plastik kalitesi zayıf"
       },
       {
         "entity": "Aras Kargo",
         "target_feature": "Teslimat",
         "sentiment": "olumsuz",
         "reason": "Paket ezilmiş ve gecikmiş"
       }
     ]
   }
   ```
2. **Büyük Veri ve Korelasyon Analizi (İstatistiksel Ayrıştırma):**
   Tek bir yorumda suç ayrıştırılamasa bile, 10.000 yorum aggregate edildiğinde:
   * Kulaklığın geçtiği yorumların genel pozitiflik oranı (%85),
   * Aras Kargo'nun geçtiği yorumların negatiflik oranı (%90) karşılaştırılarak istatistiksel olarak asıl aksaklık kaynağı tespit edilir.

---

## 3. Biz Bu Projede Neden Düz Liste Kullandık?

1. **Küçük Parametreli Modellerin (1.5B / 3B) Sınırı:**
   Küçük yerel modeller düz bir `["a", "b"]` listesi üretirken bile düşük sıcaklıkta kısırdöngüye (repetition loop) girip token sınırına toslayabilmektedir. İç içe (nested) ABSA şeması verilseydi prompt-only modda syntax çöküşü tavan yapardı.
2. **Proje Hedefi:**
   Öncelikli amaç modelin temel JSON şeması sözleşmesine ve tiplerine uyumunu doğrulamaktır.

---

## 🌟 4. Sunumda / Savunmada Kullanılacak Altın Cümle (Golden Pitch - ABSA)

> *"Hocam, biz bu çalışmada metin düzeyinde (document-level) duygu ve düz varlık listesi kullandık. Gerçek dünya senaryolarında bir metinde birden fazla varlık farklı duygu tonlarıyla yer alabilir (örneğin ürünün beğenilip kargonun eleştirilmesi). Bu durumun çözümü, şemayı 'Aspect-Based Sentiment Analysis' (ABSA) yapısına dönüştürerek her entity'ye kendi bağımsız duygu etiketini bağlamaktır. Bizim kurduğumuz yapı, bu ileri analitik mimarinin temel basamağını teşkil etmektedir."*

---

## 🔬 5. Savunma Soruları İçin Kritik Çıkarım: Sıcaklık, Saf JSON ve Şema Geçerliliği Paradoksu
*(Bölüm A3 "Hangi sonuç sizi şaşırttı?" ve Bölüm C1 "Sıcaklık artarsa parse başarısı nasıl değişir?" sorularının tam cevabıdır.)*

### ❓ Paradoks / Gözlem:
* **Soru:** Sıcaklık arttıkça Llama 3.2'de şema geçerliliği %100'den %88'e düşerken; Qwen 2.5'te saf JSON üretme oranı neden %9'dan %31'e fırladı? Neden sıcaklık her modelde genel bir başarı çöküşüne yol açmadı?

### 🧠 Derin Mühendislik Açıklaması:

1. **İki Metriğin Baktığı Yerler Tamamen Farklıdır:**
   * **Saf JSON Başarısı (Raw JSON):** Modelin **dış kabuğuna / ilk tokenına** bakar. Model cevaba doğrudan `{` ile başlayıp `}` ile bitirmiş mi? (Markdown ```` ```json ```` çiti veya selamlama cümlesi var mı?)
   * **Şema Geçerliliği (Schema Validity):** Modelin **içeriğine ve nefesinin yetip yetmediğine** bakar. İstenen 3 zorunlu alan (`category`, `sentiment`, `entities`) eksiksiz basılıp tipleri doğru verilmiş mi?

2. **Qwen 2.5 (1.5B): Sıcaklık Ezberi Kırdı!**
   * **Düşük Sıcaklıkta ($T=0.1$):** Model tamamen deterministiktir. Pre-training'den gelen en baskın refleksini çalıştırır: *"JSON mu istendi? Hemen ```` ```json ```` açmalıyım!"* Bu yüzden çıktılarının %91'i markdown çitine hapsolur; saf JSON oranı **%9**'da kalır.
   * **Sıcaklık Arttıkça ($T=0.7 - 1.0$):** Olasılık dağılımı düzleşir (flattened logits). Modelin ilk token olarak ezberindeki ```` ``` ```` yerine doğrudan `{` karakterini seçme ihtimali devreye girer. Yani sıcaklık, küçük modelin **ezber refleksini kırıp doğrudan saf JSON üretme oranını %31'e çıkarmıştır.**

3. **Llama 3.2 (3B): Sıcaklık Gevezeliği Kışkırttı ve Token Sınırına Çarptı!**
   * **Düşük Sıcaklıkta ($T=0.1$):** Model zaten saf JSON üretemiyordu (%0, çünkü söze geveze bir chat modeli gibi *"Aşağıdaki metin..."* diye giriyordu). Fakat gevezeliği kısa tuttuğu için içerideki JSON'ı 50-60 token içinde kapatabiliyordu (**%100 şema geçerliydi**).
   * **Sıcaklık Arttıkça ($T=0.7 - 1.0$):** Yüksek sıcaklık yaratıcılığı ve gevezeliği kışkırttı. Model metin hakkında uzun uzun felsefe yapmaya, yorumlar yazmaya başladı. Gevezelik uzayınca `max_tokens: 256` sınırına tosladı! Ollama üretimi aniden kestiği için `entities` listesi yarım kaldı, kapanış parantezi `}` basılamadı ve **şema geçerliliği %100'den %88'e çakıldı.**

4. **Llama 3.1 (8B): Parametre Gücü ve Kusursuz Format Disiplini**
   * 8B parametre kapasitesi sayesinde $T=0.1$'den $1.0$'a kadar hem **%100 Saf JSON** hem de **%100 Şema Geçerliliği** korudu. Ne gevezelik yaptı ne de token sınırına takıldı.

### 🌟 Sunumda / Videoda Söylenecek Vurucu Cümle (Golden Pitch):
> *"Hocam, sıcaklık artışı iki farklı modelde taban tabana zıt iki etki yarattı:*
> *Qwen 2.5'te sıcaklık, modelin deterministik ```` ```json ```` ezberini kırıp **doğrudan saf JSON üretme oranını %9'dan %31'e fırlattı**.*
> *Llama 3.2'de ise sıcaklık, chat gevezeliğini kışkırtıp modelin 256 token sınırına çarpmasına yol açtı; bu yüzden kapanamayan yarım JSON'lar sebebiyle **şema doğruluğu %100'den %88'e geriledi**."*

---

## 🎯 6. Exact Match vs. Partial Match Varlık Çıkarımı Analizi
*(Ayrı grafikler gösterilirken videoda ve raporda anlatılacak temel bulgular)*

### 📈 Grafikler:
* **Exact Match (Birebir Eşleşme):** `entity_f1_exact_vs_temperature.png` ve `multi_model_exact_f1.png`
* **Partial Match (Kısmi Eşleşme):** `entity_f1_partial_vs_temperature.png` ve `multi_model_partial_f1.png`

### 🔍 Kritik Gözlem ve Bulgular:
1. **Neden İki Ayrı Grafik Çizdik?:**
   * Türkçe eklemeli bir dildir. Modeller varlığı doğru tespit etse bile sıklıkla çekim ekiyle birlikte üretir (Örn: `De Niro` yerine `de nironun`, `KTÜ` yerine `Karadeniz Teknik Üniversitesi kurulmasında`).
   * **Exact Match** grafiğinde harfi harfine eşitlik arandığı için modeller haksız yere cezalandırılır; F1 skorları %23 - %50 bandında görünür.
   * **Partial Match** grafiğinde ise modelin varlığı metinde gerçekten bulup bulmadığı (içerme ve kelime kesişimi) test edilir; F1 skoru **%64.4'e fırlamakta**, yakalama oranı (Recall) ise **%67'ye** ulaşmaktadır.

2. **Model Karşılaştırması:**
   * **Llama 3.1 (8B):** Hem Exact (%50.4) hem de Partial (%64.4) modda tartışmasız liderdir. Varlık sınırlarını (boundary detection) en temiz kesen modeldir.
   * **Qwen 2.5 (1.5B):** Exact modda %23 iken, Partial modda %42'ye fırlar. Çünkü küçük model varlığı tek başına değil, yanındaki sıfat veya fiille birlikte (`'de nironun oyunculuğu'` gibi n-gram öbeği olarak) çekmeye meyillidir.
   * **Llama 3.2 (3B):** Düşük sıcaklıkta %42.4 Partial F1 verirken, sıcaklık arttıkça token sınırına çarpması yüzünden Recall %67'den %53'e gerilemiştir.

### 🌟 Videoda Söylenecek Altın Cümle:
> *"Hocam, bu iki grafiği özellikle ayrı ayrı çizdik. Çünkü Exact Match grafiği modelin varlık sınırını harfiyen kesebilme (boundary detection) yeteneğini gösterirken; Partial Match grafiği modelin metindeki gerçek bilgiyi yakalama (semantic recall) gücünü göstermektedir. Görüldüğü üzere modellerimiz anlamsal olarak %67'ye varan yüksek bir yakalama başarısına sahiptir."*

---

## 🛡️ 7. 10 Savunma Sorusu ve Kusursuz Cevapları (Defense Questions Master Guide)

Hocanın şartnamesine göre videoda **Bölüm A'nın 4 sorusunun tamamı** + **B, C, D, E, F, G bölümlerinden seçilen 1'er soru (toplam 10 soru)** yanıtlanmalı ve **en az 1 soru canlı deneyle gösterilmelidir**.

Aşağıda hem video kaydı (İngilizce/Türkçe) hem de 15. haftadaki sözlü sınav için seçtiğimiz 10 sorunun bilimsel analizi ve konuşma metinleri yer almaktadır.

---

### 📌 BÖLÜM A: Your Code and Results (4 Soru - Tamamı Zorunlu)

#### ❓ Soru A1: Choose one line from your JSON cleaning/parsing code or your evaluation code. Explain what it does and show what happens when it is removed.
* **Seçilen Kod Satırı (`src/metrics.py` - Satır 14):**
  ```python
  return s.replace("İ", "i").replace("I", "ı").lower()
  ```
* **Mühendislik Açıklaması:**
  Standart Python `str.lower()` fonksiyonu Türkçe yerel karakterleri tanımaz. `'İSTANBUL'.lower()` çalıştırıldığında Python birleştirici nokta (`combining dot: \u0307`) üreterek `'i̇stanbul'` sonucunu verir; `'I'` harfini ise `'ı'` yerine `'i'` yapar.
* **Silinirse Ne Olur? (Ablation Sonucu):**
  Eğer bu satır silinip standart `text.lower()` kullanılırsa, büyük harfle yazılmış tüm Türkçe varlıklar (`"İSTANBUL"` $\neq$ `"istanbul"`) eşleşme testinden geçemez. Model varlığı doğru bulsa bile **Exact Match Precision, Recall ve F1 skorları sıfıra çakılır.**
* **Canlı Deney Scripti:** `python src/demo_defense_experiments.py` çalıştırıldığında bu satırın Unicode farkı terminalde canlı olarak gösterilmektedir.
* 🎙️ **Videoda Söylenecek Cümle (English Pitch):**
  > *"I chose line 14 from `src/metrics.py`: `return s.replace('İ', 'i').replace('I', 'ı').lower()`. In standard Python, `str.lower()` incorrectly converts uppercase Turkish 'İ' into 'i' followed by a Unicode combining dot `\u0307`. If we remove this line and use standard lowercasing, `'İSTANBUL'.lower() == 'istanbul'` evaluates to `False`. As a result, our Exact Match F1 score would artificially collapse to zero for capitalized Turkish entities."*

---

#### ❓ Soru A2: Did your AI assistant suggest anything that you decided not to use? Why not?
* **Reddedilen Öneri:**
  Yapay zeka asistanı, parse hatalarını önlemek için eksik kapanış parantezlerini veya tırnak işaretlerini regex ile tamamlayan bir **"Heuristic JSON Auto-Repair / Fuzzy Parsing"** kütüphanesi (`json_repair` veya gevşek otomatik tamamlama fonksiyonu) eklemeyi önerdi.
* **Neden Reddettik? (Mühendislik Vizyonu):**
  Bu çalışma bir dil modelinin yapılandırılmış veri üretme (structured generation) ve şema sözleşmesine uyma disiplinini ölçen bir **bilimsel benchmark** çalışmasıdır. Eğer modelin yarım bıraktığı veya sentaksını bozduğu JSON'ları arkadan süpürüp yamayan bir otomatik tamirci (auto-repair) koysaydık:
  1. Yüksek sıcaklıklarda modelin yaşadığı token tükenmesi ve halüsinasyonları maskelemiş olurduk.
  2. Prompt-only mod ile Schema-constrained mod arasındaki gerçek yapısal farkı ölçemezdik.
  Bu yüzden modelin sentaksını olduğu gibi değerlendiren katı bir doğrulama (`cleaner.py` ve `validate_schema`) mekanizmasında karar kıldık.
* 🎙️ **Videoda Söylenecek Cümle (English Pitch):**
  > *"My AI assistant suggested implementing an automated JSON heuristic repair module—such as `json_repair`—to automatically close missing brackets and fix truncated quotes. I explicitly rejected this. This project is a scientific benchmark evaluating how faithfully local LLMs follow structured contracts. Auto-repairing malformed JSON would mask the model's true failure modes and hallucinations, artificially inflating our validity metrics and obscuring the true benefits of schema-constrained decoding."*

---

#### ❓ Soru A3: Which of your results surprised you? How do you explain it?
* **Şaşırtıcı Sonuç (The Temperature Paradox):**
  Beklentimiz, sıcaklık ($T$) arttıkça rastgeleliğin artması ve tüm modellerde saf JSON başarısının düşmesi yönündeydi. Ancak **Qwen 2.5 (1.5B)** modelinde tam tersi oldu: Saf JSON üretme oranı $T=0.1$'de **%9** iken, $T=1.0$'da **%31'e fırladı!**
* **Mühendislik Açıklaması:**
  * Qwen 2.5, düşük sıcaklıkta ($T=0.1$) pre-training'den gelen baskın ezberine hapsoluyor: JSON görünce otomatikman markdown kod çiti (```` ```json ````) açıyordu. Bu yüzden çıktılarının %91'i markdown bloğu oluyordu.
  * Sıcaklık $1.0$'a çıktığında logit olasılıkları düzleşti (flattened distribution). Model bu ezber refleksini kırıp doğrudan `{` karakteri ile başlamayı başardı.
  * Llama 3.2 (3B) ise tam aksine, yüksek sıcaklıkta gevezeliğe kapılıp 256 token sınırına çarptı ve şema geçerliliği %100'den %88'e düştü.
* 🎙️ **Videoda Söylenecek Cümle (English Pitch):**
  > *"The most surprising result was the temperature paradox between Qwen 2.5 and Llama 3.2. We expected higher temperature to degrade formatting universally. However, in Qwen 2.5, increasing temperature from 0.1 to 1.0 boosted pure JSON success from 9% to 31%! At low temperature, Qwen was deterministically trapped in its training reflex of wrapping output in markdown fences (` ```json `). Higher entropy flattened the logits, allowing it to break this reflex and emit raw JSON directly. In contrast, Llama 3.2 suffered from verbosity at high temperature, hitting our 256-token limit and dropping schema validity from 100% to 88%."*

---

#### ❓ Soru A4: Explain the difference between the results you obtained with and without the format parameter.
* **Mühendislik Analizi:**
  * **Without format (Prompt-only):**
    Model serbest metin (free-form autoregressive) modunda çalışır. JSON üretmesi için yalnızca prompt talimatına güveniriz. Modeller sıklıkla sohbet selamlama cümleleri ekler, markdown çiti kullanır (`format_type: markdown_fence`) veya token sınırına çarpıp yarım JSON bırakır.
  * **With format (`format: json_schema`):**
    Ollama, derleme zamanında sağladığımız JSON şemasını bir **GBNF Grameri (Context-Free Grammar)** haline getirir. Model her token üretirken (inference anında), gramere uymayan tüm geçersiz tokenların logitleri maskelenir ($-\infty$ yapılır).
* **Somut Deney Sonuçlarımız (100 Örnek × 4 Sıcaklık):**
  1. **Saf JSON ve Şema Geçerliliği:**
     * `qwen2.5:1.5b`: Saf JSON oranı **%9–%31**'den **%96–%100**'e sıçradı.
     * `llama3.2:latest`: Saf JSON oranı **%0–%3**'ten **%98–%100**'e çıktı.
     * `llama3.1:latest`: Her iki ortamda da **%100** kaldı.
     * *Kritik Nüans:* Küçük modellerde (1.5B/3B) %100 yerine %96–99 görünmesinin tek sebebi şemada `maxItems` kısıtı olmayan `"entities"` dizisinin tekrar döngüsüne girip `num_predict=256` token sınırında kesilmesidir.
  2. **Görev Doğruluğu (Task Accuracy & F1) — Sadece Format Değil, İçerik de Değişti!**
     * `llama3.1:latest` (8B): Category Macro F1 **%43.8–%67.8**'den **%91.7–%91.8**'e fırladı (çünkü `enum` kısıtı şema dışı kategori uydurmayı engelledi); Sentiment Accuracy **%77–%81**'den **%82–%83**'e yükseldi. Ancak Entity Exact/Partial F1 çok hafif düştü (**%50.4 / %64.4** → **%46.2 / %61.5** @ T=0.1).
     * `llama3.2:latest` (3B): Category Accuracy (Parsed) **%32–%35.5**'ten **%58.2–%59.6**'ya, Macro F1 **%3.1–%4.2**'den **%48.9–%50.8**'e yükseldi; fakat Sentiment Accuracy **%69.0**'dan **%64.3**'e (@ T=0.1) geriledi (logit maskeleme küçük modelin olasılık dengesini kaydırabiliyor).
  3. **Hız (TPS) Etkisi:** `qwen2.5:1.5b`'de markdown çitleri yok olduğu için TPS arttı (~144 → 151–164 TPS); `llama3.1:latest`'te ise her token adımındaki GBNF gramer maskeleme maliyeti nedeniyle TPS hafifçe düştü (~28.0 → 24.5–26.1 TPS).
* 🎙️ **Videoda Söylenecek Cümle (English Pitch):**
  > *"Without the format parameter, smaller models waste tokens on markdown fences or chat preambles—Qwen 2.5 achieved only 9–31% pure JSON and Llama 3.2 only 0–3%. With `format: json_schema`, Ollama compiles the schema into a GBNF grammar and masks invalid token logits. Pure JSON immediately jumped to 96–100% for Qwen 2.5, 98–100% for Llama 3.2, and remained 100% for Llama 3.1 (the remaining 1–4% errors in small models came solely from `entities` repetition loops hitting our 256-token limit). Crucially, constrained decoding also changed task accuracy: by enforcing strict `enum` constraints, Llama 3.1's Category Macro F1 surged from 44–68% to 91.7% and Llama 3.2's from 4% to 50%, though entity F1 dipped slightly due to altered autoregressive token probabilities."*

---

### 📌 BÖLÜM B: Language Models and Attention

#### ❓ Soru B1: Explain how the model selects the next token, using the difference between an n-gram model and a Transformer.
* **Mühendislik Açıklaması:**
  1. **N-Gram Modelleri (Markov Yaklaşımı):**
     * N-gram modelleri Markov varsayımıyla çalışır: Bir sonraki kelimenin olasılığı sadece kendinden önceki $n-1$ kelimeye bağlıdır:
       $$P(w_t | w_1, \dots, w_{t-1}) \approx P(w_t | w_{t-n+1}, \dots, w_{t-1}) = \frac{C(w_{t-n+1}, \dots, w_t)}{C(w_{t-n+1}, \dots, w_{t-1})}$$
     * Yalnızca korpustaki frekans sayımlarına (MLE) bakar. Kelimeler arasındaki anlamsal benzerliği anlayamaz (örneğin "doktor" ile "hekim" tamamen alakasız iki semboldür) ve $n$ büyüdükçe veri seyrekleşmesi (sparsity / curse of dimensionality) yaşar.
  2. **Transformer Modelleri (Self-Attention ve Sürekli Vektör Uzayı):**
     * Transformer modelleri sabit bir pencereye hapsolmaz; **Self-Attention mekanizması** ($Q K^T / \sqrt{d_k}$) ile metindeki tüm tokenlar arasındaki küresel ilişkileri modeller.
     * Tokenlar sürekli ve yoğun (dense) embedding vektörleriyle temsil edilir. Son katmandan çıkan gizli durum vektörü ($h_t$), modelin `lm_head` matrisiyle çarpılarak tüm sözlük (vocabulary) üzerinde logit değerlerine ($z \in \mathbb{R}^{|V|}$) dönüştürülür.
     * Softmax fonksiyonu ile bu logitler olasılık dağılımına çevrilir: $P(w_i) = \frac{\exp(z_i / T)}{\sum \exp(z_j / T)}$. Model bu dağılımdan sampling (veya $T=0$ ise argmax) yaparak bir sonraki tokenı seçer.
* 🎙️ **Videoda Söylenecek Cümle (English Pitch):**
  > *"An n-gram model relies on the Markov assumption, estimating next-word probability strictly by counting surface frequencies within a fixed sliding window of $n-1$ preceding words. It treats words as discrete symbols, suffering from severe sparsity and an inability to model long-range context or semantic similarity. In contrast, a Transformer utilizes multi-head self-attention to dynamically compute pairwise interactions across all input tokens across arbitrary distances. The final contextual hidden state is projected through the language model head into logits across the entire vocabulary, which softmax transforms into a probability distribution from which the next token is generated."*

---

### 📌 BÖLÜM C: Generation Parameters

#### ❓ Soru C1: How do JSON parsing success and accuracy change if you increase the temperature?
* **Mühendislik Açıklaması:**
  Sıcaklık ($T$), softmax öncesi logit değerlerini ölçeklendirir: $P(w_i) = \frac{\exp(z_i / T)}{\sum \exp(z_j / T)}$.
  * $T \to 0$ iken model tamamen açgözlü (greedy / argmax) davranır; deterministiktir.
  * $T$ arttıkça olasılık dağılımı düzleşir (entropy artar), düşük olasılıklı tokenların seçilme şansı doğar.
* **Projemizdeki Somut Etkileri:**
  1. **Büyük Modellerde (Llama 3.1 8B):** Sıcaklık $0.1$'den $1.0$'a çıksa bile güçlü dikkat kabiliyeti sayesinde hem JSON formatını (%100) hem de sınıflandırma doğruluğunu (~%85) stabil korumuştur.
  2. **Küçük Modellerde (Llama 3.2 3B):** Sıcaklık arttıkça model yaratıcılık ve aşırı açıklama (verbosity) eğilimine girmiş, 256 token limitini aşarak JSON'ı yarım bırakmış ve şema geçerliliği **%100'den %88'e gerilemiştir**.
  3. **Görev Doğruluğu (Task Accuracy):** Yüksek sıcaklıkta model izin verilen enum değerleri (`olumlu`, `olumsuz`, `notr`) yerine şema dışı türevler (`oldukça olumlu`, `karışık`) üretmeye meylettiği için görev doğruluğu düşme eğilimi gösterir.
* 🎙️ **Videoda Söylenecek Cümle (English Pitch):**
  > *"Temperature controls the entropy of the softmax token distribution. At low temperature, decoding is near-greedy and deterministic. As temperature increases, the logit distribution flattens. In our experiments with Llama 3.1 8B, high parameter capacity kept parsing success robust at 100%. However, in Llama 3.2 3B, higher temperatures provoked verbosity, exhausting our 256-token limit, leaving JSON unclosed, and degrading schema validity from 100% to 88%. Task accuracy also decreased because higher entropy causes the model to sample non-canonical sentiment descriptors instead of strict schema enum values."*

---

### 📌 BÖLÜM D: Quantization and Hardware

#### ❓ Soru D4: How would your results change if you used a smaller (1B) or a larger (8B) model?
* **Mühendislik Açıklaması (Bizim Somut Deney Sonuçlarımız):**
  Bu soru projemizin tam merkezindedir çünkü çalışmamızda doğrudan **Qwen 2.5 (1.5B)**, **Llama 3.2 (3B)** ve **Llama 3.1 (8B)** modelleri aynı veri kümesinde karşılaştırılmıştır:
  1. **Format Disiplini ve Şema Geçerliliği:**
     * **1.5B (Qwen 2.5):** Prompt-only modda saf JSON oranı sadece **%9**'da kalmış, çıktıların %91'ini markdown kod bloğuna hapsetmiştir. Düşük sıcaklıkta kısırdöngüye girip token sınırına takılmıştır.
     * **8B (Llama 3.1):** Tüm sıcaklıklarda **%100 Saf JSON** ve **%100 Şema Geçerliliği** korumuştur; hiçbir sohbet cümlesi veya gereksiz token üretmemiştir.
  2. **Varlık Çıkarımı (Entity F1):**
     * **1.5B:** Exact F1 %23, Partial F1 %42'de kalmıştır. Küçük model varlık sınırını temiz kesememiş, yanındaki fiil veya sıfatı da varlığa katmıştır.
     * **8B:** Exact F1 **%50.4**, Partial F1 ise **%64.4'e** ulaşmıştır.
  3. **Donanım ve Hız Ödünleşimi (Trade-off):**
     * 1.5B model saniyede ~144 token üretip ~1 GB depolama / ~1.5 GB VRAM harcarken; 8B model ~28 TPS üretip 4.9 GB VRAM kullanmıştır.
* 🎙️ **Videoda Söylenecek Cümle (English Pitch):**
  > *"This question directly reflects our empirical findings, where we benchmarked Qwen 2.5 (1.5B), Llama 3.2 (3B), and Llama 3.1 (8B). With a smaller 1.5B model, format discipline was brittle: in prompt-only mode, pure JSON success was only 9% due to markdown fence traps, and entity extraction Exact F1 was only 23% due to poor boundary detection. Moving to the 8B Llama 3.1 model completely resolved these issues, maintaining 100% pure JSON and schema validity across all temperatures, while boosting Partial Entity F1 to 64.4%. The trade-off is resource footprint: the 1.5B model achieved ~144 TPS (~1 GB model size), whereas the 8B model required ~4.9 GB VRAM running at ~28 TPS."*

---

### 📌 BÖLÜM E: Tokenization (CANLI DENEY SORUSU ⭐)

#### ❓ Soru E1: Why does the Turkish version of a text contain more tokens than its English version? (E2 Canlı Deneyi İle Birlikte)
* **CANLI DENEY:** Terminalde `python src/demo_defense_experiments.py` çalıştırılarak canlı gösterilir!
* **Mühendislik Açıklaması:**
  1. **Tokenizer Sözlüğü Dağılımı (BPE Vocabulary Bias):**
     LLaMA ve Qwen gibi modellerin Byte-Pair Encoding (BPE) sözlükleri ağırlıklı olarak İngilizce internet metinleriyle eğitilmiştir. İngilizcedeki yaygın kelimelerin neredeyse tamamı sözlükte tek bir bütün token olarak yer alır.
  2. **Eklemeli (Agglutinative) Morfoloji ve Parçalanma:**
     Türkçe eklemeli bir dildir. İsim çekim ekleri, iyelik, çoğul ve durum ekleri köke ardışık olarak eklenir. Canlı deneyimizde gösterdiğimiz üzere, Türkçedeki `"kitap"` kökü dahi tek parça değildir; `'kit'` + `'ap'` olarak bölünür. Kelime `"kitaplarımızdan"` olduğunda ise **4 subword parçasına** (`['kit', 'ap', 'larımız', 'dan']`) bölünmektedir.
  3. **İngilizce Kıyası:**
     Aynı kavramın İngilizcesi olan `"from our books"` ise 3 bağımsız kelimeden oluşur ve BPE tokenizer tarafından tam olarak **3 token** olarak kodlanır.
  4. **Sonuç:**
     Türkçe bir metin, İngilizce karşılığına kıyasla **%50 ila %100 daha fazla token** üretir. Bu durum yerel GPU belleğindeki 4096 context penceresinin Türkçe metinlerde çok daha hızlı tükenmesine yol açar.
* 🎙️ **Videoda Söylenecek Cümle (English Pitch):**
  > *"A Turkish text generates significantly more tokens than its English equivalent due to two factors: tokenizer vocabulary bias and agglutinative morphology. Modern BPE tokenizers are pre-trained predominantly on English corpora, meaning common English words exist as single whole tokens. In contrast, Turkish is an agglutinative language where grammatical relationships are expressed through consecutive suffixes. As demonstrated live in our script, even a basic word like 'kitaplarımızdan' is fractured into 4 subwords: 'kit', 'ap', 'larımız', 'dan', whereas its English translation 'from our books' takes only 3 tokens. Consequently, Turkish texts consume 50% to 100% more tokens, filling the context window twice as fast."*

---

### 📌 BÖLÜM F: Prompt Strategies

#### ❓ Soru F2: Does changing the order of the few-shot examples change the result?
* **Mühendislik Açıklaması:**
  1. **Evet, sonucu doğrudan ve ölçülebilir şekilde değiştirir.** Bu durum literatürde **"Recency Bias" (Yakınlık / Sonluk Önyargısı)** ve **"In-Context Order Sensitivity"** olarak bilinir.
  2. **Attention Mekanizması Açısından Sebebi:**
     * Otoregresif Transformer modellerinde promptun en sonuna yerleştirilen few-shot örnek, sınıflandırılacak hedef metne konumsal olarak en yakın örnektir ($|i - j|$ konumsal mesafesi en düşüktür).
     * Modeller, son örneğin sınıf etiketine ve biçimlendirme stiline karşı orantısız bir dikkat ağırlığı (attention bias) geliştirir.
     * Örneğin promptun sonundaki 3. örnek "olumsuz" ise, modelin sınırda kalan (ambiguous / nötr) metinleri "olumsuz" olarak sınıflandırma olasılığı (label prior) yapay olarak yükselir.
  3. **Çözüm:** Üretim sistemlerinde few-shot örneklerin sırası rastgele karıştırılarak (permutation averaging) veya dengeli dağıtılarak test edilir.
* 🎙️ **Videoda Söylenecek Cümle (English Pitch):**
  > *"Yes, changing the order of few-shot examples significantly alters model predictions—a phenomenon well-documented as 'Recency Bias' in in-context learning. In autoregressive Transformers, the final demonstration is positioned closest to the target test input. Self-attention layers assign disproportionately higher attention weights to this final example, biasing the model's output distribution toward its specific class label and syntax. For instance, placing a negative review as the final shot artificially inflates the probability of predicting 'olumsuz' for ambiguous test cases."*

---

### 📌 BÖLÜM G: Comparison with Classical Methods

#### ❓ Soru G2: If you had only 10 labeled examples, which method would perform better: a classical model or a few-shot LLM?
* **Mühendislik Açıklaması:**
  1. **Few-Shot LLM Açık Ara Üstündür!**
  2. **Klasik Modelin (TF-IDF / Naive Bayes / Lojistik Regresyon) Çöküş Sebebi:**
     * 10 örnekte geçen benzersiz kelime sayısı son derece kısıtlıdır. TF-IDF matrisi aşırı derecede seyrek (sparse) kalır.
     * Test kümesindeki metinlerde yer alan ama eğitimdeki bu 10 örnekte geçmeyen kelimeler için klasik model tamamen kördür (Out-Of-Vocabulary / zero feature weight).
     * 10 örneğe ağırlık uydurmaya çalışan lojistik regresyon katastrofik bir **aşırı öğrenme (overfitting)** yaşar; genelleme yeteneği sıfırlanır ve doğruluk yazı-tura düzeyine (%33 - %50) düşer.
  3. **Few-Shot LLM'in Zafer Sebebi:**
     * Dil modeli sıfırdan öğrenmez. Trilyonlarca tokenlık pre-training sayesinde dünya bilgisine, kelimelerin anlamsal vektörlerine ve dilbilgisi kurallarına zaten sahiptir.
     * 10 örnek, modele sıfırdan dil öğretmek için değil; sadece **görevin sınıflandırma olduğunu ve istenen JSON formatını hatırlatmak (in-context task induction)** için kullanılır. Model metindeki eş anlamlıları, argo ifadeleri ve duygu tonunu genel parametrik zekasıyla anında çözer.
* 🎙️ **Videoda Söylenecek Cümle (English Pitch):**
  > *"With only 10 labeled examples, a few-shot LLM decisively outperforms classical models. Classical approaches like TF-IDF with Logistic Regression suffer from extreme vocabulary sparsity and catastrophic overfitting; any word in the test set absent from those 10 samples receives zero weight, rendering the model blind to unseen context. In contrast, an LLM does not learn from scratch—it brings trillions of pre-trained parameters containing extensive world and linguistic knowledge. The 10 examples merely act as an in-context primer to define the task and the JSON contract, allowing the LLM to generalize effectively."*

---

## 🧩 8. Kapalı Uçlu (Closed-Set) vs. Açık Uçlu (Open-Ended) Görev Ayrımı ve Schema-Constrained Etkisi

*(Raporun Discussion bölümü ve Soru A4 savunması için temel kavram özeti)*

### 1. Kapalı Uçlu Görev Nedir? (`category` ve `sentiment`)
* **Tanım:** Modelin üretebileceği yanıtlar **önceden sabitlenmiş, sınırlı bir kümeden (`enum`)** seçilmek zorundadır. Model yeni veya serbest kelime uyduramaz.
* **Projemizdeki Karşılığı (`config.py -> JSON_SCHEMA`):**
  * `"category"`: Yalnızca `["urun", "sinema", "ansiklopedi"]` (3 seçenek)
  * `"sentiment"`: Yalnızca `["olumlu", "olumsuz", "notr"]` (3 seçenek)
* **Schema-Constrained Neden Burada Kazandırır?**
  * `Prompt-Only` modda `Llama 3.2` gibi modeller `"yorum"`, `"teşekkür"`, `Llama 3.1` ise Türkçe `"ürün"` yazarak geçerli kümenin dışına çıkabilmektedir.
  * `Schema-Constrained` modda GBNF grameri yalnızca bu 3 seçeneğe izin veren tokenların dışındaki tüm tokenları maskeler ($-\infty$). Bu sayede şema dışı etiket üretimi sıfırlanır (`Llama 3.1` Kategori Macro F1: **%67.8 → %91.7**, `Llama 3.2`: **%4.2 → %49.7**).

### 2. Açık Uçlu Bilgi Çıkarımı Nedir? (`entities`)
* **Tanım:** Olası cevapların önceden sabit bir listesi yoktur; model **girdi metnine bağlı olarak serbest string/span üretmek** zorundadır (her yeni metinde farklı kişi, marka, ürün, yer adları çıkar).
* **Projemizdeki Karşılığı (`config.py -> JSON_SCHEMA`):**
  * `"entities"`: `{"type": "array", "items": {"type": "string"}}` (içeriği kısıtsız, değişken uzunluklu liste).
* **Schema-Constrained Neden Burada Hafif Kayba Yol Açabilir?**
  1. **Otoregresif Bağlam Kayması:** JSON soldan sağa üretildiği için önceki alanlarda (`"ürün"` $\rightarrow$ `"urun"`) yapılan zorunlu değişiklikler, `"entities"` üretilirken modelin koşullandığı bağlamı (KV-cache) değiştirir.
  2. **GBNF Token Maskeleme Baskısı:** Her adımda yalnızca geçerli JSON sözdizimine uyan tokenlara izin verilmesi, modelin doğal instruction-tuned olasılık dağılımını daraltarak bazen `"film"`, `"roman seti"` gibi jenerik/fazladan varlıklar eklemesine veya asıl varlığı kaçırmasına yol açar (`Llama 3.1` Exact F1: **%50.4 → %46.2**, `Llama 3.2`: **%30.8 → %27.9**).
  3. **Sınırsız Dizi Riski:** `maxItems` kısıtı yoksa küçük modeller (`Qwen 2.5`, `Llama 3.2`) `"entities"` dizisinde tekrar döngüsüne girip `num_predict=256` sınırında kesilebilir (%1–%4 kalan hata).

* 🎙️ **Videoda / Sözlüde Kullanılacak Özet Cümle (English Pitch):**
  > *"We observed a clear dichotomy between closed-set classification and open-ended information extraction under schema-constrained decoding. For closed-set fields (`category` and `sentiment` with strict `enum` values), GBNF grammar masking prevents out-of-vocabulary label hallucinations and dramatically boosts Macro F1 (from 67.8% to 91.7% on Llama 3.1). Conversely, for open-ended entity extraction—where valid strings depend freely on the input text—strict token-level constraints slightly perturb the model's autoregressive distribution, causing a 3–4 point drop in Entity F1 on 3B and 8B models."*

---
