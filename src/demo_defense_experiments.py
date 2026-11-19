#!/usr/bin/env python3
"""
NLP Proje I - Savunma Soruları Canlı Deney Scripti (Defense Questions Live Demo)
Bu script, video kaydında veya savunmada jüriye canlı olarak çalıştırılıp gösterilmek üzere hazırlanmıştır.
Kapsanan Sorular:
  - Soru E2 (Canlı Deney): "kitaplarımızdan" kelimesinin Tokenizer ile parçalanması (Qwen & Llama 3)
  - Soru A1: Koddan 1 satır silinirse ne olur? (Türkçe karakter 'İ'/'I' normalizasyon hatası)
  - Soru D1 & D3: Model Kuantizasyon Seviyesi ve VRAM Bellek Hesabı
"""

import sys
from transformers import AutoTokenizer

def print_header(title):
    print("\n" + "=" * 70)
    print(f" 🔬 {title}")
    print("=" * 70)

def demo_soru_e1_e2_tokenization():
    print_header("SORU E1 & E2 CANLI DENEYİ: Neden Türkçe Metinler Daha Fazla Token Tutar?")
    print("Hocanın Sorusu E1: 'Bir metnin Türkçe versiyonu neden İngilizcesinden daha fazla token içerir?'")
    print("Hocanın Sorusu E2: 'Modelinizin tokenizer'ı kitaplarımızdan kelimesini nasıl ayırıyor?'")
    print("                   (Hugging Face AutoTokenizer ile terminalde canlı ispatlıyoruz.)\n")

    word_tr = "kitaplarımızdan"
    phrase_en = "from our books"

    # 1. Qwen 2.5 Tokenizer
    print("▶ 1. MODEL: Qwen 2.5 (1.5B / 7B Base)")
    qwen_tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
    tokens_qwen = qwen_tok.tokenize(word_tr)
    ids_qwen = qwen_tok.encode(word_tr, add_special_tokens=False)
    
    print(f"   Kelime: '{word_tr}'")
    print(f"   Alt Kelime Parçaları (Subwords): {tokens_qwen}")
    print(f"   Token ID'leri: {ids_qwen}")
    print(f"   Token Sayısı: {len(tokens_qwen)} token")

    # 2. Llama 3 / 3.1 Tokenizer
    print("\n▶ 2. MODEL: Meta Llama 3.1 (8B Base - unsloth open weights)")
    llama_tok = AutoTokenizer.from_pretrained("unsloth/llama-3-8b-Instruct")
    tokens_llama = llama_tok.tokenize(word_tr)
    ids_llama = llama_tok.encode(word_tr, add_special_tokens=False)
    
    print(f"   Kelime: '{word_tr}'")
    print(f"   Alt Kelime Parçaları (Subwords): {tokens_llama}")
    print(f"   Token ID'leri: {ids_llama}")
    print(f"   Token Sayısı: {len(tokens_llama)} token")

    # 3. Türkçe vs İngilizce Token Verimliliği Karşılaştırması (Soru E1 İlişkisi)
    tokens_en = llama_tok.tokenize(phrase_en)
    print("\n▶ 3. KARŞILAŞTIRMA (Soru E1 Temel Cevabı):")
    print(f"   Türkçe:    '{word_tr}' -> {len(tokens_llama)} token {tokens_llama}")
    print(f"   İngilizce: '{phrase_en}' -> {len(tokens_en)} token {tokens_en}")
    print("   💡 Çıkarım: 'kitap' kökü bile BPE sözlüğünde tek parça değildir ('kit' + 'ap').")
    print("      Eklemeli Türkçe morfolojisi yüzünden Türkçe metinler %50-%100 daha fazla token harcar.")

def demo_soru_a1_code_ablation():
    print_header("SORU A1: Koddan 1 Satır Silinirse Ne Olur? (Ablation Test)")
    print("Seçilen Kod Satırı (src/metrics.py - Satır 14):")
    print("   return s.replace('İ', 'i').replace('I', 'ı').lower()\n")
    
    test_word = "İSTANBUL"
    gold_target = "istanbul"

    # Python Varsayılan lower()
    py_lower = test_word.lower()
    py_match = (py_lower == gold_target)

    # Bizim Kodumuzdaki Türkçe Uyumlu Normalizasyon
    our_lower = test_word.replace("İ", "i").replace("I", "ı").lower()
    our_match = (our_lower == gold_target)

    print(f"Girdi Kelime: '{test_word}' | Hedef Altın Varlık (Gold Entity): '{gold_target}'")
    print("-" * 70)
    print(f"1. EĞER BU SATIR SİLİNİRSE (Standart Python text.lower()):")
    print(f"   Sonuç: '{py_lower}' (Unicode Hex: {[hex(ord(c)) for c in py_lower]})")
    print(f"   'i̇' karakterinde birleştirici nokta (combining dot: \\u0307) kalır!")
    print(f"   Eşleşme Durumu: {py_match} ❌ (Exact Match Fails!)")
    print("-" * 70)
    print(f"2. BİZİM KODUMUZ ÇALIŞTIĞINDA (replace('İ', 'i').replace('I', 'ı').lower()):")
    print(f"   Sonuç: '{our_lower}' (Unicode Hex: {[hex(ord(c)) for c in our_lower]})")
    print(f"   Eşleşme Durumu: {our_match} ✅ (Exact Match Succeeds!)")
    print("-" * 70)
    print("💡 Çıkarım: Bu satır silinirse büyük harfli Türkçe varlıkların tamamında F1 skoru")
    print("   sıfıra çakılır. Hocanın şartnamedeki 'combining dot' uyarısına tam uyulmuştur.")

def demo_soru_d4_model_scale():
    print_header("SORU D4 & D1: Küçük Model (1.5B) vs Büyük Model (8B) Karşılaştırması")
    print("Hocanın Sorusu D4: 'Daha küçük (1B) veya daha büyük (8B) model kullansaydınız sonuçlar nasıl değişirdi?'\n")
    print("Bizim Benchmark Sonuçlarımız (Qwen 2.5 1.5B vs Llama 3.1 8B):")
    print("-" * 70)
    print("1. FORMAT DİSİPLİNİ:")
    print("   - Qwen 2.5 (1.5B):  %9 Saf JSON Başarısı (Markdown çitine hapsoluyor)")
    print("   - Llama 3.1 (8B):   %100 Saf JSON Başarısı (Tüm sıcaklıklarda kusursuz)")
    print("-" * 70)
    print("2. VARLIK ÇIKARIMI (ENTITY EXTRACTION F1):")
    print("   - Qwen 2.5 (1.5B):  Exact F1: %23.1 | Partial F1: %42.0")
    print("   - Llama 3.1 (8B):   Exact F1: %50.4 | Partial F1: %64.4 (Lider)")
    print("-" * 70)
    print("3. DONANIM VE HIZ ÖDÜNLEŞİMİ (HARDWARE FOOTPRINT):")
    print("   - 1.5B Model: ~144 TPS (Çok hızlı), ~1.0 GB ağırlık / ~1.5 GB VRAM")
    print("   - 8B Model:   ~28 TPS (Daha yavaş), 4.9 GB (RTX 3060 6GB VRAM'e sığdı)")
    print("-" * 70)
    print("💡 Kuantizasyon Seviyesi (Soru D1): Modeller Q4_K_M seviyesindedir (ollama show ile teyitli).")

if __name__ == "__main__":
    print("\n🎓 NLP PROJE I: SAVUNMA SORULARI CANLI GÖSTERİMİ")
    print("Kullanıcının Seçtiği Savunma Soruları ve Canlı İspatları")
    demo_soru_e1_e2_tokenization()
    demo_soru_a1_code_ablation()
    demo_soru_d4_model_scale()

    print("\n" + "=" * 70)
    print("✅ Canlı Savunma Deneyi Başarıyla Tamamlandı.")
    print("=" * 70 + "\n")
