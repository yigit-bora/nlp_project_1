import re
from typing import List, Dict, Any
from sklearn.metrics import f1_score


def turkish_lower(s: str) -> str:
    """
    Türkçe kurallarına uygun küçük harfe çevirme.
    Hocanın yönergesi: 'İ' -> 'i' ve 'I' -> 'ı' dönüşümü lower()'dan ÖNCE yapılmalıdır.
    Böylece 'İSTANBUL'.lower() -> 'i̇stanbul' (combining dot) hatası engellenir.
    """
    if not isinstance(s, str):
        return ""
    return s.replace("İ", "i").replace("I", "ı").lower()


def normalize(s: str) -> str:
    """
    Karşılaştırma için metin normalizasyonu.
    1. Hocanın kuralına göre 'İ' ve 'I' harflerini önce güvenli çevirir, sonra lower() çağırır.
    2. Karşılaştırma tutarlılığı için Türkçe karakterleri normalize eder.
    3. Noktalama işaretlerini ve fazla boşlukları temizler.
    """
    if not isinstance(s, str):
        return ""
    cleaned = turkish_lower(s.strip())
    tr_map = str.maketrans("ıöüçşğ", "ioucsg")
    cleaned = cleaned.translate(tr_map)
    cleaned = re.sub(r"[^\w\s]", "", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()


def calculate_accuracy(predictions: List[str], ground_truths: List[str]) -> float:
    """
    Sınıflandırma doğruluğu (Accuracy).
    Doğru tahmin sayısı / toplam sayı.
    """
    if not predictions or len(predictions) != len(ground_truths):
        return 0.0
    
    correct = sum(
        1 for p, g in zip(predictions, ground_truths)
        if normalize(p) == normalize(g)
    )
    return (correct / len(predictions)) * 100


def calculate_macro_f1(predictions: List[str], ground_truths: List[str]) -> float:
    """
    Sınıflandırma Macro F1 skoru (Kategori ve Duygu için).
    """
    if not predictions or len(predictions) != len(ground_truths):
        return 0.0
    
    norm_preds = [normalize(p) for p in predictions]
    norm_gts = [normalize(g) for g in ground_truths]
    
    score = f1_score(norm_gts, norm_preds, average="macro", zero_division=0)
    return round(float(score) * 100, 2)


def is_entity_match(pred: str, gold: str, match_mode: str = "exact") -> bool:
    """
    İki varlık (entity) arasındaki eşleşme kontrolü.
    - match_mode="exact": Birebir harf eşitliği (Türkçe normalizasyonlu).
    - match_mode="partial": Kısmi eşleşme (biri diğerini içeriyor mu veya kelime kesişimi var mı).
    """
    p = normalize(pred)
    g = normalize(gold)
    
    if not p or not g:
        return False
        
    if match_mode == "exact":
        return p == g
    elif match_mode == "partial":
        # Alt küme kontrolü (örn: 'de niro' in 'robert de niro' veya 'robert de niro' in 'de niro')
        if p in g or g in p:
            return True
        # Kelime bazlı kesişim (örn: 'karadeniz teknik universitesi' ile 'teknik universite')
        p_words = set(p.split())
        g_words = set(g.split())
        if p_words and g_words and (p_words & g_words):
            return True
        return False
    else:
        raise ValueError(f"Geçersiz match_mode: {match_mode}")


def calculate_entity_metrics(
    predicted_entities_list: List[List[str]],
    ground_truth_entities_list: List[List[str]],
    match_mode: str = "exact"
) -> Dict[str, Any]:
    """
    Varlık çıkarımı (Entity Extraction) Micro-averaged Precision, Recall ve F1 skoru.
    Her örnek için çıkarılan entity'ler altın etiketlerle (gold entities) eşleştirilir.
    """
    tp = 0
    fp = 0
    fn = 0
    
    for preds, golds in zip(predicted_entities_list, ground_truth_entities_list):
        matched_gold_indices = set()
        
        # Temiz ve benzersiz tahminler
        cleaned_preds = [p for p in preds if p and str(p).strip()]
        cleaned_golds = [g for g in golds if g and str(g).strip()]
        
        for p in cleaned_preds:
            match_found = False
            for g_idx, g in enumerate(cleaned_golds):
                if g_idx not in matched_gold_indices and is_entity_match(p, g, match_mode=match_mode):
                    matched_gold_indices.add(g_idx)
                    match_found = True
                    break
            if match_found:
                tp += 1
            else:
                fp += 1
                
        # Eşleşmeyen altın etiketler = False Negative
        fn += (len(cleaned_golds) - len(matched_gold_indices))
        
    precision = (tp / (tp + fp) * 100) if (tp + fp) > 0 else 0.0
    recall = (tp / (tp + fn) * 100) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    
    return {
        "match_mode": match_mode,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": round(precision, 2),
        "recall": round(recall, 2),
        "f1": round(f1, 2)
    }


def calculate_entity_f1(
    predicted_entities_list: List[List[str]],
    ground_truth_entities_list: List[List[str]]
) -> Dict[str, Any]:
    """
    Hem Exact Match hem de Partial Match F1 skorlarını birlikte hesaplar.
    """
    exact = calculate_entity_metrics(predicted_entities_list, ground_truth_entities_list, match_mode="exact")
    partial = calculate_entity_metrics(predicted_entities_list, ground_truth_entities_list, match_mode="partial")
    
    total_pred = sum(len(p) for p in predicted_entities_list)
    total_gt = sum(len(g) for g in ground_truth_entities_list)
    
    return {
        "exact_f1": exact["f1"],
        "exact_precision": exact["precision"],
        "exact_recall": exact["recall"],
        "partial_f1": partial["f1"],
        "partial_precision": partial["precision"],
        "partial_recall": partial["recall"],
        "total_predicted_entities": total_pred,
        "total_gt_entities": total_gt,
        "avg_entities_per_sample": round(total_pred / len(predicted_entities_list), 2) if predicted_entities_list else 0
    }
