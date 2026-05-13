"""
Lokal Türkçe açıklama üretimi — API yok, tamamen offline.

Strateji:
1. Helsinki-NLP (Marian) zaten projede çeviri için kullanılıyor.
   Bunu tekrar yüklemek yerine inference.py üzerinden gelecek
   translated_question'ı direkt kullanıyoruz.
2. mT5-small ile kısa, dinamik Türkçe cümle üretimi yapıyoruz.
   Prompt'u Türkçe olarak veriyoruz; mT5 çok dilli olduğundan
   Türkçe output üretiyor.
3. Fallback: mT5 yüklenemezse (bellek kısıtı vb.) zengin template
   sistemi devreye giriyor. Templatler rastgele seçim + interpolasyon
   ile her seferinde farklı cümle üretiyor.
"""

import random
import re

# mT5 opsiyonel — yüklenemezse template moduna geçer
_mt5_model = None
_mt5_tokenizer = None
_MT5_LOADED = False


def _try_load_mt5():
    global _mt5_model, _mt5_tokenizer, _MT5_LOADED
    if _MT5_LOADED:
        return _mt5_model is not None
    _MT5_LOADED = True
    try:
        from transformers import MT5ForConditionalGeneration, T5Tokenizer
        import torch
        MODEL_NAME = "google/mt5-small"
        print(f"[NLG] mT5-small yükleniyor: {MODEL_NAME}")
        _mt5_tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
        _mt5_model = MT5ForConditionalGeneration.from_pretrained(MODEL_NAME)
        _mt5_model.eval()
        print("[NLG] mT5-small yüklendi ✓")
        return True
    except Exception as e:
        print(f"[NLG] mT5 yüklenemedi, template moduna geçiliyor: {e}")
        return False


# ---------------------------------------------------------------------------
# mT5 tabanlı üretim
# ---------------------------------------------------------------------------

def _generate_with_mt5(
    user_question: str,
    vqa_answer: str,
    confidence: float,
    translated_question: str,
) -> str:
    import torch
    confidence_pct = int(confidence * 100)
    answer_tr = "evet" if vqa_answer.lower() == "yes" else "hayır"

    prompt = (
        f"Türkçe tıbbi görüntü analizi açıklaması yaz. "
        f"Soru: {user_question} "
        f"Model cevabı: {answer_tr} (güven: %{confidence_pct}). "
        f"Kısa, sade, anlaşılır açıkla. Teşhis koyma."
    )

    inputs = _mt5_tokenizer(
        prompt,
        return_tensors="pt",
        max_length=200,
        truncation=True,
    )

    with torch.no_grad():
        outputs = _mt5_model.generate(
            **inputs,
            max_new_tokens=120,
            num_beams=4,
            early_stopping=True,
            no_repeat_ngram_size=3,
            temperature=0.85,
            do_sample=False,
        )

    raw = _mt5_tokenizer.decode(outputs[0], skip_special_tokens=True)
    # mT5 bazen prompt'u tekrar ediyor, temizle
    raw = re.sub(r"^(Türkçe tıbbi.*?\.)\s*", "", raw).strip()
    if len(raw) < 20:
        raise ValueError("mT5 çıktısı çok kısa, template'e geç")
    return raw


# ---------------------------------------------------------------------------
# Template tabanlı NLG — dinamik interpolasyon
# ---------------------------------------------------------------------------

# Her liste: (evet|hayır) → cümle parçaları
_OPENERS_YES = [
    "Görüntü analizi, {soru_konusu} açısından **olumlu** bir bulgu ortaya koyuyor.",
    "Model, patoloji görüntüsünü inceledi ve {soru_konusu} varlığını tespit etti.",
    "Yapılan analiz sonucunda {soru_konusu} ile uyumlu bulgular gözlemlendi.",
    "Görüntüde {soru_konusu} ile ilişkili işaretler saptandı.",
]

_OPENERS_NO = [
    "Görüntü analizi, {soru_konusu} açısından **olumsuz** bir sonuç ortaya koyuyor.",
    "Model, patoloji görüntüsünü inceledi ve {soru_konusu} varlığına dair belirgin bir bulgu tespit etmedi.",
    "Yapılan analiz, görüntüde {soru_konusu} ile uyumlu net bir işaret göstermedi.",
    "İnceleme sonucunda {soru_konusu} ile ilişkili belirgin bir özellik gözlemlenmedi.",
]

_CONFIDENCE_HIGH = [
    "Model bu kararı **%{pct} güven** ile verdi; bu oldukça yüksek bir emin olma düzeyi.",
    "Güven skoru **%{pct}** olup modelin bu konuda oldukça kararlı olduğuna işaret ediyor.",
    "**%{pct} güven skoru**, modelin bu analizi yüksek bir kesinlikle yaptığını gösteriyor.",
]

_CONFIDENCE_MID = [
    "Model bu sonuca **%{pct} güven** ile ulaştı; sonuçlar değerlendirilirken göz önünde bulundurulmalı.",
    "Güven skoru **%{pct}** düzeyinde; bu oran, bulgunun daha ayrıntılı inceleme gerektirebileceğine işaret ediyor.",
    "**%{pct}** güven oranıyla bu bulgu, ek klinik değerlendirme için bir başlangıç noktası niteliği taşıyor.",
]

_CONFIDENCE_LOW = [
    "Model **%{pct} güven** ile karar verdi; bu düşük skor nedeniyle bulgunun dikkatli yorumlanması önerilir.",
    "Güven skoru **%{pct}** gibi düşük bir seviyede olduğundan bu sonuç tek başına yeterli değildir.",
    "**%{pct}** güven oranı, modelin bu görüntü üzerinde emin olmadığını gösteriyor; ek analiz faydalı olabilir.",
]

_DISCLAIMERS = [
    "Bu sistem bir karar destek aracıdır; kesin tıbbi teşhis için uzman hekime başvurmanız önerilir.",
    "Sonuçlar tanı amaçlı kullanılamaz; bir patolog veya ilgili uzman görüşü alınmalıdır.",
    "Bu analiz bilgilendirme amaçlıdır ve klinik karar süreçlerinde tek kaynak olarak kullanılmamalıdır.",
    "Tıbbi karar almadan önce mutlaka nitelikli bir sağlık profesyoneline danışınız.",
]

_TOPIC_FALLBACKS = [
    "ilgili patolojik yapı",
    "söz konusu bulgu",
    "incelenen özellik",
    "belirtilen durum",
]


def _extract_topic(question: str) -> str:
    """Sorudan konu çıkarmaya çalışır, başaramazsa fallback döner."""
    q = question.strip().rstrip("?").strip()
    # "... var mı", "... görülüyor mu", "... mevcut mu" gibi kalıpları temizle
    patterns = [
        r"\bvar mı\b", r"\bgörülüyor mu\b", r"\bmevcut mu\b",
        r"\bsaptandı mı\b", r"\btespit edildi mi\b", r"\bgözlemlendi mi\b",
        r"^(bu görüntüde|patoloji görüntüsünde|görüntüde)\s+",
    ]
    for p in patterns:
        q = re.sub(p, "", q, flags=re.IGNORECASE).strip()

    q = q.strip().lstrip("?").strip()
    if len(q) > 5:
        return q.lower()
    return random.choice(_TOPIC_FALLBACKS)


def _generate_with_templates(
    user_question: str,
    vqa_answer: str,
    confidence: float,
    translated_question: str,
) -> str:
    confidence_pct = int(confidence * 100)
    topic = _extract_topic(user_question)

    if vqa_answer.lower() == "yes":
        opener = random.choice(_OPENERS_YES).format(soru_konusu=topic)
    else:
        opener = random.choice(_OPENERS_NO).format(soru_konusu=topic)

    if confidence_pct >= 80:
        conf_sentence = random.choice(_CONFIDENCE_HIGH).format(pct=confidence_pct)
    elif confidence_pct >= 55:
        conf_sentence = random.choice(_CONFIDENCE_MID).format(pct=confidence_pct)
    else:
        conf_sentence = random.choice(_CONFIDENCE_LOW).format(pct=confidence_pct)

    disclaimer = random.choice(_DISCLAIMERS)

    return f"{opener} {conf_sentence} {disclaimer}"


# ---------------------------------------------------------------------------
# Dışarıya açık tek fonksiyon
# ---------------------------------------------------------------------------

def generate_explanation(
    user_question: str,
    vqa_answer: str,
    confidence: float,
    translated_question: str,
) -> str:
    """
    Türkçe açıklama üretir.
    Önce mT5-small'ı dener; başarısız olursa zengin template sistemine geçer.
    """
    mt5_available = _try_load_mt5()

    if mt5_available:
        try:
            return _generate_with_mt5(
                user_question, vqa_answer, confidence, translated_question
            )
        except Exception as e:
            print(f"[NLG] mT5 üretim hatası, template'e geçildi: {e}")

    return _generate_with_templates(
        user_question, vqa_answer, confidence, translated_question
    )