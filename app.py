import re
import numpy as np
import joblib
import torch

from flask import Flask, request, jsonify, render_template
from transformers import RobertaTokenizerFast, RobertaForSequenceClassification
from scipy.sparse import hstack
from langdetect import detect, DetectorFactory, LangDetectException

# ------------------------------------------------
# AYARLAR
# ------------------------------------------------
MIN_WORDS = 30
ENGLISH_RATIO_THRESHOLD = 0.80
DetectorFactory.seed = 0

# Ağırlıklar (Toplam 1.0 olmalı)
WEIGHTS = {
    "RoBERTa": 0.30,
    "RandomForest": 0.30,
    "XGBoosting": 0.40
}

app = Flask(__name__)

print("🔄 Modeller yükleniyor...")

# -------------------------------------------------
# MODEL DOSYALARI (DOSYA YAPINA UYUMLU)
# -------------------------------------------------
rf_model = joblib.load("rf_model.pkl")
tfidf_main = joblib.load("tfidf_vectorizer.pkl")

xgb_model = joblib.load("xgb_150trees.joblib")
tfidf_word = joblib.load("tfidf_word.joblib")
tfidf_char = joblib.load("tfidf_char.joblib")

tokenizer = RobertaTokenizerFast.from_pretrained("roberta_model_folder")
roberta = RobertaForSequenceClassification.from_pretrained("roberta_model_folder")
roberta.eval()

print("✅ Modeller yüklendi")

# -------------------------------------------------
# YARDIMCI FONKSİYONLAR
# -------------------------------------------------
def clean_text(text: str) -> str:
    text = (text or "").strip()
    text = re.sub(r"\s+", " ", text)
    return text

def count_words(text: str) -> int:
    return len(text.split())

def english_ratio(text: str) -> float:
    sentences = re.split(r"[.!?]\s+", text)
    sentences = [s for s in sentences if len(s) > 10]

    if not sentences:
        return 0.0

    en = 0
    total = 0
    for s in sentences:
        try:
            if detect(s) == "en":
                en += 1
            total += 1
        except LangDetectException:
            continue

    return en / total if total else 0.0

def softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    e = np.exp(x)
    return e / np.sum(e)

def format_result(name: str, ai_prob: float) -> dict:
    ai_prob = float(ai_prob)
    return {
        "name": name,
        "label": "AI" if ai_prob >= 0.5 else "HUMAN",
        "ai_pct": round(ai_prob * 100, 2),
        "human_pct": round((1 - ai_prob) * 100, 2)
    }

# -------------------------------------------------
# MODEL TAHMİNLERİ
# -------------------------------------------------
def predict_roberta(text: str) -> dict:
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    )
    with torch.no_grad():
        logits = roberta(**inputs).logits[0].cpu().numpy()
        probs = softmax(logits)

    # probs[1] -> AI sınıfı varsayımı
    return format_result("RoBERTa", probs[1])

def predict_rf(text: str) -> dict:
    X = tfidf_main.transform([text])
    proba = rf_model.predict_proba(X)[0]
    return format_result("RandomForest", proba[1])

def predict_xgb(text: str) -> dict:
    Xw = tfidf_word.transform([text])
    Xc = tfidf_char.transform([text])
    X = hstack([Xw, Xc])
    proba = xgb_model.predict_proba(X)[0]
    return format_result("XGBoosting", proba[1])

# -------------------------------------------------
# GENEL KARAR
# -------------------------------------------------
def overall_decision(results: list[dict]) -> dict:
    labels = [r["label"] for r in results]

    if len(set(labels)) > 1:
        return {
            "title": "DÜŞÜK GÜVEN (Hibrit İçerik)",
            "desc": "Modeller arasında tutarsızlık var veya metin karışık olabilir."
        }

    if labels[0] == "AI":
        return {
            "title": "YÜKSEK OLASILIK (AI)",
            "desc": "Tüm modeller metnin yapay zeka üretimi olduğunu gösteriyor."
        }

    return {
        "title": "YÜKSEK OLASILIK (HUMAN)",
        "desc": "Tüm modeller metnin insan yazımı olduğunu gösteriyor."
    }

def weighted_average_result(results: list[dict]) -> dict:
    """
    İstenen ağırlıklarla ortalama AI olasılığını hesaplar:
      XGBoosting: 0.4
      RandomForest: 0.3
      RoBERTa: 0.3
    """
    # Güvenlik: ağırlık toplamı 1 değilse normalize et
    wsum = sum(WEIGHTS.values())
    weights_norm = {k: v / wsum for k, v in WEIGHTS.items()} if wsum else WEIGHTS

    # results içinden isim -> ai_prob eşleştir
    name_to_ai = {r["name"]: (r["ai_pct"] / 100.0) for r in results}

    # Eksik model adı varsa ağırlığını 0 say (ve kalanları yeniden normalize et)
    available = {k: v for k, v in weights_norm.items() if k in name_to_ai}
    if not available:
        # Hiç yoksa fallback: düz ortalama
        ai_probs = [r["ai_pct"] / 100.0 for r in results]
        avg_ai = float(np.mean(ai_probs)) if ai_probs else 0.0
        return {
            "name": "WeightedAverage",
            "label": "AI" if avg_ai >= 0.5 else "HUMAN",
            "ai_pct": round(avg_ai * 100, 2),
            "human_pct": round((1 - avg_ai) * 100, 2),
            "weights": WEIGHTS
        }

    avail_sum = sum(available.values())
    available = {k: v / avail_sum for k, v in available.items()} if avail_sum else available

    avg_ai = 0.0
    for model_name, w in available.items():
        avg_ai += w * name_to_ai[model_name]

    return {
        "name": "WeightedAverage",
        "label": "AI" if avg_ai >= 0.5 else "HUMAN",
        "ai_pct": round(avg_ai * 100, 2),
        "human_pct": round((1 - avg_ai) * 100, 2),
        "weights": WEIGHTS
    }

# -------------------------------------------------
# ROUTES
# -------------------------------------------------
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)
        text = clean_text(data.get("text", ""))

        if not text:
            return jsonify({"error": "Metin boş olamaz."}), 400

        if count_words(text) < MIN_WORDS:
            return jsonify({"error": "Metin çok kısa."}), 400

        ratio = english_ratio(text)

        if ratio < 0.3:
            return jsonify({"error": "Desteklenmeyen dil."}), 400

        results = [
            predict_roberta(text),
            predict_rf(text),
            predict_xgb(text)
        ]

        banner = overall_decision(results)

        # Çok dilli ise zorla düşük güven
        if ratio < ENGLISH_RATIO_THRESHOLD:
            banner = {
                "title": "DÜŞÜK GÜVEN (Çok Dilli Metin)",
                "desc": "Metin birden fazla dil içeriyor."
            }

        aggregate = weighted_average_result(results)

        return jsonify({
            "banner": banner,
            "results": results,
            "aggregate": aggregate,
            "lang": {
                "english_ratio": round(float(ratio), 3),
                "min_words": MIN_WORDS,
                "threshold": ENGLISH_RATIO_THRESHOLD
            }
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# -------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)
