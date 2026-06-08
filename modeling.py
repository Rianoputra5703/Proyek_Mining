import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from preprocessing import load_dataset, split_dataset

MODEL_DIR  = ROOT / "output" / "model"
MODEL_PATH = MODEL_DIR / "hoax_detector.joblib"


def build_pipeline(n_estimators=200, max_depth=None, min_samples_split=5, min_samples_leaf=2):
    return Pipeline([
        ("tfidf", TfidfVectorizer(max_features=25000, ngram_range=(1, 2), lowercase=True, token_pattern=r"(?u)\b\w\w+\b")),
        ("clf", RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, class_weight="balanced", random_state=42, n_jobs=-1)),
    ])


def ensure_model_dir():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)


def train_model(test_size=0.2, n_estimators=200, max_depth=None, min_samples_split=5, min_samples_leaf=2):
    print("Memuat dataset...")
    df = load_dataset()
    label_dist = df["label"].value_counts().to_dict()
    print(f"Total data: {len(df)} | Distribusi label: {label_dist}")
    X_train, X_test, y_train, y_test = split_dataset(df, test_size=test_size)
    print("Melatih model Random Forest...")
    model = build_pipeline(n_estimators, max_depth, min_samples_split, min_samples_leaf)
    model.fit(X_train, y_train)
    metrics = evaluate_model(model, X_test, y_test)
    print(f"Akurasi: {metrics['accuracy']:.2f}%")
    ensure_model_dir()
    joblib.dump(model, MODEL_PATH)
    print(f"Model disimpan: {MODEL_PATH}")
    return {"model": model, "metrics": metrics, "sample_count": len(df), "label_distribution": label_dist}


def load_model():
    if MODEL_PATH.exists():
        print(f"Memuat model dari: {MODEL_PATH}")
        return joblib.load(MODEL_PATH)
    print("Model belum ada, melatih model baru...")
    return train_model()["model"]


def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    report = classification_report(y_test, predictions, output_dict=True, zero_division=0)
    matrix = confusion_matrix(y_test, predictions).tolist()
    return {"accuracy": float(accuracy_score(y_test, predictions) * 100), "classification_report": report, "confusion_matrix": matrix}


def predict_text(model, title, text):
    content = f"{title or ''} {text or ''}".strip()
    if not content:
        return {"prediction": None, "probability": None, "message": "Masukkan judul atau isi berita."}
    proba = model.predict_proba([content])[0]
    prediction = int(model.predict([content])[0])
    return {"prediction": prediction, "probability": {"valid": float(proba[0]), "hoax": float(proba[1])}, "message": "Prediksi selesai.", "text": content}


def load_data_preview(sample_size=5):
    df = load_dataset()
    return df.sample(n=min(sample_size, len(df)), random_state=42)


if __name__ == "__main__":
    result = train_model()
    print("\n=== Hasil Evaluasi Model ===")
    print(f"Akurasi         : {result['metrics']['accuracy']:.2f}%")
    print(f"Confusion Matrix: {result['metrics']['confusion_matrix']}")
    report = result["metrics"]["classification_report"]
    print(f"\nValid - F1: {report['0']['f1-score']:.4f}")
    print(f"Hoax  - F1: {report['1']['f1-score']:.4f}")