import sys
from pathlib import Path

# === FIX IMPORT: tambahkan root project ke sys.path ===
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Import lokal
from preprocessing import load_dataset, split_dataset

MODEL_DIR   = ROOT / "output" / "model"
MODEL_PATH  = MODEL_DIR / "hoax_detector.joblib"
MLFLOW_DB   = ROOT / "mlflow.db"

# Gunakan SQLite sebagai backend (MLflow versi baru tidak support file store)
mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB}")
mlflow.set_experiment("hoax-detection-random-forest")


def build_pipeline(
    n_estimators: int = 200,
    max_depth=None,
    min_samples_split: int = 5,
    min_samples_leaf: int = 2,
) -> Pipeline:
    return Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                max_features=25000,
                ngram_range=(1, 2),
                lowercase=True,
                token_pattern=r"(?u)\b\w\w+\b",
            ),
        ),
        (
            "clf",
            RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            ),
        ),
    ])


def ensure_model_dir() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)


def train_model(
    test_size: float = 0.2,
    n_estimators: int = 200,
    max_depth=None,
    min_samples_split: int = 5,
    min_samples_leaf: int = 2,
) -> dict:
    """Melatih model Random Forest dengan MLflow tracking."""
    print("Memuat dataset...")
    df = load_dataset()
    label_dist = df["label"].value_counts().to_dict()
    print(f"Total data: {len(df)} | Distribusi label: {label_dist}")

    X_train, X_test, y_train, y_test = split_dataset(df, test_size=test_size)

    with mlflow.start_run():
        # ── Log parameter ──────────────────────────────────────────────────
        mlflow.log_params({
            "model":              "RandomForestClassifier",
            "n_estimators":       n_estimators,
            "max_depth":          str(max_depth),
            "min_samples_split":  min_samples_split,
            "min_samples_leaf":   min_samples_leaf,
            "class_weight":       "balanced",
            "tfidf_max_features": 25000,
            "tfidf_ngram_range":  "(1, 2)",
            "test_size":          test_size,
            "total_data":         len(df),
            "label_0_valid":      label_dist.get(0, 0),
            "label_1_hoax":       label_dist.get(1, 0),
        })

        # ── Latih model ────────────────────────────────────────────────────
        print("Melatih model Random Forest (mungkin memakan beberapa menit)...")
        model = build_pipeline(n_estimators, max_depth, min_samples_split, min_samples_leaf)
        model.fit(X_train, y_train)

        # ── Evaluasi & log metrik ──────────────────────────────────────────
        metrics = evaluate_model(model, X_test, y_test)
        report  = metrics["classification_report"]

        mlflow.log_metrics({
            "accuracy":         metrics["accuracy"],
            "valid_precision":  report["0"]["precision"],
            "valid_recall":     report["0"]["recall"],
            "valid_f1":         report["0"]["f1-score"],
            "hoax_precision":   report["1"]["precision"],
            "hoax_recall":      report["1"]["recall"],
            "hoax_f1":          report["1"]["f1-score"],
            "macro_f1":         report["macro avg"]["f1-score"],
            "weighted_f1":      report["weighted avg"]["f1-score"],
        })

        # ── Log model ke MLflow ────────────────────────────────────────────
        mlflow.sklearn.log_model(model, artifact_path="model")

        run_id = mlflow.active_run().info.run_id
        print(f"MLflow run_id : {run_id}")
        print(f"Akurasi       : {metrics['accuracy']:.2f}%")

    # ── Simpan juga ke disk (untuk Streamlit) ─────────────────────────────
    ensure_model_dir()
    joblib.dump(model, MODEL_PATH)
    print(f"Model disimpan: {MODEL_PATH}")

    return {
        "model": model,
        "metrics": metrics,
        "sample_count": len(df),
        "label_distribution": label_dist,
    }


def load_model() -> Pipeline:
    """Memuat model dari disk; latih ulang jika belum ada."""
    if MODEL_PATH.exists():
        print(f"Memuat model dari: {MODEL_PATH}")
        return joblib.load(MODEL_PATH)
    print("Model belum ada, melatih model baru...")
    result = train_model()
    return result["model"]


def evaluate_model(model: Pipeline, X_test: pd.Series, y_test: pd.Series) -> dict:
    """Mengevaluasi model dan mengembalikan metrik."""
    predictions = model.predict(X_test)
    report  = classification_report(y_test, predictions, output_dict=True, zero_division=0)
    matrix  = confusion_matrix(y_test, predictions).tolist()
    return {
        "accuracy": float(accuracy_score(y_test, predictions) * 100),
        "classification_report": report,
        "confusion_matrix": matrix,
    }


def predict_text(model: Pipeline, title: str, text: str) -> dict:
    """Melakukan prediksi pada teks berita."""
    content = f"{title or ''} {text or ''}".strip()
    if not content:
        return {
            "prediction": None,
            "probability": None,
            "message": "Masukkan judul atau isi berita untuk melakukan prediksi.",
        }
    proba      = model.predict_proba([content])[0]
    prediction = int(model.predict([content])[0])
    return {
        "prediction": prediction,
        "probability": {
            "valid": float(proba[0]),
            "hoax":  float(proba[1]),
        },
        "message": "Prediksi selesai.",
        "text": content,
    }


def load_data_preview(sample_size: int = 5) -> pd.DataFrame:
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
    print("\nJalankan MLflow UI dengan:")
    print("  mlflow ui")
    print("Buka browser: http://localhost:5000")