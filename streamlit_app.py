import sys
from pathlib import Path

# === FIX IMPORT: tambahkan root project ke sys.path ===
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st
import pandas as pd

# Import lokal — tidak perlu prefix 'src.'
from eda import (
    get_sample_data,
    plot_label_distribution,
    plot_label_pie,
    plot_text_length_hist,
    plot_text_length_box,
    plot_top_words,
)
from modeling import MODEL_PATH, load_model, train_model, predict_text
from preprocessing import load_dataset, get_label_distribution


def format_metrics(metrics: dict) -> str:
    accuracy     = metrics.get("accuracy", 0)
    report       = metrics.get("classification_report", {})
    valid_report = report.get("0", {})
    hoax_report  = report.get("1", {})
    return (
        f"Akurasi: {accuracy:.4f}\n"
        f"\nValid - Precision: {valid_report.get('precision', 0):.4f}, "
        f"Recall: {valid_report.get('recall', 0):.4f}, "
        f"F1: {valid_report.get('f1-score', 0):.4f}\n"
        f"Hoax  - Precision: {hoax_report.get('precision', 0):.4f}, "
        f"Recall: {hoax_report.get('recall', 0):.4f}, "
        f"F1: {hoax_report.get('f1-score', 0):.4f}\n"
    )


def main() -> None:
    st.set_page_config(
        page_title="Deteksi Hoax Berita",
        page_icon="📰",
        layout="wide",
    )

    st.title("Deteksi Hoax Berita Kompas")
    st.write(
        "Aplikasi ini memprediksi apakah sebuah berita valid atau hoax "
        "menggunakan model klasifikasi teks yang dibuat dari dataset Kompas."
    )

    with st.expander("Tentang Dataset dan Model", expanded=True):
        st.write(
            "Dataset memuat berita yang telah dibersihkan dalam kolom `clean_text` "
            "dengan label 0 = valid dan 1 = hoax. "
            "Model menggunakan TF-IDF dan Logistic Regression untuk mendeteksi pola teks."
        )

    df      = load_dataset()
    model   = load_model() if MODEL_PATH.exists() else None
    metrics = None

    # ── Sidebar ──────────────────────────────────────────────────────────────
    st.sidebar.header("Pengaturan")
    sample_size  = st.sidebar.slider("Jumlah sampel dataset", min_value=3, max_value=20, value=5, step=1)
    train_button = st.sidebar.button("Latih/Ulang Model")
    st.sidebar.markdown("---")
    st.sidebar.write("Label: 0 = Valid, 1 = Hoax")
    st.sidebar.write("Model status:")
    if MODEL_PATH.exists():
        st.sidebar.success("Model tersimpan di output/model.")
    else:
        st.sidebar.warning("Model belum tersedia.")

    if train_button:
        with st.spinner("Melatih model, harap tunggu..."):
            train_result = train_model()
        model   = train_result["model"]
        metrics = train_result["metrics"]
        st.sidebar.success("Model berhasil dilatih ulang dan disimpan.")

    # ── Tabs ─────────────────────────────────────────────────────────────────
    overview_tab, eda_tab, prediction_tab = st.tabs(["Overview", "EDA", "Prediction"])

    with overview_tab:
        st.header("Ringkasan Dataset")
        label_distribution = get_label_distribution(df)
        st.bar_chart(
            pd.DataFrame({"Jumlah": label_distribution})
            .rename_axis("Label")
            .reset_index()
        )
        st.write("Jumlah total data:", len(df))
        st.write("Label 0 = Valid, Label 1 = Hoax")

        if st.checkbox("Tampilkan sampel dataset", value=False):
            st.dataframe(get_sample_data(df, sample_size))

        st.markdown("---")
        st.write(
            "Gunakan tab EDA untuk melihat visualisasi, "
            "dan tab Prediction untuk mencoba prediksi berita baru."
        )

    with eda_tab:
        st.header("Visualisasi EDA")
        st.write(
            "Grafik ini menunjukkan distribusi label, sebaran panjang teks, "
            "dan kata-kata paling sering dalam dataset."
        )

        col1, col2 = st.columns(2)
        with col1:
            st.pyplot(plot_label_distribution(df))
            st.pyplot(plot_text_length_hist(df))
        with col2:
            st.pyplot(plot_label_pie(df))
            st.pyplot(plot_text_length_box(df))

        st.markdown("---")
        st.pyplot(plot_top_words(df))

    with prediction_tab:
        st.header("Prediksi Berita Baru")
        title_input = st.text_input("Judul berita")
        body_input  = st.text_area("Isi berita / narasi")

        if st.button("Prediksi"):
            if model is None:
                st.warning(
                    "Model belum tersedia. Tekan tombol 'Latih/Ulang Model' "
                    "di sidebar terlebih dahulu."
                )
            else:
                result = predict_text(model, title_input, body_input)
                if result["prediction"] is None:
                    st.warning(result["message"])
                else:
                    label = "Hoax" if result["prediction"] == 1 else "Valid"
                    st.write("**Hasil Prediksi**")
                    st.write(f"- Prediksi: **{label}**")
                    st.write(f"- Probabilitas Valid : {result['probability']['valid'] * 100:.2f}%")
                    st.write(f"- Probabilitas Hoax  : {result['probability']['hoax']  * 100:.2f}%")
                    st.markdown("---")
                    st.write("**Teks yang diprediksi:**")
                    st.write(result["text"])

        if MODEL_PATH.exists() or metrics is not None:
            st.markdown("---")
            st.subheader("Evaluasi Model")
            if metrics is not None:
                st.text(format_metrics(metrics))
            else:
                st.info(
                    "Model sudah tersimpan. Latih ulang model untuk memperbarui metrik evaluasi."
                )


if __name__ == "__main__":
    main()