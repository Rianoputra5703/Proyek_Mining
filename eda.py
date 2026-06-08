import sys
from pathlib import Path

# === FIX IMPORT: tambahkan root project ke sys.path ===
# Ini memastikan 'preprocessing' bisa diimport dari mana pun script dijalankan
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# Import lokal — tidak perlu prefix 'src.'
from preprocessing import load_dataset

OUTPUT_PATH = ROOT / "output"
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)


def ensure_text_length(df):
    if "text_length" not in df.columns:
        df = df.copy()
        df["text_length"] = df["clean_text"].astype(str).apply(len)
    return df


def plot_label_distribution(df):
    label_counts = df["label"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(label_counts.index, label_counts.values, color=["#66b3ff", "#ff9999"])
    ax.set_title("Distribusi Label (Hoax vs Valid)")
    ax.set_xlabel("Label")
    ax.set_ylabel("Jumlah")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Valid", "Hoax"])
    fig.tight_layout()
    return fig


def plot_label_pie(df):
    label_counts = df["label"].value_counts()
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(
        label_counts.values,
        labels=["Valid", "Hoax"],
        autopct="%1.1f%%",
        startangle=90,
        colors=["#66b3ff", "#ff9999"],
    )
    ax.set_title("Persentase Label Dataset")
    ax.axis("equal")
    fig.tight_layout()
    return fig


def plot_text_length_hist(df):
    df = ensure_text_length(df)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(df["text_length"], bins=30, kde=False, color="purple", ax=ax)
    ax.set_title("Distribusi Panjang Text")
    ax.set_xlabel("Jumlah Karakter")
    ax.set_ylabel("Frekuensi")
    fig.tight_layout()
    return fig


def plot_text_length_box(df):
    df = ensure_text_length(df)
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.boxplot(x="label", y="text_length", data=df, ax=ax)
    ax.set_title("Perbandingan Panjang Text per Label")
    ax.set_xlabel("Label (0=Valid, 1=Hoax)")
    ax.set_ylabel("Panjang Text")
    fig.tight_layout()
    return fig


def plot_top_words(df, sample_size: int = 5000, top_n: int = 10):
    sample_text = (
        df["clean_text"]
        .sample(n=min(sample_size, len(df)), random_state=42)
        .astype(str)
    )
    all_words = " ".join(sample_text).split()[:50000]
    word_counts = Counter(all_words)
    top_words = word_counts.most_common(top_n)
    words, counts = zip(*top_words)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x=list(counts), y=list(words), ax=ax)
    ax.set_title(f"Top {top_n} Kata Paling Sering (Sample)")
    ax.set_xlabel("Frekuensi")
    ax.set_ylabel("Kata")
    fig.tight_layout()
    return fig


def get_sample_data(df, sample_size: int = 5):
    return df[["judul", "label", "clean_text"]].sample(
        n=min(sample_size, len(df)), random_state=42
    )


def save_figure(fig, filename: str):
    fig.savefig(OUTPUT_PATH / filename, dpi=80)
    plt.close(fig)


def generate_all_figures(df):
    figs = {
        "label_distribution": plot_label_distribution(df),
        "label_pie":          plot_label_pie(df),
        "text_length_hist":   plot_text_length_hist(df),
        "text_length_box":    plot_text_length_box(df),
        "top_words":          plot_top_words(df),
    }
    return figs


# Jalankan standalone untuk test
if __name__ == "__main__":
    print("Memuat dataset...")
    df = load_dataset()
    print(f"Dataset dimuat: {len(df)} baris")
    figs = generate_all_figures(df)
    for name, fig in figs.items():
        save_figure(fig, f"{name}.png")
        print(f"Saved: output/{name}.png")
    print("EDA selesai!")