import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import os
import time

# =========================
# SETUP OUTPUT FOLDER
# =========================
os.makedirs(r"D:\project mining\output", exist_ok=True)
output_path = r"D:\project mining\output"

# =========================
# LOAD DATASET
# =========================
start = time.time()
df = pd.read_csv(r"D:\project mining\data\processed\dataset_gabungan.csv")
print(f"✓ Load data: {time.time()-start:.2f}s")

print("\n=== INFO DATASET ===")
print(df.info())

print("\n=== MISSING VALUE ===")
print(df.isnull().sum())

# =========================
# LABEL DISTRIBUTION (FASTER)
# =========================
start = time.time()
plt.figure(figsize=(6,4))
label_counts = df["label"].value_counts().sort_index()
plt.bar(label_counts.index, label_counts.values, color=["#66b3ff","#ff9999"])
plt.title("Distribusi Label (Hoax vs Valid)")
plt.xlabel("Label")
plt.ylabel("Jumlah")
plt.xticks([0, 1], ["Valid", "Hoax"])
plt.tight_layout()
plt.savefig(f"{output_path}/01_label_distribution.png", dpi=80)
plt.close()
print(f"✓ Label distribution: {time.time()-start:.2f}s")

# =========================
# PIE CHART LABEL
# =========================
start = time.time()
plt.figure(figsize=(6,6))
df["label"].value_counts().plot.pie(
    autopct="%1.1f%%",
    startangle=90,
    colors=["#66b3ff","#ff9999"]
)
plt.title("Persentase Label Dataset")
plt.ylabel("")
plt.tight_layout()
plt.savefig(f"{output_path}/02_label_pie.png", dpi=80)
plt.close()
print(f"✓ Label pie: {time.time()-start:.2f}s")

# =========================
# TEXT LENGTH FEATURE
# =========================
start = time.time()
df["text_length"] = df["clean_text"].astype(str).apply(len)

plt.figure(figsize=(8,5))
sns.histplot(df["text_length"], bins=30, kde=False, color="purple")
plt.title("Distribusi Panjang Text")
plt.xlabel("Jumlah Karakter")
plt.ylabel("Frekuensi")
plt.tight_layout()
plt.savefig(f"{output_path}/03_text_length.png", dpi=80)
plt.close()
print(f"✓ Text length histogram: {time.time()-start:.2f}s")

# =========================
# BOX PLOT TEXT LENGTH
# =========================
start = time.time()
plt.figure(figsize=(7,5))
sns.boxplot(x="label", y="text_length", data=df)
plt.title("Perbandingan Panjang Text")
plt.xlabel("Label")
plt.ylabel("Panjang Text")
plt.tight_layout()
plt.savefig(f"{output_path}/04_boxplot.png", dpi=80)
plt.close()
print(f"✓ Boxplot: {time.time()-start:.2f}s")

# =========================
# TOP 10 WORDS (FAST)
# =========================
start = time.time()
# Sample 5000 rows untuk word counting lebih cepat
sample_text = df["clean_text"].sample(n=min(5000, len(df)), random_state=42).astype(str)
all_words = " ".join(sample_text).split()[:50000]
word_counts = Counter(all_words)

top_words = word_counts.most_common(10)
words, counts = zip(*top_words)

plt.figure(figsize=(8,5))
sns.barplot(x=list(counts), y=list(words))
plt.title("Top 10 Kata Paling Sering (Sample)")
plt.xlabel("Frekuensi")
plt.ylabel("Kata")
plt.tight_layout()
plt.savefig(f"{output_path}/05_top_words.png", dpi=80)
plt.close()
print(f"✓ Top words: {time.time()-start:.2f}s")

# =========================
# SAMPLE DATA
# =========================
print("\n=== SAMPLE DATA ===")
print(df[["judul", "label", "clean_text"]].head())

print("\n✔ EDA SELESAI! Semua visual tersimpan di folder output/")