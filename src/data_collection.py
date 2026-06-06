import pandas as pd
import os

# Path dataset
kompas_file = r"D:\project mining\data\raw\Cleaned_Kompas_v2 valid.csv"
hoax_file = r"D:\project mining\data\raw\Cleaned_TurnBackHoax_v3.csv"

print("Membaca dataset Kompas...")
df_valid = pd.read_csv(kompas_file)

print("Membaca dataset TurnBackHoax...")
df_hoax = pd.read_csv(hoax_file)

# Kolom yang wajib ada
required_columns = [
    "url",
    "judul",
    "narasi",
    "label",
    "clean_text"
]

# Cek dataset Kompas
for col in required_columns:
    if col not in df_valid.columns:
        raise ValueError(
            f"Kolom '{col}' tidak ditemukan pada dataset Kompas"
        )

# Cek dataset Hoax
for col in required_columns:
    if col not in df_hoax.columns:
        raise ValueError(
            f"Kolom '{col}' tidak ditemukan pada dataset TurnBackHoax"
        )

# Ambil kolom yang diperlukan
df_valid = df_valid[required_columns]
df_hoax = df_hoax[required_columns]

print("Jumlah data valid :", len(df_valid))
print("Jumlah data hoax  :", len(df_hoax))

# Gabungkan dataset
df = pd.concat([df_valid, df_hoax], ignore_index=True)

# Hapus duplikat
df = df.drop_duplicates()

print("Total data setelah digabung :", len(df))

# Buat folder output
os.makedirs(r"D:\project mining\data\processed", exist_ok=True)

# Simpan hasil
output_file = r"D:\project mining\data\processed\dataset_gabungan.csv"

df.to_csv(
    output_file,
    index=False,
    encoding="utf-8-sig"
)

print("\nDataset berhasil disimpan:")
print(output_file)

print("\n5 Data Pertama:")
print(df.head())