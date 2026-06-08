import pandas as pd
import os
from pathlib import Path

# === PATH KONFIGURASI ===
# Gunakan path relatif dari lokasi file ini
ROOT = Path(__file__).resolve().parent

kompas_file = ROOT / "data" / "raw" / "Cleaned_Kompas_v2 valid.csv"
hoax_file   = ROOT / "data" / "raw" / "Cleaned_TurnBackHoax_v3.csv"

# Kolom yang wajib ada
required_columns = ["url", "judul", "narasi", "label", "clean_text"]


def load_and_validate(filepath: Path, name: str) -> pd.DataFrame:
    """Membaca CSV dan memvalidasi kolom yang diperlukan."""
    if not filepath.exists():
        raise FileNotFoundError(
            f"File '{name}' tidak ditemukan di: {filepath}\n"
            f"Pastikan path sudah benar."
        )
    print(f"Membaca dataset {name}...")
    df = pd.read_csv(filepath)
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(
                f"Kolom '{col}' tidak ditemukan pada dataset {name}.\n"
                f"Kolom yang tersedia: {list(df.columns)}"
            )
    return df[required_columns]


def main():
    df_valid = load_and_validate(kompas_file, "Kompas")
    df_hoax  = load_and_validate(hoax_file, "TurnBackHoax")

    print(f"Jumlah data valid : {len(df_valid)}")
    print(f"Jumlah data hoax  : {len(df_hoax)}")

    # Gabungkan dan hapus duplikat
    df = pd.concat([df_valid, df_hoax], ignore_index=True)
    df = df.drop_duplicates()
    print(f"Total data setelah digabung : {len(df)}")

    # Simpan hasil
    output_dir = ROOT / "data" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "dataset_gabungan.csv"
    df.to_csv(output_file, index=False, encoding="utf-8-sig")

    print(f"\nDataset berhasil disimpan:\n{output_file}")
    print("\n5 Data Pertama:")
    print(df.head())


if __name__ == "__main__":
    main()