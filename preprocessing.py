import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

# ROOT = dua level di atas file ini (misal: project_root/)
ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "processed" / "dataset_gabungan.csv"


def load_dataset(path: Path | str = DATA_PATH) -> pd.DataFrame:
    """Memuat dataset dari CSV dan melakukan validasi dasar."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset tidak ditemukan di: {path}\n"
            f"Jalankan data_collection.py terlebih dahulu untuk membuat dataset."
        )
    df = pd.read_csv(path)
    df = df.dropna(subset=["judul", "clean_text", "label"])
    df["judul"]      = df["judul"].astype(str)
    df["clean_text"] = df["clean_text"].astype(str)
    df["label"]      = df["label"].astype(int)
    df["text"]       = df["judul"] + " " + df["clean_text"]
    return df


def split_dataset(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Membagi dataset menjadi train dan test set."""
    X = df["text"]
    y = df["label"]
    return train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )


def get_label_distribution(df: pd.DataFrame) -> pd.Series:
    """Mengembalikan distribusi label."""
    return df["label"].value_counts().sort_index()