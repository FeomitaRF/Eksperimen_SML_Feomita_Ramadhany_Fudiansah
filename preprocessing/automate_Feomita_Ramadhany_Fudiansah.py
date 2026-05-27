from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_PATH = ROOT_DIR / "breast_cancer_raw" / "breast_cancer_raw.csv"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "breast_cancer_preprocessing"
TARGET_COLUMN = "diagnosis"


def load_data(input_path: str | Path) -> pd.DataFrame:
    """Memuat dataset mentah dari file CSV."""
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"File dataset tidak ditemukan: {input_path}")
    return pd.read_csv(input_path)


def clean_data(df: pd.DataFrame, target_column: str = TARGET_COLUMN) -> pd.DataFrame:
    
    data = df.copy()
    data = data.dropna(axis=1, how="all")
    data = data.drop_duplicates().reset_index(drop=True)

    if target_column not in data.columns:
        raise ValueError(f"Kolom target '{target_column}' tidak ditemukan di dataset.")
    data = data.dropna(subset=[target_column]).reset_index(drop=True)

    return data


def cap_outliers_iqr(df: pd.DataFrame, numeric_columns: list[str]) -> pd.DataFrame:
    data = df.copy()
    for column in numeric_columns:
        q1 = data[column].quantile(0.25)
        q3 = data[column].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        data[column] = data[column].clip(lower=lower_bound, upper=upper_bound)
    return data


def preprocess_data(
    input_path: str | Path = DEFAULT_INPUT_PATH,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    target_column: str = TARGET_COLUMN,
) -> pd.DataFrame:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_data = load_data(input_path)
    cleaned_data = clean_data(raw_data, target_column=target_column)

    feature_columns = [column for column in cleaned_data.columns if column != target_column]
    numeric_columns = cleaned_data[feature_columns].select_dtypes(include=[np.number]).columns.tolist()

    if not numeric_columns:
        raise ValueError("Tidak ditemukan fitur numerik untuk diproses.")

    X = cleaned_data[numeric_columns].copy()
    y = cleaned_data[target_column].astype(int).copy()

    imputer = SimpleImputer(strategy="median")
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=numeric_columns)
    
    X_capped = cap_outliers_iqr(X_imputed, numeric_columns)

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X_capped), columns=numeric_columns)

    processed_data = X_scaled.copy()
    processed_data[target_column] = y.values

    output_path = output_dir / "breast_cancer_preprocessed.csv"
    processed_data.to_csv(output_path, index=False)

    print(f"Dataset raw shape       : {raw_data.shape}")
    print(f"Dataset processed shape : {processed_data.shape}")
    print(f"Hasil preprocessing disimpan di: {output_path}")

    return processed_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Automated data preprocessing for MSML submission.")
    parser.add_argument(
        "--input",
        default=str(DEFAULT_INPUT_PATH),
        help="Path menuju dataset raw CSV.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Folder output untuk dataset hasil preprocessing.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    preprocess_data(input_path=args.input, output_dir=args.output)
