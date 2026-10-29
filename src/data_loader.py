"""
Data Loading, Verification, and Stratified Partitioning Utilities.
Handles raw dataset loading, schema verification, and 70/15/15 stratified splitting.
"""
from pathlib import Path
from typing import Tuple, Optional
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    RAW_DATA_PATH,
    PROCESSED_DATA_DIR,
    TARGET_COL,
    ALL_FEATURE_COLS,
    RANDOM_STATE,
    TRAIN_RATIO,
    VAL_RATIO,
    TEST_RATIO,
)


def load_raw_data(filepath: Optional[Path] = None) -> pd.DataFrame:
    """
    Load raw cardiovascular dataset from disk and perform basic integrity verification.
    """
    path = Path(filepath or RAW_DATA_PATH)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Please ensure 'heart_disease_prediction_2026.csv' exists in data/."
        )

    df = pd.read_csv(path)
    
    # Sanity checks
    missing_cols = set(ALL_FEATURE_COLS + [TARGET_COL]) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Dataset is missing required columns: {missing_cols}")

    return df


def split_data(
    df: pd.DataFrame,
    val_ratio: float = VAL_RATIO,
    test_ratio: float = TEST_RATIO,
    random_state: int = RANDOM_STATE,
    save: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Perform a 3-way stratified partition: Train (70%), Val (15%), Test (15%).
    Preserves class balance (heart_disease ~ 46.1%) across all partitions.
    
    Returns:
        (df_train, df_val, df_test)
    """
    assert val_ratio + test_ratio < 1.0, "Validation + Test ratios must sum to less than 1.0"
    
    n_total = len(df)
    n_test = int(round(n_total * test_ratio))
    n_val = int(round(n_total * val_ratio))

    # Step 1: Separate Test set (exact integer count)
    df_train_val, df_test = train_test_split(
        df,
        test_size=n_test,
        random_state=random_state,
        stratify=df[TARGET_COL],
    )

    # Step 2: Separate Validation set from Train+Val (exact integer count)
    df_train, df_val = train_test_split(
        df_train_val,
        test_size=n_val,
        random_state=random_state,
        stratify=df_train_val[TARGET_COL],
    )

    if save:
        save_splits(df_train, df_val, df_test)

    return df_train.reset_index(drop=True), df_val.reset_index(drop=True), df_test.reset_index(drop=True)


def save_splits(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    output_dir: Optional[Path] = None,
) -> None:
    """Save stratified partitions to data/processed/."""
    out = Path(output_dir or PROCESSED_DATA_DIR)
    out.mkdir(parents=True, exist_ok=True)
    df_train.to_csv(out / "train.csv", index=False)
    df_val.to_csv(out / "val.csv", index=False)
    df_test.to_csv(out / "test.csv", index=False)


def load_processed_splits(
    processed_dir: Optional[Path] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load train, val, and test partitions from data/processed/."""
    p_dir = Path(processed_dir or PROCESSED_DATA_DIR)
    train_path = p_dir / "train.csv"
    val_path = p_dir / "val.csv"
    test_path = p_dir / "test.csv"

    if not (train_path.exists() and val_path.exists() and test_path.exists()):
        raise FileNotFoundError(
            f"Processed splits not found in {p_dir}. Run `split_data(save=True)` first."
        )

    return pd.read_csv(train_path), pd.read_csv(val_path), pd.read_csv(test_path)
