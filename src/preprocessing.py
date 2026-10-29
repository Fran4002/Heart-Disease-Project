"""
Preprocessing Pipelines and Column Transformers.
Builds scikit-learn ColumnTransformers tailored for linear models (scaling + one-hot)
and tree-based ensembles (ordinal / categorical handling).
"""
from typing import List, Optional
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
    RobustScaler,
)

from src.config import (
    NOMINAL_COLS,
    ORDINAL_COLS,
    ORDINAL_CATEGORIES,
    BINARY_NUMERIC_COLS,
    CONTINUOUS_NUMERIC_COLS,
)


def build_linear_preprocessor(
    nominal_cols: Optional[List[str]] = None,
    ordinal_cols: Optional[List[str]] = None,
    continuous_cols: Optional[List[str]] = None,
    binary_cols: Optional[List[str]] = None,
    scaler_type: str = "standard",
) -> ColumnTransformer:
    """
    Build a ColumnTransformer suitable for linear and distance-based models
    (Logistic Regression, SVM, Neural Nets).
    
    Requirements for linear models:
    - Continuous features MUST be standardized (mean=0, variance=1) or robustly scaled.
    - Nominal categorical variables should be One-Hot Encoded (with drop='first' to avoid collinearity dummy variable trap).
    - Ordinal variables should be monotonically encoded with explicit category ranking.
    - Binary indicators (0/1) can be passed through without transformation.
    """
    nominal_cols = nominal_cols or NOMINAL_COLS
    ordinal_cols = ordinal_cols or ORDINAL_COLS
    continuous_cols = continuous_cols or CONTINUOUS_NUMERIC_COLS
    binary_cols = binary_cols or BINARY_NUMERIC_COLS

    # ==========================================================================
    # TODO: Configure Scaler (StandardScaler or RobustScaler)
    # ==========================================================================
    # Hint: StandardScaler assumes Gaussian distributions, RobustScaler uses medians & IQR
    # which is less sensitive to extreme outliers.
    if scaler_type == "robust":
        numeric_scaler = RobustScaler()
    else:
        numeric_scaler = StandardScaler()

    # ==========================================================================
    # TODO: Configure OrdinalEncoder with explicit rankings
    # ==========================================================================
    # Why explicit order?
    # Default OrdinalEncoder assigns values alphabetically ("High": 0, "Low": 1, "Middle": 2),
    # which completely corrupts the monotonic socio-economic or activity ranking!
    # Explicit categories order ensures: Low=0, Middle=1, High=2.
    categories_list = [ORDINAL_CATEGORIES[col] for col in ordinal_cols]
    ordinal_encoder = OrdinalEncoder(
        categories=categories_list,
        handle_unknown="use_encoded_value",
        unknown_value=-1,
    )

    # ==========================================================================
    # TODO: Configure OneHotEncoder for nominal variables
    # ==========================================================================
    # For linear models: drop='first' is essential to eliminate perfect multicollinearity.
    one_hot_encoder = OneHotEncoder(
        drop="first",
        handle_unknown="ignore",
        sparse_output=False,
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num_continuous", numeric_scaler, continuous_cols),
            ("num_binary", "passthrough", binary_cols),
            ("cat_ordinal", ordinal_encoder, ordinal_cols),
            ("cat_nominal", one_hot_encoder, nominal_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )

    return preprocessor


def build_tree_preprocessor(
    categorical_cols: Optional[List[str]] = None,
    numeric_cols: Optional[List[str]] = None,
) -> ColumnTransformer:
    """
    Build a ColumnTransformer optimized for Tree-based models (Random Forest, LightGBM, XGBoost).
    
    Tree models are invariant to monotonic monotonic transformations (no need to scale numeric features).
    They can take integer/ordinal encodings directly or one-hot encodings.
    """
    # For tree models, pass numeric through and encode categoricals
    cat_cols = categorical_cols or (NOMINAL_COLS + ORDINAL_COLS)
    num_cols = numeric_cols or (CONTINUOUS_NUMERIC_COLS + BINARY_NUMERIC_COLS)

    tree_encoder = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False,
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", "passthrough", num_cols),
            ("categorical", tree_encoder, cat_cols),
        ],
        remainder="drop",
    )

    return preprocessor


def verify_preprocessing_pipeline():
    """Verification function to test transformations on the processed dataset."""
    from src.data_loader import load_processed_splits
    from src.config import TARGET_COL

    train_df, _, _ = load_processed_splits()
    X_train = train_df.drop(columns=[TARGET_COL])

    preprocessor = build_linear_preprocessor()
    X_transformed = preprocessor.fit_transform(X_train)

    feature_names = preprocessor.get_feature_names_out()
    print(f"Transformed feature matrix shape: {X_transformed.shape}")
    print(f"Total encoded features: {len(feature_names)}")
    print(f"Sample transformed feature names:\n{list(feature_names[:10])}")
    return X_transformed


if __name__ == "__main__":
    verify_preprocessing_pipeline()

