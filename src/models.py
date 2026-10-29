"""
Model Definition, Pipeline Builders, and Calibration Wrappers.
Provides standard model templates for baseline logistic regression, ensemble trees,
and probability calibration.
"""
from typing import Dict, Any, Optional
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
import lightgbm as lgb
import xgboost as xgb

from src.config import RANDOM_STATE
from src.preprocessing import build_linear_preprocessor, build_tree_preprocessor


def build_logistic_baseline_pipeline(C: float = 1.0, max_iter: int = 1000) -> Pipeline:
    """
    Constructs an end-to-end Pipeline combining linear preprocessing and Logistic Regression.
    """
    preprocessor = build_linear_preprocessor()
    classifier = LogisticRegression(
        C=C,
        max_iter=max_iter,
        random_state=RANDOM_STATE,
        solver="lbfgs",
    )
    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier),
    ])


def build_random_forest_pipeline(
    n_estimators: int = 200,
    max_depth: Optional[int] = 12,
    min_samples_split: int = 10,
    n_jobs: int = -1,
) -> Pipeline:
    """
    Constructs an end-to-end Pipeline combining tree preprocessing and Random Forest.
    """
    preprocessor = build_tree_preprocessor()
    classifier = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=RANDOM_STATE,
        n_jobs=n_jobs,
    )
    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier),
    ])


def build_lightgbm_pipeline(
    n_estimators: int = 300,
    learning_rate: float = 0.05,
    max_depth: int = 6,
    num_leaves: int = 31,
    subsample: float = 0.8,
) -> Pipeline:
    """
    Constructs an end-to-end Pipeline with LightGBM Classifier.
    """
    preprocessor = build_tree_preprocessor()
    classifier = lgb.LGBMClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        num_leaves=num_leaves,
        subsample=subsample,
        random_state=RANDOM_STATE,
        verbose=-1,
        n_jobs=-1,
    )
    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier),
    ])


def calibrate_model(
    base_estimator,
    method: str = "sigmoid",
    cv: str = "prefit",
) -> CalibratedClassifierCV:
    """
    Wraps an estimator with Scikit-Learn's CalibratedClassifierCV.
    
    Methods:
    - 'sigmoid' (Platt Scaling): Fits logistic regression mapping model output to probabilities.
      Best suited for smaller samples or well-behaved margins.
    - 'isotonic' (Isotonic Regression): Non-parametric monotonic step function.
      More flexible but requires larger validation samples to avoid overfitting.
    """
    return CalibratedClassifierCV(
        estimator=base_estimator,
        method=method,
        cv=cv,
    )

