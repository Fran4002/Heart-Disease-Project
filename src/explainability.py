"""
Explainability and SHAP Analysis Module.
Enables computing SHapley Additive exPlanations (SHAP) and aggregating
importance across functional groups (Clinical vs Lifestyle vs Wearable vs Demographic).
"""
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

from src.config import (
    DEMOGRAPHIC_COLS,
    LIFESTYLE_COLS,
    MEDICAL_HISTORY_COLS,
    CLINICAL_MARKER_COLS,
    WEARABLE_COLS,
    FIGURES_DIR,
)


def compute_tree_shap_values(
    model,
    X_sample: pd.DataFrame,
    max_evals: int = 1000,
) -> shap.Explanation:
    """
    Computes SHAP values using TreeExplainer for tree ensemble models.
    """
    # If model is a pipeline, extract the classifier
    if hasattr(model, "named_steps") and "classifier" in model.named_steps:
        clf = model.named_steps["classifier"]
        # Transform X through the preprocessor
        if "preprocessor" in model.named_steps:
            X_trans = model.named_steps["preprocessor"].transform(X_sample)
            feature_names = model.named_steps["preprocessor"].get_feature_names_out()
            X_df = pd.DataFrame(X_trans, columns=feature_names, index=X_sample.index)
        else:
            X_df = X_sample
    else:
        clf = model
        X_df = X_sample

    explainer = shap.TreeExplainer(clf)
    shap_values = explainer(X_df)
    return shap_values


def aggregate_group_importance(
    shap_values: shap.Explanation,
) -> pd.DataFrame:
    """
    Aggregates mean absolute SHAP values across feature taxonomy groups:
    - Clinical Markers
    - Medical History
    - Lifestyle Factors
    - Wearables
    - Demographics
    
    This provides the direct quantitative answer to the primary research question!
    """
    feature_names = shap_values.feature_names
    # Compute mean |SHAP| per individual feature
    vals = np.abs(shap_values.values)
    # Handle binary classification (if 3D array: [samples, features, classes])
    if len(vals.shape) == 3:
        vals = vals[:, :, 1]
    mean_abs_shap = np.mean(vals, axis=0)

    feature_importance_series = pd.Series(mean_abs_shap, index=feature_names)

    # Helper function to classify feature names into our taxonomy groups
    def map_feature_to_group(name: str) -> str:
        clean_name = name.split("__")[-1]  # remove preprocessor prefix
        if any(c in clean_name for c in CLINICAL_MARKER_COLS):
            return "Marcadores Clínicos"
        if any(c in clean_name for c in MEDICAL_HISTORY_COLS):
            return "Historial Médico"
        if any(c in clean_name for c in LIFESTYLE_COLS):
            return "Estilo de Vida (Modificable)"
        if any(c in clean_name for c in WEARABLE_COLS):
            return "Dispositivos Vestibles (Wearables)"
        if any(c in clean_name for c in DEMOGRAPHIC_COLS):
            return "Demografía"
        return "Otros / Derivados"

    group_mapping = {feat: map_feature_to_group(feat) for feat in feature_names}
    df_imp = pd.DataFrame({
        "Feature": feature_names,
        "Mean_Abs_SHAP": mean_abs_shap,
        "Group": [group_mapping[f] for f in feature_names],
    })

    # Grouped summary
    group_summary = (
        df_imp.groupby("Group")["Mean_Abs_SHAP"]
        .sum()
        .reset_index()
        .sort_values(by="Mean_Abs_SHAP", ascending=False)
    )
    total_shap = group_summary["Mean_Abs_SHAP"].sum()
    group_summary["Percentage_Contribution"] = (group_summary["Mean_Abs_SHAP"] / total_shap) * 100.0

    return group_summary

