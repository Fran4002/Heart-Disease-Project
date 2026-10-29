"""
Evaluation and Visualization Utilities for Cardiovascular Risk Modeling.
Provides clinical metric computation, ROC/PR curves, reliability/calibration plots,
and subgroup fairness auditing tools.
"""
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    recall_score,
    precision_score,
    f1_score,
    accuracy_score,
    confusion_matrix,
    brier_score_loss,
    RocCurveDisplay,
    PrecisionRecallDisplay,
    ConfusionMatrixDisplay,
)
from sklearn.calibration import calibration_curve

from src.config import PALETTE, POS_COLOR, NEG_COLOR, FIGURES_DIR, TABLES_DIR


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Compute comprehensive clinical classification and calibration metrics.
    Emphasizes Recall (Sensitivity) and Brier Score for medical decision making.
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    metrics = {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Sensitivity (Recall)": float(recall_score(y_true, y_pred)),
        "Specificity": float(specificity),
        "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "F1-Score": float(f1_score(y_true, y_pred, zero_division=0)),
        "True Positives": int(tp),
        "False Negatives": int(fn),
        "False Positives": int(fp),
        "True Negatives": int(tn),
    }

    if y_proba is not None:
        metrics["ROC-AUC"] = float(roc_auc_score(y_true, y_proba))
        metrics["PR-AUC"] = float(average_precision_score(y_true, y_proba))
        metrics["Brier Score"] = float(brier_score_loss(y_true, y_proba))

    return metrics


def metrics_to_dataframe(metrics_dict_of_models: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    """Format dictionary of multiple models' metrics into a styled DataFrame."""
    df = pd.DataFrame(metrics_dict_of_models).T
    # Reorder columns logically
    primary_cols = [
        "ROC-AUC",
        "PR-AUC",
        "Brier Score",
        "Sensitivity (Recall)",
        "Specificity",
        "Precision",
        "F1-Score",
        "Accuracy",
    ]
    existing_primary = [c for c in primary_cols if c in df.columns]
    other_cols = [c for c in df.columns if c not in existing_primary]
    return df[existing_primary + other_cols]


def plot_roc_pr_comparison(
    models_dict: Dict[str, Any],
    X_eval,
    y_eval,
    save_name: Optional[str] = None,
) -> None:
    """
    Plot ROC and Precision-Recall curves side by side for all fitted models.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for idx, (name, model) in enumerate(models_dict.items()):
        color = PALETTE[idx % len(PALETTE)]
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_eval)[:, 1]
        elif hasattr(model, "decision_function"):
            y_proba = model.decision_function(X_eval)
        else:
            continue

        RocCurveDisplay.from_predictions(
            y_eval, y_proba, name=name, ax=axes[0], color=color
        )
        PrecisionRecallDisplay.from_predictions(
            y_eval, y_proba, name=name, ax=axes[1], color=color
        )

    # Reference lines
    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random Guess (0.50)")
    baseline_pr = float(np.mean(y_eval))
    axes[1].axhline(
        baseline_pr,
        color="k",
        linestyle="--",
        alpha=0.5,
        label=f"Baseline Prevalence ({baseline_pr:.2f})",
    )

    axes[0].set_title("ROC Curves Comparison", fontweight="bold")
    axes[1].set_title("Precision-Recall Curves Comparison", fontweight="bold")
    axes[0].legend(loc="lower right")
    axes[1].legend(loc="lower left")

    plt.tight_layout()
    if save_name:
        plt.savefig(FIGURES_DIR / save_name, dpi=300, bbox_inches="tight")
    plt.show()


def plot_calibration_curves(
    models_dict: Dict[str, Any],
    X_eval,
    y_eval,
    n_bins: int = 10,
    save_name: Optional[str] = None,
) -> None:
    """
    Plot reliability curves (Calibration Curves) and Brier scores for probabilistic risk models.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")

    for idx, (name, model) in enumerate(models_dict.items()):
        color = PALETTE[idx % len(PALETTE)]
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_eval)[:, 1]
        else:
            continue

        prob_true, prob_pred = calibration_curve(y_eval, y_proba, n_bins=n_bins, strategy="uniform")
        brier = brier_score_loss(y_eval, y_proba)
        ax.plot(prob_pred, prob_true, marker="o", linewidth=2, color=color,
                label=f"{name} (Brier: {brier:.4f})")

    ax.set_xlabel("Mean Predicted Probability (Risk Score)")
    ax.set_ylabel("Fraction of Positives (Actual Event Rate)")
    ax.set_title("Probability Calibration (Reliability Diagram)", fontweight="bold")
    ax.legend(loc="upper left")

    plt.tight_layout()
    if save_name:
        plt.savefig(FIGURES_DIR / save_name, dpi=300, bbox_inches="tight")
    plt.show()


def audit_subgroup_fairness(
    y_true: pd.Series,
    y_proba: np.ndarray,
    subgroup_series: pd.Series,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """
    Audit model performance across demographic/socioeconomic subgroups.
    Computes Subgroup Size, Prevalence, ROC-AUC, Recall (TPR), and False Positive Rate (FPR).
    """
    subgroup_name = subgroup_series.name or "Subgroup"
    y_pred = (y_proba >= threshold).astype(int)
    
    records = []
    for group_val, indices in subgroup_series.groupby(subgroup_series).groups.items():
        g_y_true = np.array(y_true.iloc[indices])
        g_y_pred = np.array(y_pred[indices])
        g_y_proba = np.array(y_proba[indices])

        tn, fp, fn, tp = confusion_matrix(g_y_true, g_y_pred, labels=[0, 1]).ravel()
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        auc = roc_auc_score(g_y_true, g_y_proba) if len(np.unique(g_y_true)) > 1 else np.nan

        records.append({
            subgroup_name: group_val,
            "Count": len(indices),
            "Actual Prevalence": float(np.mean(g_y_true)),
            "ROC-AUC": float(auc),
            "Recall (TPR)": float(tpr),
            "FPR": float(fpr),
            "False Negatives": int(fn),
        })

    return pd.DataFrame(records).sort_values(by="Count", ascending=False).reset_index(drop=True)

