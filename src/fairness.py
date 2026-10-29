"""
Fairness and Algorithmic Equity Auditing Module.
Evaluates model performance across demographic slices (sex, income level, age brackets)
and computes fairness metrics such as Equal Opportunity Difference and Disparate Impact.
"""
from typing import Dict, Any, List
import numpy as np
import pandas as pd
from sklearn.metrics import recall_score, roc_auc_score


def evaluate_equal_opportunity(
    y_true: pd.Series,
    y_pred: np.ndarray,
    protected_attribute: pd.Series,
) -> pd.DataFrame:
    r"""
    Evaluates the Equal Opportunity metric:
    $P(\hat{Y}=1 \mid Y=1, A=a) = P(\hat{Y}=1 \mid Y=1, A=b)$
    In a clinical setting, ensuring equal True Positive Rate (Sensitivity / Recall) across
    sex or income groups prevents under-diagnosis in vulnerable populations.
    """
    attr_name = protected_attribute.name or "Subgroup"
    records = []

    for group_val in protected_attribute.unique():
        idx = (protected_attribute == group_val).values
        g_y_true = y_true.iloc[idx].values
        g_y_pred = y_pred[idx]

        recall = recall_score(g_y_true, g_y_pred, zero_division=0)
        
        # False Positive Rate: FP / (FP + TN)
        fp = np.sum((g_y_pred == 1) & (g_y_true == 0))
        tn = np.sum((g_y_pred == 0) & (g_y_true == 0))
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

        records.append({
            attr_name: group_val,
            "Sample Size": int(np.sum(idx)),
            "Prevalence": float(np.mean(g_y_true)),
            "True Positive Rate (Recall)": float(recall),
            "False Positive Rate (FPR)": float(fpr),
        })

    df_fairness = pd.DataFrame(records).sort_values(by="Sample Size", ascending=False)
    
    # Calculate Equal Opportunity Difference (Max TPR - Min TPR)
    tpr_max = df_fairness["True Positive Rate (Recall)"].max()
    tpr_min = df_fairness["True Positive Rate (Recall)"].min()
    df_fairness["Max_TPR_Disparity"] = tpr_max - tpr_min

    return df_fairness
