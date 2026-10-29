"""
Cardiovascular Feature Engineering Module.

This module provides the framework for domain-specific feature engineering.
Cardiovascular risk modeling benefits significantly from composite physiological indices
and interaction terms that capture multi-system risk (e.g., metabolic-vascular interactions).

References & Clinical Context:
- AHA/ACC Cardiovascular Risk Guidelines (Framingham / ASCVD Risk Estimators)
- U-shaped Sleep Duration vs. CVD: Cappuccio et al., European Heart Journal
- Pulse Pressure & Arterial Stiffness: Franklin et al., Circulation
"""
from typing import Optional
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class CardiovascularFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Scikit-Learn compatible transformer to generate domain-specific cardiovascular features.
    
    Transforms a raw DataFrame of patient records into an enriched feature representation.
    """

    def __init__(
        self,
        create_interactions: bool = True,
        create_lifestyle_index: bool = True,
        create_sleep_deviation: bool = True,
        create_hemodynamic_ratios: bool = True,
    ):
        self.create_interactions = create_interactions
        self.create_lifestyle_index = create_lifestyle_index
        self.create_sleep_deviation = create_sleep_deviation
        self.create_hemodynamic_ratios = create_hemodynamic_ratios

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """Fit method (stateless transformer)."""
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Apply feature engineering transformations.
        
        Args:
            X (pd.DataFrame): Input dataframe containing the 25 predictor columns.
            
        Returns:
            pd.DataFrame: Copy of X with new engineered columns added.
        """
        df = X.copy()

        # ======================================================================
        # TODO 1: Sleep Duration Deviation (U-Shaped Risk Modeling)
        # ======================================================================
        # Clinical Context:
        # Both short sleep (<6h) and long sleep (>9h) are independently associated
        # with increased cardiovascular mortality (U-shaped curve).
        # Linear models fail to capture this non-linearity unless explicitly engineered.
        #
        # Formula:
        #   sleep_deviation = abs(sleep_hours - optimal_sleep)
        #   where optimal_sleep is typically ~7.5 hours.
        #
        # Reference: https://doi.org/10.1093/eurheartj/ehq419
        optimal_sleep = 7.5
        if self.create_sleep_deviation and "sleep_hours" in df.columns:
            df["sleep_deviation"] = (df["sleep_hours"] - optimal_sleep).abs()

        # ======================================================================
        # TODO 2: Hemodynamic & Heart Rate Ratios
        # ======================================================================
        # Clinical Context:
        # The relationship between maximum heart rate during stress and average
        # resting heart rate reflects chronotropic competence and autonomic balance.
        # HRV (Heart Rate Variability) reflects parasympathetic tone.
        #
        # Proposed features:
        #   1. hr_reserve_ratio = max_heart_rate / (avg_resting_hr + 1e-5)
        #   2. hrv_to_resting_hr = hrv_score / (avg_resting_hr + 1e-5)
        # ----------------------------------------------------------------------
        if self.create_hemodynamic_ratios:
            if "max_heart_rate" in df.columns and "avg_resting_hr" in df.columns:
                df["hr_reserve_ratio"] = df["max_heart_rate"] / (df["avg_resting_hr"] + 1e-5)
            if "hrv_score" in df.columns and "avg_resting_hr" in df.columns:
                df["hrv_to_resting_hr"] = df["hrv_score"] / (df["avg_resting_hr"] + 1e-5)

        # ======================================================================
        # TODO 3: Physiological Interaction Terms
        # ======================================================================
        # Clinical Context:
        # Risk factors rarely act in isolation. Hypertension compounded by age
        # causes arterial remodeling; Diabetes combined with elevated BMI
        # accelerates metabolic syndrome (synergistic risk).
        #
        # Proposed interactions:
        #   1. age_x_hypertension = age * hypertension
        #   2. bmi_x_diabetes = bmi * diabetes
        # ----------------------------------------------------------------------
        if self.create_interactions:
            if "age" in df.columns and "hypertension" in df.columns:
                df["age_x_hypertension"] = df["age"] * df["hypertension"]
            if "bmi" in df.columns and "diabetes" in df.columns:
                df["bmi_x_diabetes"] = df["bmi"] * df["diabetes"]

        # ======================================================================
        # TODO 4: Composite Lifestyle Score (Modifiable Factor Aggregate)
        # ======================================================================
        # Clinical Context:
        # One of the main research questions is contrasting clinical vs lifestyle factors.
        # A composite Lifestyle Index aggregates modifiable protective vs adverse behaviors:
        # - Higher diet_quality (+ protective)
        # - Higher physical_activity (+ protective)
        # - Smoking status (- adverse)
        # - Heavy alcohol intake (- adverse)
        #
        # Notice: You will need to map categorical levels to numerical scores first
        # (e.g. Sedentary=0, Light=1, Active=2, Very Active=3; Never=2, Former=1, Current=0).
        # ----------------------------------------------------------------------
        if self.create_lifestyle_index:
            activity_map = {"Sedentary": 0.0, "Light": 1.0, "Active": 2.0, "Very Active": 3.0}
            smoking_map = {"Current": 0.0, "Former": 1.0, "Never": 2.0}
            alcohol_map = {"Heavy": 0.0, "Moderate": 1.0, "Non-drinker": 2.0}

            act_score = df["physical_activity"].map(activity_map).fillna(1.0) if "physical_activity" in df.columns else 1.0
            smk_score = df["smoking_status"].map(smoking_map).fillna(1.0) if "smoking_status" in df.columns else 1.0
            alc_score = df["alcohol_intake"].map(alcohol_map).fillna(1.0) if "alcohol_intake" in df.columns else 1.0
            diet_score = df["diet_quality"] if "diet_quality" in df.columns else 5.0

            # Composite index normalized to [0, 10]:
            # Diet (0-10 -> 0-3.5), Activity (0-3 -> 0-2.5), Smoking (0-2 -> 0-2.0), Alcohol (0-2 -> 0-2.0)
            df["composite_lifestyle_score"] = (
                (diet_score / 10.0) * 3.5
                + (act_score / 3.0) * 2.5
                + (smk_score / 2.0) * 2.0
                + (alc_score / 2.0) * 2.0
            )

        return df


def verify_feature_engineering():
    """Simple verification test to test the student's implementation."""
    from src.data_loader import load_processed_splits
    train_df, _, _ = load_processed_splits()
    
    engineer = CardiovascularFeatureEngineer()
    enriched = engineer.transform(train_df)
    
    print(f"Original shape: {train_df.shape}")
    print(f"Enriched shape: {enriched.shape}")
    print(f"New columns created: {[c for c in enriched.columns if c not in train_df.columns]}")
    return enriched


if __name__ == "__main__":
    verify_feature_engineering()

