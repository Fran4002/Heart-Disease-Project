"""
Utility script to generate the 4 guided educational Jupyter notebooks.
Creates clean, structured, reproducible notebooks with scaffolding,
theory, instructions, hints, and TODO cells.
"""
import json
from pathlib import Path

NOTEBOOKS_DIR = Path(__file__).resolve().parent.parent / "notebooks"
NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)

KERNEL_META = {
    "kernelspec": {
        "display_name": "Python (ML Heart Disease)",
        "language": "python",
        "name": "ml-heart-disease",
    },
    "language_info": {
        "name": "python",
        "version": "3.12",
    },
}


def cell(cell_type: str, content: str):
    return {
        "cell_type": cell_type,
        "metadata": {},
        "source": [line + "\n" for line in content.strip().split("\n")],
        **({"execution_count": None, "outputs": []} if cell_type == "code" else {}),
    }


def save_notebook(cells, filename):
    nb = {
        "cells": cells,
        "metadata": KERNEL_META,
        "nbformat": 4,
        "nbformat_minor": 4,
    }
    path = NOTEBOOKS_DIR / filename
    with open(path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2, ensure_ascii=False)
    print(f"Created {path.name}")


# ==============================================================================
# NOTEBOOK 01: EDA & Data Profiling
# ==============================================================================
def build_nb01():
    cells = [
        cell("markdown", """# Fase 1 & 2: Análisis Exploratorio de Datos (EDA) y Perfilado Clínico

> **Proyecto:** Modelado predictivo y análisis multifactorial de riesgo de enfermedades cardíacas  
> **Institución:** Universidad Autónoma de San Luis Potosí (UASLP) — Facultad de Ingeniería  
> **Dataset:** 100,000 registros sintéticos calibrados a parámetros poblacionales (CDC / AHA / NHANES)  
> **Objetivo:** Comprender a fondo las distribuciones, calidad de datos y correlaciones diferenciales entre factores clínicos, demográficos, de estilo de vida y señales de wearables.

---

## Estructura de este Notebook Guiado
1. **Configuración y Carga de Datos** *(Pre-configurado: Trabajo de fontanería listo)*
2. **Inspección de Integridad y Calidad**
3. **Análisis de la Variable Objetivo (`heart_disease`)**
4. **Perfilado Demográfico y Socioeconómico**
5. **Marcadores Clínicos y Diagnósticos**
6. **Factores Modificables de Estilo de Vida**
7. **Biomarcadores de Dispositivos Vestibles (`hrv_score`, `avg_resting_hr`)**
8. **Ejercicios Guiados para el Alumno (TODOs)**"""),

        cell("code", """import sys
from pathlib import Path

ROOT_DIR = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.config import (
    RAW_DATA_PATH,
    TARGET_COL,
    DEMOGRAPHIC_COLS,
    LIFESTYLE_COLS,
    MEDICAL_HISTORY_COLS,
    CLINICAL_MARKER_COLS,
    WEARABLE_COLS,
    PALETTE,
    POS_COLOR,
    NEG_COLOR,
    PLOT_STYLE,
)
from src.data_loader import load_processed_splits

plt.rcParams.update(PLOT_STYLE)
print("Entorno inicializado correctamente con matplotlib, seaborn y src.config!")"""),

        cell("markdown", """## 1. Carga del Dataset y Verificación de Integridad
Utilizamos la utilidad `load_processed_splits()` de `src.data_loader` para trabajar con las particiones estratificadas ya generadas sin fuga de datos."""),

        cell("code", """train_df, val_df, test_df = load_processed_splits()
print(f"Conjunto de Entrenamiento: {train_df.shape}")
print(f"Conjunto de Validación:    {val_df.shape}")
print(f"Conjunto de Prueba:        {test_df.shape}")
train_df.head()"""),

        cell("markdown", """## 2. Visión General de Tipos y Distribución del Target
La prevalencia de la variable objetivo `heart_disease` es de **46.1%**, lo que proporciona un balance natural ideal sin necesidad de sobremuestreo artificial."""),

        cell("code", """fig, axes = plt.subplots(1, 2, figsize=(12, 4))

target_counts = train_df[TARGET_COL].value_counts()
labels = ['Sin Enfermedad (0)', 'Enfermedad Cardíaca (1)']
bars = axes[0].bar(labels, target_counts.values, color=[NEG_COLOR, POS_COLOR], width=0.5)

for bar, val in zip(bars, target_counts.values):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 500,
                 f"{val:,} ({val/len(train_df):.1%})", ha='center', va='bottom', fontweight='bold')
axes[0].set_title('Distribución de Clases en Train', fontweight='bold')
axes[0].set_ylabel('Pacientes')

summary = train_df[CLINICAL_MARKER_COLS[:4] + WEARABLE_COLS].describe().T[['mean', 'std', 'min', '50%', 'max']]
axes[1].axis('off')
table = axes[1].table(cellText=summary.round(2).values, rowLabels=summary.index,
                      colLabels=['Media', 'Desv. Est.', 'Mín', 'Mediana', 'Máx'], loc='center')
table.scale(1, 1.6)
axes[1].set_title('Resumen Estadístico Seleccionado', fontweight='bold')

plt.tight_layout()
plt.show()"""),

        cell("markdown", """---
## 3. Ejercicios Guiados para el Alumno (TODOs)

A continuación, implementa los análisis clínicos y de estilo de vida esenciales para responder la pregunta de investigación:
> *¿Cuál es el factor de riesgo de sufrir una enfermedad cardíaca, evaluando el impacto diferenciado entre factores clínicos y factores de estilo de vida?*"""),

        cell("markdown", """### 🛠️ Ejercicio 1.1: Cálculo y Visualización de Riesgo Relativo / Tasa de Prevalencia
Calcula la tasa de prevalencia cardíaca (`mean()`) condicionada a cada categoría de los factores de riesgo clave:
- `hypertension` (0 vs 1)
- `diabetes` (0 vs 1)
- `smoking_status` (Never, Former, Current)
- `exercise_angina` (0 vs 1)

**Objetivo:** Determinar qué factor incrementa en mayor proporción la probabilidad empírica de evento cardíaco."""),

        cell("code", """# TODO: Calcula la tasa de prevalencia de heart_disease para cada grupo.
# 1. Hipertensión:
prev_hyp = train_df.groupby('hypertension')['heart_disease'].agg(['count', 'mean']).rename(columns={'count': 'Pacientes', 'mean': 'Prevalencia'})
print("=== Prevalencia por Hipertensión ===")
display(prev_hyp)

# 2. Diabetes:
prev_diab = train_df.groupby('diabetes')['heart_disease'].agg(['count', 'mean']).rename(columns={'count': 'Pacientes', 'mean': 'Prevalencia'})
print("=== Prevalencia por Diabetes ===")
display(prev_diab)

# 3. Tabaquismo (smoking_status):
prev_smoke = train_df.groupby('smoking_status')['heart_disease'].agg(['count', 'mean']).rename(columns={'count': 'Pacientes', 'mean': 'Prevalencia'})
print("=== Prevalencia por Tabaquismo ===")
display(prev_smoke)

# 4. Angina inducida por ejercicio (exercise_angina):
prev_angina = train_df.groupby('exercise_angina')['heart_disease'].agg(['count', 'mean']).rename(columns={'count': 'Pacientes', 'mean': 'Prevalencia'})
print("=== Prevalencia por Angina de Ejercicio ===")
display(prev_angina)"""),

        cell("markdown", """### 🛠️ Ejercicio 1.2: Análisis de la No-Linealidad del Sueño (U-Shaped Curve)
La literatura médica sugiere que tanto la privación de sueño (<6 horas) como el sueño excesivo (>9 horas) se asocian a un mayor riesgo cardiovascular.

**Tu tarea:**
1. Agrupa la columna `sleep_hours` en rangos o redondea a horas enteras con `train_df['sleep_hours'].round()`.
2. Calcula la tasa promedio de `heart_disease` por cada intervalo.
3. Grafica la curva resultante (línea o barras) y confirma visualmente si se observa el patrón en forma de 'U'."""),

        cell("code", """# TODO: Implementa la agrupación y gráfica de sleep_hours vs heart_disease
sleep_rounded = train_df['sleep_hours'].round()
sleep_grouped = train_df.groupby(sleep_rounded)['heart_disease'].agg(['count', 'mean']).reset_index()

plt.figure(figsize=(9, 4.5))
plt.plot(sleep_grouped['sleep_hours'], sleep_grouped['mean'] * 100, marker='o', linewidth=2.5, color='#eb6834')
plt.axvline(7.5, color='#1baf7a', linestyle='--', label='Duración Óptima (~7.5 hrs)')
plt.title('Tasa de Enfermedad Cardíaca vs Horas de Sueño (Curva en U)', fontweight='bold')
plt.xlabel('Horas de Sueño (redondeadas)')
plt.ylabel('Prevalencia de Enfermedad Cardíaca (%)')
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.show()"""),

        cell("markdown", """### 🛠️ Ejercicio 1.3: Análisis de Wearables (HRV Score y Frecuencia Cardíaca)
El biomarcador `hrv_score` (Variabilidad de la Frecuencia Cardíaca) refleja la actividad parasimpática y la salud del sistema nervioso autónomo.
Una menor HRV se asocia clínicamente con fatiga cardíaca y mayor mortalidad.

**Tu tarea:**
1. Grafica la distribución de `hrv_score` y `avg_resting_hr` comparando pacientes sanos (`heart_disease=0`) vs con enfermedad (`heart_disease=1`).
2. Compara la media y desviación estándar de ambos grupos."""),

        cell("code", """# TODO: Genera histogramas superpuestos o diagramas boxplot para hrv_score y avg_resting_hr
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# HRV Score por condición cardíaca
sns.kdeplot(data=train_df[train_df['heart_disease'] == 0]['hrv_score'], ax=axes[0], fill=True, label='Sano (0)', color='#2a78d6')
sns.kdeplot(data=train_df[train_df['heart_disease'] == 1]['hrv_score'], ax=axes[0], fill=True, label='Enfermedad (1)', color='#e34948')
axes[0].set_title('Distribución de HRV Score según Condición Cardíaca', fontweight='bold')
axes[0].set_xlabel('HRV Score (ms)')
axes[0].legend()

# Frecuencia Cardíaca en Reposo
sns.boxplot(x='heart_disease', y='avg_resting_hr', data=train_df, ax=axes[1], palette=['#2a78d6', '#e34948'])
axes[1].set_title('Frecuencia Cardíaca en Reposo por Grupo', fontweight='bold')
axes[1].set_xticklabels(['Sano (0)', 'Enfermedad (1)'])
axes[1].set_ylabel('Frecuencia Cardíaca Media en Reposo (lpm)')

plt.tight_layout()
plt.show()

# Estadísticos comparativos
stats_wearables = train_df.groupby('heart_disease')[['hrv_score', 'avg_resting_hr']].agg(['mean', 'std']).T
display(stats_wearables)"""),

        cell("markdown", """### 🛠️ Ejercicio 1.4: Síntesis de Correlaciones (Clínicos vs Estilo de Vida)
Calcula las correlaciones de Pearson y Spearman de las variables con respecto a `heart_disease`.

**Preguntas de reflexión para tu reporte:**
1. ¿Cuáles son las 3 variables con mayor correlación con la enfermedad? ¿Pertenecen al grupo clínico o al grupo de estilo de vida?
2. ¿Existe alguna variable de estilo de vida con correlación comparable a los marcadores clínicos?"""),

        cell("code", """# TODO: Codifica o selecciona variables numéricas y calcula la correlación con TARGET_COL.
corr_df = train_df.copy()
for col in corr_df.select_dtypes(include=['object', 'category']).columns:
    corr_df[col] = pd.factorize(corr_df[col])[0]

correlations = corr_df.corr(method='pearson')['heart_disease'].drop('heart_disease').sort_values(ascending=True)

plt.figure(figsize=(10, 8))
colors = ['#e34948' if c > 0.20 else '#2a78d6' if c < -0.20 else '#7f8c8d' for c in correlations.values]
correlations.plot(kind='barh', color=colors)
plt.title('Correlación de Pearson con Enfermedad Cardíaca (TARGET)', fontweight='bold')
plt.xlabel('Coeficiente de Correlación (r)')
plt.axvline(0, color='black', linewidth=0.8, linestyle='--')
plt.grid(True, linestyle=':', alpha=0.5)
plt.tight_layout()
plt.show()

print("Top 5 factores de mayor correlación absoluta:")
display(correlations.abs().sort_values(ascending=False).head(5))"""),
    ]
    save_notebook(cells, "01_eda_and_data_profiling.ipynb")


# ==============================================================================
# NOTEBOOK 02: Feature Engineering & Preprocessing
# ==============================================================================
def build_nb02():
    cells = [
        cell("markdown", """# Fase 3: Preprocesamiento y Feature Engineering Clínico

> **Proyecto:** Modelado predictivo y análisis multifactorial de riesgo de enfermedades cardíacas  
> **Objetivo:** Construir canalizaciones robustas de transformación (`ColumnTransformer` y `Pipeline`), generar nuevas variables fisiológicas/interactivas sin fuga de datos, y preparar las matrices de características para los modelos.

---

## Contenido de este Notebook Guiado
1. **Prevención de Fuga de Datos (*Data Leakage*)**: Por qué ajustar transformadores exclusivamente en Train.
2. **Ingeniería de Características Cardiovasculares** (`src/features.py`).
3. **Canalizaciones de Preprocesamiento** (`src/preprocessing.py`):
   - One-Hot Encoding para nominales (`region`, `sex`, etc.) con `drop='first'`.
   - Ordinal Encoding con orden explícito para `income_level` y `physical_activity`.
   - Escalado estándar para modelos lineales (`StandardScaler`).
4. **Ejercicios Guiados para el Alumno (TODOs)**"""),

        cell("code", """import sys
from pathlib import Path

ROOT_DIR = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.config import (
    TARGET_COL,
    NOMINAL_COLS,
    ORDINAL_COLS,
    ORDINAL_CATEGORIES,
    BINARY_NUMERIC_COLS,
    CONTINUOUS_NUMERIC_COLS,
)
from src.data_loader import load_processed_splits
from src.features import CardiovascularFeatureEngineer
from src.preprocessing import build_linear_preprocessor, build_tree_preprocessor

train_df, val_df, test_df = load_processed_splits()
X_train = train_df.drop(columns=[TARGET_COL])
y_train = train_df[TARGET_COL]

X_val = val_df.drop(columns=[TARGET_COL])
y_val = val_df[TARGET_COL]

X_test = test_df.drop(columns=[TARGET_COL])
y_test = test_df[TARGET_COL]

print(f"X_train: {X_train.shape}, X_val: {X_val.shape}, X_test: {X_test.shape}")"""),

        cell("markdown", r"""## 1. Implementación de Feature Engineering en `src/features.py`
En `src/features.py` se estructuró la clase `CardiovascularFeatureEngineer`.
Tu tarea en esta sección es completar las fórmulas clave descritas en los comentarios del módulo:

1. **`sleep_deviation`**: Distancia absoluta respecto al sueño óptimo (7.5 horas).
   $$ \text{sleep\_deviation} = |\text{sleep\_hours} - 7.5| $$
2. **`hr_reserve_ratio`**: Relación entre frecuencia cardíaca máxima y frecuencia cardíaca basal.
   $$ \text{hr\_reserve\_ratio} = \frac{\text{max\_heart\_rate}}{\text{avg\_resting\_hr} + 10^{-5}} $$
3. **`age_x_hypertension`**: Interacción multiplicativa entre edad e hipertensión.
   $$ \text{age\_x\_hypertension} = \text{age} \times \text{hypertension} $$
4. **`bmi_x_diabetes`**: Interacción multiplicativa metabólica.
   $$ \text{bmi\_x\_diabetes} = \text{bmi} \times \text{diabetes} $$"""),

        cell("markdown", """### 🛠️ Ejercicio 2.1: Probar y Verificar tu Feature Engineering
Una vez implementadas las fórmulas en [src/features.py](file:///home/fran/Documents/ML/src/features.py), ejecuta la siguiente celda para validar que se generen las columnas adicionales en `X_train`:"""),

        cell("code", """# TODO: Instancia CardiovascularFeatureEngineer y transforma X_train
feature_engineer = CardiovascularFeatureEngineer()
X_train_enriched = feature_engineer.transform(X_train)

print(f"Dimensiones originales: {X_train.shape}")
print(f"Dimensiones enriquecidas: {X_train_enriched.shape}")
new_cols = [c for c in X_train_enriched.columns if c not in X_train.columns]
print(f"Nuevas variables generadas ({len(new_cols)}): {new_cols}")
display(X_train_enriched[new_cols].head())"""),

        cell("markdown", """## 2. Inspección del `ColumnTransformer`
El preprocesador para modelos lineales debe transformar las variables heterogéneas en una matriz numérica continua y acotada."""),

        cell("code", """linear_preprocessor = build_linear_preprocessor()

# Ajustar exclusivamente en Train para evitar Data Leakage
linear_preprocessor.fit(X_train)

X_train_proc = linear_preprocessor.transform(X_train)
X_val_proc = linear_preprocessor.transform(X_val)

feature_names = linear_preprocessor.get_feature_names_out()
print(f"Dimensión resultante para modelos lineales: {X_train_proc.shape}")
print(f"Total de características resultantes tras codificación: {len(feature_names)}")"""),

        cell("markdown", """### 🛠️ Ejercicio 2.2: Construir un Pipeline Completo (Feature Engineering + Preprocesamiento)
En Scikit-Learn, podemos encadenar `CardiovascularFeatureEngineer` y `ColumnTransformer` en un único `Pipeline`.

**Tu tarea:**
Crea un `Pipeline` que primero ejecute tu ingeniería de características y luego aplique el preprocesamiento correspondiente."""),

        cell("code", """# TODO: Importa Pipeline de sklearn.pipeline y combina ambos pasos:
from sklearn.pipeline import Pipeline

full_preprocessing_pipeline = Pipeline([
    ('features', CardiovascularFeatureEngineer()),
    ('preprocessor', build_linear_preprocessor())
])

X_train_full = full_preprocessing_pipeline.fit_transform(X_train)
X_val_full = full_preprocessing_pipeline.transform(X_val)

print(f"Matriz de Train transformada: {X_train_full.shape}")
print(f"Matriz de Val transformada: {X_val_full.shape}")
print(f"Número de características finales: {X_train_full.shape[1]}")"""),
    ]
    save_notebook(cells, "02_feature_engineering_and_preprocessing.ipynb")


# ==============================================================================
# NOTEBOOK 03: Model Training, Tuning & Probability Calibration
# ==============================================================================
def build_nb03():
    cells = [
        cell("markdown", """# Fase 4 & 5: Entrenamiento, Optimización y Calibración de Probabilidades

> **Proyecto:** Modelado predictivo y análisis multifactorial de riesgo de enfermedades cardíacas  
> **Objetivo:** Entrenar y comparar múltiples clasificadores (Regresión Logística, Random Forest, LightGBM / XGBoost), optimizar hiperparámetros, calibrar las probabilidades de salida (`heart_disease_prob`) y ajustar el umbral de decisión médico para priorizar la Sensibilidad (*Recall*).

---

## Contenido de este Notebook Guiado
1. **Modelos Base**: Regresión Logística y Random Forest.
2. **Modelos de Gradiente Avanzados**: LightGBM / XGBoost.
3. **Comparación Global de Curvas ROC y PR**.
4. **Calibración Probabilística de Riesgo (`CalibratedClassifierCV`) & Brier Score**.
5. **Optimización del Umbral de Decisión Clínico (*Medical Threshold Tuning*)**.
6. **Ejercicios Guiados para el Alumno (TODOs)**"""),

        cell("code", """import sys
from pathlib import Path

ROOT_DIR = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.config import TARGET_COL, RANDOM_STATE
from src.data_loader import load_processed_splits
from src.models import (
    build_logistic_baseline_pipeline,
    build_random_forest_pipeline,
    build_lightgbm_pipeline,
    calibrate_model,
)
from src.evaluation import (
    compute_metrics,
    metrics_to_dataframe,
    plot_roc_pr_comparison,
    plot_calibration_curves,
)

train_df, val_df, test_df = load_processed_splits()
X_train, y_train = train_df.drop(columns=[TARGET_COL]), train_df[TARGET_COL]
X_val, y_val = val_df.drop(columns=[TARGET_COL]), val_df[TARGET_COL]
X_test, y_test = test_df.drop(columns=[TARGET_COL]), test_df[TARGET_COL]

print(f"Particiones listas: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")"""),

        cell("markdown", """## 1. Entrenamiento de Modelos Base
Comenzamos entrenando el modelo base de Regresión Logística (con codificación correcta y escalado estándar) y el ensamble de Random Forest."""),

        cell("code", """%%time
# 1. Regresión Logística Baseline
lr_model = build_logistic_baseline_pipeline(C=1.0)
lr_model.fit(X_train, y_train)

# 2. Random Forest Baseline
rf_model = build_random_forest_pipeline(n_estimators=150, max_depth=12)
rf_model.fit(X_train, y_train)

print("Modelos base entrenados exitosamente!")"""),

        cell("markdown", """## 2. Entrenamiento de Modelo de Gradiente (LightGBM)
LightGBM suele obtener el mejor compromiso entre velocidad y discriminación en datos tabulares de 100,000 registros."""),

        cell("code", """%%time
lgb_model = build_lightgbm_pipeline(n_estimators=300, learning_rate=0.05, max_depth=6)
lgb_model.fit(X_train, y_train)
print("LightGBM entrenado exitosamente!")"""),

        cell("markdown", """## 3. Comparación de Métricas en el Conjunto de Validación
Evaluamos el desempeño discriminativo sin tocar el conjunto de prueba (Test)."""),

        cell("code", """models = {
    "Logistic Regression": lr_model,
    "Random Forest": rf_model,
    "LightGBM": lgb_model,
}

val_metrics = {}
for name, model in models.items():
    preds = model.predict(X_val)
    probas = model.predict_proba(X_val)[:, 1]
    val_metrics[name] = compute_metrics(y_val, preds, probas)

df_results = metrics_to_dataframe(val_metrics)
display(df_results.round(4))

plot_roc_pr_comparison(models, X_val, y_val)"""),

        cell("markdown", """---
## 4. Ejercicios Guiados para el Alumno (TODOs)"""),

        cell("markdown", r"""### 🛠️ Ejercicio 3.1: Calibración de Probabilidades con `CalibratedClassifierCV`
En aplicaciones de salud pública y clínicas, no basta con predecir 0 o 1: se requiere una probabilidad empírica confiable (estimador de riesgo continuo $\hat{P}(Y=1 \mid X)$).

> **Nota sobre Scikit-Learn moderno:** En versiones recientes de scikit-learn, el parámetro `cv='prefit'` fue reemplazado por `FrozenEstimator` (`from sklearn.frozen import FrozenEstimator`). La función auxiliar `calibrate_model()` gestiona automáticamente esta compatibilidad interna.

**Tu tarea:**
1. Aplica `calibrate_model()` (o directamente `CalibratedClassifierCV` con `FrozenEstimator`) al mejor modelo (ej. `lgb_model`).
2. Ajusta el calibrador utilizando el conjunto de validación (`X_val, y_val`).
3. Evalúa la reducción en el **Brier Score** (donde un valor menor indica mejor calibración).
4. Utiliza `plot_calibration_curves()` para comparar la curva de confiabilidad antes y después de calibrar."""),

        cell("code", """# TODO: Calibra el modelo seleccionado y compara Brier Score y curvas de confiabilidad
# Opción 1: Utilizando el helper calibrate_model (compatible con FrozenEstimator internamente)
cal_lgb = calibrate_model(lgb_model, method='sigmoid')
cal_lgb.fit(X_val, y_val)

# Opción 2 equivalente directa en scikit-learn moderno:
# from sklearn.frozen import FrozenEstimator
# from sklearn.calibration import CalibratedClassifierCV
# cal_lgb = CalibratedClassifierCV(estimator=FrozenEstimator(lgb_model), method='sigmoid')
# cal_lgb.fit(X_val, y_val)

# Probabilidades estimadas en validación
val_proba_uncal = lgb_model.predict_proba(X_val)[:, 1]
val_proba_cal = cal_lgb.predict_proba(X_val)[:, 1]

brier_uncal = compute_metrics(y_val, (val_proba_uncal >= 0.5).astype(int), val_proba_uncal)["Brier Score"]
brier_cal = compute_metrics(y_val, (val_proba_cal >= 0.5).astype(int), val_proba_cal)["Brier Score"]

print(f"Brier Score Sin Calibrar: {brier_uncal:.4f}")
print(f"Brier Score Calibrado (Platt Scaling): {brier_cal:.4f}")
print(f"Mejora relativa en calibración: {((brier_uncal - brier_cal) / brier_uncal) * 100:.2f}%")

plot_calibration_curves(
    {'LightGBM Sin Calibrar': lgb_model, 'LightGBM Calibrado (Platt)': cal_lgb},
    X_val,
    y_val,
    save_name="calibracion_lightgbm.png"
)"""),

        cell("markdown", r"""### 🛠️ Ejercicio 3.2: Ajuste de Umbral de Decisión Clínico (*Medical Threshold Tuning*)
Por defecto, Scikit-Learn utiliza un umbral de corte de $\tau = 0.50$.  
En el diagnóstico médico de cardiopatías, **un Falso Negativo (paciente enfermo clasificado como sano) es mucho más grave que un Falso Positivo**.

**Tu tarea:**
1. Itera a través de una malla de umbrales $\tau \in [0.10, 0.90]$.
2. Calcula la Sensibilidad (*Recall*) y la Precisión para cada umbral.
3. Encuentra el umbral óptimo que garantiza una **Sensibilidad $\ge 85\%$ o $90\%$**, minimizando falsos negativos.
4. Grafica la curva de Recall vs Precision en función del umbral."""),

        cell("code", """# TODO: Implementa la búsqueda del umbral óptimo
from sklearn.metrics import recall_score, precision_score, f1_score

thresholds = np.linspace(0.1, 0.9, 81)
y_proba_val = lgb_model.predict_proba(X_val)[:, 1]

recalls = []
precisions = []
f1_scores = []

for t in thresholds:
    y_pred_t = (y_proba_val >= t).astype(int)
    recalls.append(recall_score(y_val, y_pred_t, zero_division=0))
    precisions.append(precision_score(y_val, y_pred_t, zero_division=0))
    f1_scores.append(f1_score(y_val, y_pred_t, zero_division=0))

valid_idx = [i for i, r in enumerate(recalls) if r >= 0.90]
optimal_idx = valid_idx[-1] if valid_idx else 0
optimal_thresh = thresholds[optimal_idx]

print(f"Umbral clínico óptimo (Recall >= 90%): tau = {optimal_thresh:.2f}")
print(f"Sensibilidad alcanzada: {recalls[optimal_idx]:.4f}")
print(f"Precisión correspondiente: {precisions[optimal_idx]:.4f}")
print(f"F1-Score correspondiente: {f1_scores[optimal_idx]:.4f}")

plt.figure(figsize=(10, 5))
plt.plot(thresholds, recalls, label='Sensibilidad (Recall)', color='#e34948', linewidth=2.5)
plt.plot(thresholds, precisions, label='Precisión', color='#2a78d6', linewidth=2.5)
plt.plot(thresholds, f1_scores, label='F1-Score', color='#1baf7a', linestyle='--', linewidth=2)
plt.axvline(optimal_thresh, color='#9b59b6', linestyle=':', linewidth=2, label=f'Umbral Clínico ({optimal_thresh:.2f})')
plt.axvline(0.50, color='gray', linestyle='--', linewidth=1, label='Umbral Estándar (0.50)')
plt.title('Ajuste de Umbral de Decisión Clínico (Medical Threshold Tuning)', fontweight='bold')
plt.xlabel('Umbral de Decisión (tau)', fontweight='bold')
plt.ylabel('Puntuación de Métrica', fontweight='bold')
plt.legend(frameon=True)
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
plt.show()"""),
    ]
    save_notebook(cells, "03_model_training_tuning_calibration.ipynb")


# ==============================================================================
# NOTEBOOK 04: Subgroups Fairness & SHAP Analysis
# ==============================================================================
def build_nb04():
    cells = [
        cell("markdown", """# Fase 6 & 7: Explicabilidad con SHAP y Auditoría de Equidad (Fairness)

> **Proyecto:** Modelado predictivo y análisis multifactorial de riesgo de enfermedades cardíacas  
> **Objetivo:** Responder cuantitativamente a la pregunta central de investigación mediante SHAP (*impacto clínico vs. estilo de vida*), y auditar la equidad de desempeño algorítmico en diversos subgrupos demográficos y socioeconómicos.

---

## Contenido de este Notebook Guiado
1. **Evaluación Final en el Conjunto de Prueba Aislado (Test Set)**.
2. **Explicabilidad Global y Local con SHAP** (`shap.TreeExplainer`).
3. **Desglose de Importancia: Factores Clínicos vs. Estilo de Vida**.
4. **Auditoría de Subgrupos y Equidad (*Fairness Audit*)**:
   - Desempeño por Sexo biológico (`Male` vs. `Female`).
   - Desempeño por Estrato Socioeconómico (`income_level`: Low, Middle, High).
   - Desempeño por Rango de Edad (<45, 45-60, >60 años).
5. **Ejercicios Guiados para el Alumno (TODOs)**"""),

        cell("code", """import sys
from pathlib import Path

ROOT_DIR = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

from src.config import TARGET_COL, FIGURES_DIR
from src.data_loader import load_processed_splits
from src.models import build_lightgbm_pipeline
from src.evaluation import compute_metrics, audit_subgroup_fairness
from src.explainability import compute_tree_shap_values, aggregate_group_importance
from src.fairness import evaluate_equal_opportunity

train_df, val_df, test_df = load_processed_splits()
X_train, y_train = train_df.drop(columns=[TARGET_COL]), train_df[TARGET_COL]
X_test, y_test = test_df.drop(columns=[TARGET_COL]), test_df[TARGET_COL]

# Ajustar modelo en Train+Val (85k) o directamente en Train para evaluación final
print("Entrenando modelo final LightGBM...")
final_model = build_lightgbm_pipeline(n_estimators=300, learning_rate=0.05)
final_model.fit(X_train, y_train)

test_probas = final_model.predict_proba(X_test)[:, 1]
test_preds = (test_probas >= 0.50).astype(int)

test_metrics = compute_metrics(y_test, test_preds, test_probas)
print("=== Métricas en Test (15,000 registros independientes) ===")
for k, v in test_metrics.items():
    if isinstance(v, float):
        print(f"{k:25s}: {v:.4f}")
    else:
        print(f"{k:25s}: {v}")"""),

        cell("markdown", """## 1. Explicabilidad con SHAP (*TreeExplainer*)
Calculamos los valores SHAP sobre una muestra representativa del conjunto de prueba (ej. 2,000 registros para alta velocidad y precisión numérica)."""),

        cell("code", """sample_test = X_test.sample(2000, random_state=42)
shap_values = compute_tree_shap_values(final_model, sample_test)

# Beeswarm Summary Plot
plt.figure(figsize=(10, 8))
shap.plots.beeswarm(shap_values, max_display=15, show=False)
plt.title("Impacto Global de Características (SHAP Beeswarm)", fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "shap_beeswarm.png", dpi=300, bbox_inches='tight')
plt.show()"""),

        cell("markdown", """---
## 2. Ejercicios Guiados para el Alumno (TODOs)"""),

        cell("markdown", """### 🛠️ Ejercicio 4.1: Responder a la Pregunta Central de Investigación
> *¿Cuál es el factor de riesgo de sufrir una enfermedad cardíaca, evaluando el impacto diferenciado entre factores clínicos y factores de estilo de vida?*

**Tu tarea:**
1. Utiliza la función `aggregate_group_importance(shap_values)` de `src.explainability`.
2. Genera un gráfico de pastel (*pie chart*) o de barras mostrando el **porcentaje de contribución relativa**:
   - Factores Clínicos (Marcadores Clínicos + Historial Médico)
   - Estilo de Vida (Dieta, Ejercicio, Tabaquismo, Alcohol, Sueño, Estrés)
   - Dispositivos Vestibles (HRV, Frecuencia Cardíaca en Reposo)
   - Demografía (Edad, Sexo, Región, Ingresos)
3. Escribe en una celda de Markdown tu conclusión técnica formal para el reporte de la UASLP."""),

        cell("code", """# TODO: Ejecuta aggregate_group_importance y grafica el porcentaje de impacto
group_summary = aggregate_group_importance(shap_values)
print("=== Resumen Agrupado de Importancia SHAP ===")
display(group_summary)

plt.figure(figsize=(8, 6))
plt.pie(
    group_summary['Percentage_Contribution'],
    labels=group_summary['Group'],
    autopct='%1.1f%%',
    startangle=140,
    colors=['#2a78d6', '#eb6834', '#1baf7a', '#eda100'],
    explode=[0.05, 0.05, 0.05, 0.05]
)
plt.title('Contribución Relativa de Factores en el Riesgo Cardíaco', fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "shap_group_importance.png", dpi=300, bbox_inches='tight')
plt.show()"""),

        cell("markdown", """### 🛠️ Ejercicio 4.2: Auditoría de Equidad y Subgrupos (*Fairness Audit*)
Evalúa si el modelo presenta sesgos o disparidad en el diagnóstico médico según:
1. **Sexo biológico**: `test_df['sex']` (Male vs Female).
2. **Nivel de Ingresos**: `test_df['income_level']` (Low vs Middle vs High).
3. **Grupo de Edad**: Crea rangos etarios (<45, 45-60, >60 años) a partir de `test_df['age']`.

**Preguntas clave para el reporte:**
- ¿Existe una brecha significativa en el *Recall* (Sensibilidad) entre hombres y mujeres?
- ¿El modelo subestima o sobreestima el riesgo en pacientes de bajos ingresos (*Low income*)?
- Utiliza `audit_subgroup_fairness` o `evaluate_equal_opportunity` para cuantificar estas métricas."""),

        cell("code", """# TODO: Realiza la auditoría de subgrupos por Sexo, Ingresos y Edad
print("=== 1. Auditoría de Equidad por Sexo Biológico ===")
audit_sex = audit_subgroup_fairness(y_test, test_probas, test_df['sex'])
display(audit_sex)

print("\\n=== 2. Auditoría de Equidad por Nivel de Ingresos ===")
audit_income = audit_subgroup_fairness(y_test, test_probas, test_df['income_level'])
display(audit_income)

print("\\n=== 3. Auditoría de Equidad por Grupos de Edad ===")
age_bins = pd.cut(test_df['age'], bins=[0, 45, 60, 100], labels=['<45 años', '45-60 años', '>60 años'])
audit_age = audit_subgroup_fairness(y_test, test_probas, age_bins)
display(audit_age)"""),
    ]
    save_notebook(cells, "04_subgroups_fairness_and_shap_analysis.ipynb")


if __name__ == "__main__":
    build_nb01()
    build_nb02()
    build_nb03()
    build_nb04()
    print("All 4 educational notebooks built successfully!")

