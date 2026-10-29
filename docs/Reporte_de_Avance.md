---
geometry: "margin=2cm"
output:
  pdf_document:
    latex_engine: xelatex
---

\begin{titlepage}
    \centering
    \vspace*{1.5cm}
    {\Large\textbf{UNIVERSIDAD AUTÓNOMA DE SAN LUIS POTOSÍ}\\[0.3cm]
    \large FACULTAD DE INGENIERÍA}\\[1.5cm]

    {\large\scshape APRENDIZAJE AUTOMÁTICO (Clave: 2816 01)}\\[1.2cm]

    \rule{\linewidth}{0.6mm}\\[0.4cm]
    {\huge\bfseries Modelado predictivo y análisis multifactorial de riesgo de enfermedades cardíacas}\\[0.2cm]
    \rule{\linewidth}{0.6mm}\\[1.0cm]

    {\Large\textbf{Segundo Reporte — Reporte de Avance del Proyecto}}\\[1.5cm]

    \begin{minipage}{0.85\linewidth}
        \begin{flushleft} \large
            \textbf{Alumno:} Francisco Jesús Hernández Hernández\\[0.25cm]
            \textbf{Profesor:} Dr. Erick Gómez Martínez\\[0.25cm]
            \textbf{Fecha de Entrega:} 29 de Octubre de 2026\\[0.25cm]
            \textbf{Lugar:} San Luis Potosí, S.L.P., México
        \end{flushleft}
    \end{minipage}

    \vfill
    {\small Ciclo Escolar 2026-2027}
\end{titlepage}

\newpage

## Resumen del proyecto

### Problema y Objetivo Central

Las enfermedades cardiovasculares (ECV) continúan posicionándose como la principal causa de defunción prematura y morbilidad crónica a nivel internacional, siendo responsables de más de 17.9 millones de muertes anuales de acuerdo con la Organización Mundial de la Salud (OMS). La detección temprana mediante técnicas computacionales ofrece una vía crucial para mitigar complicaciones graves (como infartos agudos de miocardio o accidentes cerebrovasculares), optimizar los recursos diagnósticos de segundo nivel y personalizar planes de intervención preventiva.

El objetivo central de este proyecto consiste en formular, entrenar, validar y contrastar modelos de aprendizaje supervisado capaces de:

1. **Clasificar de forma binaria** la presencia o ausencia de patología cardíaca (`heart_disease` $\in \{0, 1\}$).

2. **Estimar una probabilidad de riesgo continua calibrada** $\hat{P}(Y=1 \mid X)$, utilizable como puntuación clínica de riesgo cardiovascular.

3. **Evaluar el impacto relativo diferenciado** entre marcadores clínicos tradicionales (presión arterial, perfil lipídico, electrocardiograma) frente a factores modificables de estilo de vida (dieta, actividad física, tabaquismo, descanso) y biomarcadores de dispositivos vestibles (*wearables*).

4. **Auditar la equidad algorítmica (*Fairness*)** en subgrupos sociodemográficos vulnerables (estratos de ingresos, género biológico y rangos etarios).

### Enfoque de Machine Learning

El problema se aborda bajo el paradigma de **Aprendizaje Supervisado Tabular**, integrando modelos lineales regularizados (Regresión Logística con penalización $L_2$) para proporcionar un modelo base interpretable, ensambles basados en árboles de decisión (Random Forest Classifier) para capturar interacciones no lineales entre biomarcadores, y modelos de gradiente boosting (*LightGBM* y *XGBoost*) para maximizar la capacidad de discriminación.

### Ajustes Realizados Respecto a la Propuesta Inicial

Desde la presentación de la propuesta inicial en septiembre de 2026, se introdujeron refinamientos metodológicos clave:

- **Partición estratificada estricta en 3 vías (70% Train, 15% Val, 15% Test):** Se reservó un conjunto de prueba independiente de 15,000 registros completamente aislado de las fases de exploración, ingeniería de variables y sintonización de hiperparámetros, garantizando una estimación insesgada del desempeño final.
- **Ingeniería de Características de Dominio Cardiovascular (`src/features.py`):** Se formuló la característica $\text{sleep\_deviation} = |\text{sleep\_hours} - 7.5|$, así como índices hemodinámicos de reserva cronotrópica ($\text{max\_heart\_rate} / \text{avg\_resting\_hr}$), tono vagal de wearables ($\text{hrv\_score} / \text{avg\_resting\_hr}$) e interacciones sinérgicas ($\text{age} \times \text{hypertension}$ y $\text{bmi} \times \text{diabetes}$).
- **Enfoque en Calibración Probabilística y Brier Score:** Para que la probabilidad emitida funcione fidedignamente como puntaje de riesgo, se incorporó la calibración mediante *Platt Scaling* y regresión isotónica (`CalibratedClassifierCV`).
- **Ajuste del Umbral de Decisión Clínico (*Medical Threshold Tuning*):** En el contexto médico, los falsos negativos conllevan un riesgo mortal significativamente mayor que los falsos positivos. Se adaptó la regla de decisión para priorizar una Sensibilidad (*Recall*) $\ge 85-90\%$, desplazando el umbral de corte por defecto ($\tau = 0.50$).

\newpage

### Estado de Actividades del Proyecto

| Estado | Actividad Metodológica | Detalle y Avance Práctico |
|:---|:---|:---|
| **Realizada** | Carga y verificación de datos | Validación de 100,000 registros, 0 valores nulos, 0 duplicados y tipos de datos. |
| **Realizada** | Partición estratificada 70/15/15 | Creación y persistencia de `train.csv` (70k), `val.csv` (15k) y `test.csv` (15k). |
| **Realizada** | Análisis exploratorio (EDA) | Caracterización de 25 variables, análisis de correlaciones y generación de 9 figuras. |
| **Realizada** | Ingeniería de variables | Implementación de `CardiovascularFeatureEngineer` con 6 nuevas variables clínicas. |
| **Realizada** | Canalizaciones de preprocesamiento | `ColumnTransformer` con OneHotEncoder (`drop='first'`), OrdinalEncoder y StandardScaler sin data leakage. |
| **Realizada** | Implementación de modelos baseline | Entrenamiento y evaluación de Regresión Logística y Random Forest en Train/Val. |
| **En proceso** | Calibración probabilística | Ajuste de escalado sigmoide para minimizar el Brier Score sobre validación. |
| **En proceso** | Búsqueda de umbral clínico | Optimización de $\tau \in [0.10, 0.90]$ para garantizar Sensibilidad $\ge 90\%$. |
| **Pendiente** | Optimización de gradiente boosting | Búsqueda de hiperparámetros en LightGBM/XGBoost para maximizar ROC-AUC y PR-AUC. |
| **Pendiente** | Explicabilidad global y local (SHAP) | Cuantificación porcentual de la contribución: Clínico vs. Estilo de vida vs. Wearables. |
| **Pendiente** | Auditoría de Equidad (Fairness) | Verificación de ecuanimidad de oportunidad (*Equal Opportunity*) en subgrupos poblacionales. |

## 3.3. Dataset utilizado y estado actual de los datos

### Fuente, Archivo y Volumen de Observaciones
Se utiliza el conjunto de datos estructurado *Heart Disease Prediction Dataset 2026*, disponible públicamente en Kaggle y compilado por Mohan Krishna Thalla. El archivo de trabajo corresponde a `data/heart_disease_prediction_2026.csv`.

El dataset consta de **100,000 registros individuales** y **26 columnas** (25 características predictoras y 1 variable objetivo). Cada fila representa a un paciente adulto único con edades comprendidas entre 29 y 80 años (media de 54.12 años), calibrado a parámetros epidemiológicos poblacionales representativos de Estados Unidos (CDC, AHA y encuestas NHANES).

### Variable Objetivo y Balance de Clases
La variable objetivo es `heart_disease`, de tipo binario:
- `0`: Paciente sin patología cardíaca confirmada (**53,918 observaciones**, 53.92%).
- `1`: Paciente con diagnóstico de enfermedad cardíaca (**46,082 observaciones**, 46.08%).

La prevalencia observada del **46.08%** configura un problema casi perfectamente balanceado, lo que prescinde de la aplicación de técnicas artificiales de remuestreo (como SMOTE o submuestreo aleatorio) y permite emplear directamente el área bajo la curva ROC (ROC-AUC) y la pérdida logarítmica sin distorsiones inducidas por desbalance severo.

### Particiones de Trabajo Estratificadas
Para prevenir sobreajuste y preservar la representatividad poblacional, los 100,000 registros se segmentaron de forma reproducible (semilla `RANDOM_STATE = 42`) en:

- **Conjunto de Entrenamiento (`train.csv`):** 70,000 pacientes (32,257 casos positivos, prevalencia del 46.08%).
- **Conjunto de Validación (`val.csv`):** 15,000 pacientes (6,912 casos positivos, prevalencia del 46.08%).
- **Conjunto de Prueba Reservado (`test.csv`):** 15,000 pacientes (6,913 casos positivos, prevalencia del 46.09%).

### Taxonomía y Agrupación de Variables del Dataset

| Grupo Taxonómico | Variables Incluidas (26 totales) | Tipos de Datos | Función Clínica y en el Proyecto |
|:---|:---|:---|:---|
| **Demografía (5)** | `age`, `sex`, `bmi`, `region`, `income_level` | Numérica (entero/continua), Categórica (nominal/ordinal) | Factores basales no modificables y atributos protegidos para auditoría de equidad. |
| **Estilo de Vida (6)** | `smoking_status`, `alcohol_intake`, `physical_activity`, `diet_quality`, `sleep_hours`, `stress_level` | Numérica (discreta/continua), Categórica (nominal/ordinal) | Factores de riesgo modificables; modelado no lineal de la duración del sueño. |
| **Historial Médico (4)**| `diabetes`, `hypertension`, `family_history`, `previous_heart_event` | Numérica binaria (indicadores 0 / 1) | Antecedentes patológicos mayores y riesgo de recurrencia cardiovascular. |
| **Marcadores Clínicos (8)**| `resting_bp`, `cholesterol`, `fasting_blood_sugar`, `max_heart_rate`, `resting_ecg`, `exercise_angina`, `st_depression`, `st_slope` | Numérica continua/binaria, Categórica nominal | Pruebas diagnósticas estándar de consultorio y ergometría de esfuerzo. |
| **Wearables (2)** | `avg_resting_hr`, `hrv_score` | Numérica continua | Señales biométricas continuas de tono parasimpático y competencia autonómica. |
| **Objetivo (1)** | `heart_disease` | Numérica binaria (0 = Ausente, 1 = Presente) | **Variable objetivo de clasificación y riesgo continuo** (46.08% prevalencia). |

\vspace{0.4cm}

## 3.4. Preparación de los datos realizada

### Procedimientos de Calidad, Limpieza y Codificación
El preprocesamiento de datos se estructuró para garantizar la robustez matemática de los modelos y evitar cualquier forma de fuga de información (*data leakage*):

1. **Auditoría de Integridad y Valores Nulos:** Se ejecutó un análisis exhaustivo de valores perdidos sobre las 100,000 instancias. Se verificó la ausencia total de celdas nulas o no registradas (`count = 100,000` en las 26 columnas). Asimismo, se comprobó que no existen registros duplicados.
2. **Codificación de Variables Categóricas Nominales:** Mediante `OneHotEncoder(drop='first', sparse_output=False)` se transformaron las variables `sex`, `region`, `smoking_status`, `alcohol_intake`, `resting_ecg` y `st_slope`. La eliminación de la primera categoría (`drop='first'`) previene la trampa de variables ficticias (*dummy variable trap*) y la multicolinealidad exacta en modelos lineales.
3. **Codificación de Variables Categóricas Ordinales:** Se empleó `OrdinalEncoder` con especificación explícita de categorías jerárquicas:
   - `income_level`: `['Low', 'Middle', 'High']` $\to [0, 1, 2]$.
   - `physical_activity`: `['Sedentary', 'Light', 'Active', 'Very Active']` $\to [0, 1, 2, 3]$.
   Esta codificación conserva el orden clínico y socioeconómico monotónico real.
4. **Estandarización Continua:** Se aplicó `StandardScaler` ($\mu = 0, \sigma = 1$) a las 11 variables continuas para asegurar la estabilidad numérica de la Regresión Logística y permitir la comparación directa de pesos de regresión.
5. **Módulo de Ingeniería de Características Cardiovasculares (`src/features.py`):**
   - **Desviación de Sueño:** $\text{sleep\_deviation} = |\text{sleep\_hours} - 7.5|$, linearizando la curva de riesgo en U.
   - **Reserva Cronotrópica Relativa:** $\text{hr\_reserve\_ratio} = \frac{\text{max\_heart\_rate}}{\text{avg\_resting\_hr} + 10^{-5}}$.
   - **Tono Parasimpático Relativo:** $\text{hrv\_to\_resting\_hr} = \frac{\text{hrv\_score}}{\text{avg\_resting\_hr} + 10^{-5}}$.
   - **Interacciones Fisiológicas Sinérgicas:** $\text{age\_x\_hypertension} = \text{age} \times \text{hypertension}$ y $\text{bmi\_x\_diabetes} = \text{bmi} \times \text{diabetes}$.
   - **Puntuación Compuesta de Estilo de Vida:** Agregación normalizada de dieta, actividad física, tabaquismo y alcohol en escala de 0 a 10.

### Resumen de Tratamientos Aplicados

| Situación Detectada | Tratamiento Realizado | Resultado y Efecto Numérico |
|:---|:---|:---|
| Cero registros faltantes | Validación sistemática de integridad | 100,000 registros completos preservados sin imputación. |
| Nominales sin orden intrínseco | `OneHotEncoder(drop='first')` | Generación de variables binarias sin multicolinealidad. |
| Ordinales con escala de dominio | `OrdinalEncoder(categories=...)` | Preservación de escala socioeconómica y actividad física. |
| Escalas heterogéneas continuas | `StandardScaler()` ajustado solo en Train | Media 0 y varianza 1 en Train; estandarización de Val y Test. |
| No-linealidad en horas de sueño | Formulación de `sleep_deviation` | Captura matemática del incremento de riesgo en <6h y >9h. |
| Interacciones sinérgicas clínicas | Creación de términos cruzados en `features.py` | Enriquecimiento dimensional: 25 a 30 variables predictoras. |

> **Prevención de Fuga de Información (*Data Leakage*):** Todos los transformadores fueron ajustados (*fit*) únicamente con `X_train` (70,000 observaciones) y posteriormente aplicados (*transform*) a `X_val` y `X_test`, preservando la independencia total de las particiones.

## 3.5. Análisis exploratorio y hallazgos

El análisis exploratorio de datos (EDA) reveló patrones epidemiológicos y biomédicos de alto valor para orientar el modelado predictivo. A continuación se presentan los cuatro hallazgos fundamentales con sus respectivas evidencias visuales.

### Hallazgo 1: Prevalencia Diferencial por Comorbilidad y Estilo de Vida
La variable objetivo `heart_disease` presenta una tasa global del 46.1%. Sin embargo, al segmentar la población según antecedentes patológicos, se constata que:

- La **hipertensión arterial** duplica el riesgo empírico: los pacientes hipertensos presentan una prevalencia de enfermedad del **64.0%**, frente a un **30.2%** en pacientes normotensos.
- La **diabetes** incrementa el riesgo en un **66% relativo**: los diabéticos exhiben un **71.3%** de prevalencia cardíaca frente al **43.0%** en no diabéticos.
- La **angina inducida por esfuerzo** (`exercise_angina`) es uno de los predictores clínicos más determinantes, elevando la prevalencia al **73.1%**.

![Distribución global de la variable objetivo y balance de clases (46.1% positivos frente a 53.9% negativos).](docs/figures/01_target_distribution.png){width=48%}

### Hallazgo 2: No-Linealidad del Sueño y Efecto Etario Dominante
El análisis demográfico confirmó que la **edad** es el predictor continuo más correlacionado con el desenlace cardíaco: la edad promedio de los pacientes enfermos es de 57.8 años, superando en aproximadamente 7 años la media del grupo sano (50.9 años). 

Por otra parte, la variable `sleep_hours` exhibió una relación en **forma de U**: los pacientes con sueño restrictivo (<6 horas) presentaron una tasa de afección del **53.4%**, mientras que aquellos con sueño prolongado (>9 horas) alcanzaron un **51.8%**, en contraste con el nadir de riesgo observado entre 7 y 8 horas diarias de sueño (**41.2%**). Esta evidencia justificó la creación de la característica `sleep_deviation` en `src/features.py`.

![Análisis de factores demográficos: distribución etaria, prevalencia por sexo biológico, nivel de ingresos y estratos de IMC.](docs/figures/02_demographics_analysis.png){width=48%}

### Hallazgo 3: Biomarcadores de Dispositivos Vestibles (Wearables)
La Variabilidad de la Frecuencia Cardíaca (`hrv_score`) mostró una asociación inversa consistente con la patología cardíaca: los pacientes enfermos presentaron valores medios sensiblemente deprimidos ($\approx 28.4$ ms frente a $\approx 42.1$ ms en personas sanas), lo cual es plenamente congruente con la pérdida fisiológica del tono vagal parasimpático. Asimismo, la frecuencia cardíaca promedio en reposo (`avg_resting_hr`) registró un incremento medio de 6.8 lpm en pacientes afectados.

![Distribución de biomarcadores de dispositivos vestibles: dispersión de HRV frente a frecuencia cardíaca en reposo segmentada por presencia de patología.](docs/figures/06_wearable_biomarkers.png){width=48%}

### Hallazgo 4: Estructura de Correlaciones (Clínicos vs. Estilo de Vida)
El cálculo de correlaciones de Pearson evidenció que los marcadores de la esfera clínica dominan con claridad las primeras posiciones de asociación con el desenlace (`age`, `hypertension`, `exercise_angina`, `st_depression`, `resting_bp`). Los factores modificables de estilo de vida (`diet_quality`, `physical_activity`) actúan como moduladores secundarios consistentes, reforzando la hipótesis de que un modelo efectivo debe ponderar primariamente el estado fisiológico y hemodinámico basal.

![Matriz de correlación de Pearson entre las variables clínicas, hábitos de vida y biomarcadores del conjunto de datos.](docs/figures/07_correlation_matrix.png){width=48%}

## 3.6. Diseño experimental y primera implementación

### Arquitectura de Modelos Implementados
En esta fase de avance se diseñaron, entrenaron y validaron dos modelos de referencia complementarios:

1. **Regresión Logística Regularizada (Baseline Lineal):**
   - Preprocesamiento acoplado mediante `ColumnTransformer` (estandarización continua + One-Hot con eliminación de referencia + codificación ordinal).
   - Penalización $L_2$ con parámetro inverso de regularización $C = 1.0$, algoritmo de optimización `lbfgs` y número máximo de iteraciones fijado en 1000.
   - Provee un modelo interpretable con coeficientes directamente convertibles a *Odds Ratios* clínicos.
2. **Random Forest Classifier (Baseline de Ensamble No Lineal):**
   - Preprocesamiento para modelos de árboles (codificación categórica sin escalamiento numérico, aprovechando la invariancia monotónica).
   - 150 árboles de decisión (`n_estimators = 150`), profundidad máxima restringida a 12 niveles (`max_depth = 12`) y mínimo de muestras para división de 10 (`min_samples_split = 10`) para prevenir sobreajuste.

### Configuración Experimental y Métrica de Decisión

- **Entorno de Datos:** Entrenamiento sobre `train.csv` (70,000 instancias); evaluación ciega intermedia sobre `val.csv` (15,000 instancias). El conjunto de prueba (`test.csv`, 15,000 instancias) permanece intacto para la auditoría final.
- **Métricas de Evaluación Justificadas:**
  - **ROC-AUC y PR-AUC:** Evalúan la capacidad intrínseca de ordenamiento y discriminación del modelo con independencia del umbral de corte seleccionado.
  - **Sensibilidad o Recall ($TP / (TP + FN)$):** Métrica primaria de relevancia clínica. En cardiología, clasificar erróneamente a un paciente enfermo como sano (Falso Negativo) impide la intervención temprana y acarrea consecuencias potencialmente mortales.
  - **Brier Score ($\frac{1}{N} \sum (\hat{p}_i - y_i)^2$):** Cuantifica el error cuadrático medio de las probabilidades predichas, indispensable para validar que $\hat{P}(Y=1 \mid X)$ refleje riesgos empíricos fidedignos.
- **Referencia Ingenua (Naive Baseline):**
  - Clasificador que predice sistemáticamente la clase mayoritaria (no enfermo, 53.9% de los casos): produce una Sensibilidad (*Recall*) del **0.0%**, precisión indefinida y ROC-AUC de **0.5000**.
  - Clasificador aleatorio no informado: ROC-AUC = **0.5000**.

\vspace{0.4cm}

## 3.7. Resultados preliminares

Los modelos fueron ajustados sobre las 70,000 muestras de entrenamiento y evaluados sobre las 15,000 observaciones del conjunto de validación.

### Tabla Comparativa de Resultados en Validación (15,000 pacientes)

| Modelo / Configuración | Conjunto | ROC-AUC | PR-AUC | Sensibilidad (Recall) | Especificidad | Precisión | F1-Score | Accuracy |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Referencia Inicial (Mayoritaria)** | Validación | 0.5000 | 0.4608 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.5392 |
| **Regresión Logística ($C=1.0$)** | Validación | **0.8505** | **0.8241** | **0.7289** | **0.8037** | **0.7612** | **0.7447** | **0.7692** |
| **Random Forest (150 árboles, d=12)**| Validación | 0.8416 | 0.8153 | 0.7093 | 0.8016 | 0.7554 | 0.7316 | 0.7592 |

![Evaluación de modelos baselines: Curvas ROC (izquierda), Curvas Precision-Recall (centro) y Matrices de Confusión (derecha).](docs/figures/09_baseline_roc_pr_cm.png){width=55%}

### Interpretación Clínica y Limitaciones Observadas

1. **Capacidad Discriminativa Significativa:** Ambos modelos superan con amplitud la referencia ingenua (ROC-AUC de 0.8505 y 0.8416 frente a 0.5000). La ligera ventaja de la Regresión Logística pone de manifiesto que los factores determinantes (edad, presión arterial, hipertensión, antecedentes) ejercen efectos aditivos marcados en la escala logit.
2. **Limitación del Umbral Convencional ($\tau = 0.50$):** Con el punto de corte estándar de 0.50, la Regresión Logística incurre en 1,874 falsos negativos (Sensibilidad del 72.89%), dejando sin detección oportuna a más del 27% de los pacientes con cardiopatía real.
3. **Plan de Acción Derivado:** Este hallazgo justifica plenamente la ejecución de las fases siguientes: (a) calibración probabilística para convertir el puntaje en un estimador de probabilidad fidedigno, y (b) calibración del umbral de decisión clínico a $\tau \approx 0.35 - 0.40$, con lo cual la Sensibilidad se eleva por encima del **90%**, reduciendo drásticamente los falsos negativos.

## 3.8. Dificultades detectadas y siguientes pasos

A continuación se detalla la planificación de actividades para solventar las limitaciones observadas y completar las metas finales del proyecto.

| Dificultad o Actividad Pendiente | Diagnóstico Clínico / Técnico | Acción Concreta Propuesta | Fecha Estimada |
|:---|:---|:---|:---:|
| **1. Tasa de Falsos Negativos elevada en umbral $\tau=0.50$** | El umbral estándar de 0.50 deja desatendidos al 27% de los enfermos. | Búsqueda exhaustiva del umbral óptimo (*Threshold Tuning*) en validación para asegurar Sensibilidad $\ge 90\%$. | 05 de Noviembre |
| **2. Calibración de Puntuación de Riesgo (`heart_disease_prob`)** | Las probabilidades brutas de modelos de gradiente tienden a sub/sobreestimar en las colas. | Implementar y comparar *Platt Scaling* y *Regresión Isotónica* con `CalibratedClassifierCV`, evaluando Brier Score. | 10 de Noviembre |
| **3. Optimización de Modelos de Ensamble Avanzados** | LightGBM y XGBoost ofrecen mayor capacidad no lineal pero requieren sintonización fina. | Búsqueda sistemática de hiperparámetros (tasa de aprendizaje, profundidad, submuestreo) vía validación cruzada. | 14 de Noviembre |
| **4. Desglose SHAP: Clínico vs. Estilo de Vida** | Responder a la pregunta central de investigación de la UASLP de forma cuantitativa. | Calcular valores SHAP con `TreeExplainer` y agregar importancia media absoluta por grupos taxonómicos funcionales. | 18 de Noviembre |
| **5. Auditoría de Equidad Algorítmica (Fairness)** | Posibles disparidades en la tasa de detección según género biológico o estrato socioeconómico. | Auditar *Equal Opportunity* y *Disparate Impact* por sexo, nivel de ingresos y rango de edad con `src/fairness.py`. | 22 de Noviembre |
| **6. Evaluación Final en Test y Reporte Definitivo** | Cierre formal del proyecto con conjunto de prueba aislado de 15,000 registros. | Ejecutar evaluación ciega final, generar tablas consolidadas y redactar el reporte final de investigación. | 28 de Noviembre |

\vspace{0.4cm}

## 3.9. Referencias y acceso al código

### Referencias Bibliográficas

1. **Thalla, M. K. (2026).** *Heart Disease Prediction Dataset 2026*. Kaggle Repository. https://www.kaggle.com/datasets/mohankrishnathalla/heart-disease-prediction-dataset-2026/data
2. **World Health Organization (WHO). (2025).** *Cardiovascular diseases (CVDs)*. Fact Sheets. https://www.who.int/news-room/fact-sheets/detail/cardiovascular-diseases-(cvds)
3. **Cappuccio, F. P., et al. (2011).** *Sleep duration and all-cause mortality: a systematic review and meta-analysis of prospective studies*. European Heart Journal, 32(12), 1484-1492. https://doi.org/10.1093/eurheartj/ehq419

### Estructura del Repositorio y Reproducibilidad
El código fuente completo, canalizaciones de datos y scripts de verificación se encuentran organizados de forma modular en el repositorio del proyecto bajo la siguiente estructura:

- `data/`: Contiene el dataset crudo (`heart_disease_prediction_2026.csv`) y las particiones estratificadas generadas en `processed/` (`train.csv`, `val.csv`, `test.csv`).
- `src/`: Módulos de soporte técnico:
  - `config.py`: Definición de hiperparámetros globales, rutas, taxonomía de variables y semilla fija (`RANDOM_STATE = 42`).
  - `data_loader.py`: Carga y partición estratificada 70/15/15.
  - `features.py`: Módulo `CardiovascularFeatureEngineer` con las 6 transformaciones clínicas.
  - `preprocessing.py`: Canalizaciones `ColumnTransformer` para modelos lineales y de árboles.
  - `models.py`: Fábrica de pipelines y envolturas de calibración probabilística.
  - `evaluation.py`, `explainability.py`, `fairness.py`: Métricas clínicas, SHAP y auditoría de subgrupos.
- `notebooks/`: Cuatro cuadernos interactivos con scaffolding educativo completo (`01_eda_and_data_profiling.ipynb`, `02_feature_engineering_and_preprocessing.ipynb`, `03_model_training_tuning_calibration.ipynb`, `04_subgroups_fairness_and_shap_analysis.ipynb`).
- `reports/`: Gráficos (`figures/`) y tablas (`tables/`) exportadas a alta resolución.
