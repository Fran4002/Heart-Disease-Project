\begin{titlepage}
    \centering
    \vspace*{2cm}
    {\Large Universidad Autónoma de San Luis Potosí \\[0.25cm] Facultad de Ingeniería}\\[1.5cm]
    {\Huge\bfseries Propuesta de proyecto\\[0.5cm]}
    {\LARGE Modelado predictivo y análisis multifactorial de riesgo de enfermedades cardíacas}\\[1.5cm]
    {\large Profesor: Erick Gómez Martínez}\\[0.15cm]
    {\large APRENDIZAJE AUTOMÁTICO}\\[0.15cm]
    {\large Grupo: 2816 01}\\[1.5cm]
    \vfill
    {\large Francisco Jesús Hernández Hernández}\\[0.25cm]
    {\large San Luis Potosí, México}\\[0.25cm] 
    {\large 23 Septiembre 2026}
\end{titlepage}

\newpage


## 1. Título del proyecto

Modelado predictivo y análisis multifactorial de riesgo de enfermedades cardíacas

## 2. Introducción

Las enfermedades cardiovasculares representan una de las principales causas de mortalidad a nivel mundial. La detección temprana y la evaluación del riesgo individual son fundamentales para implementar intervenciones preventivas eficaces.   En este proyecto se analizará un conjunto de datos sintético calibrado de 100,000 registros clínicos y de estilo de vida correspondiente a población adulta.

Se busca construir modelos predictivos capaces de identificar la presencia de patologías cardíacas, estimar la probabilidad de riesgo de un paciente, analizar el impacto relativo de factores clínicos frente a los de estilo de vida, y evaluar si los modelos presentan un desempeño equitativo a través de diversos subgrupos demográficos y socioeconómicos.

## 3. Planteamiento del problema

El diagnóstico oportuno de enfermedades cardiacas suele depender de pruebas invasivas o análisis clínicos de alto costo que no siempre están disponibles de manera inmediata. A su vez, los factores de riesgo son complejos y abarcan métricas clínicas, hábitos de estilo de vida, datos demográficos y señales continuas provenientes de dispositivos vestibles (*wearables*).

Lograr identificar oportunamente a personas con alto riesgo de enfermedad cardíaca y comprender cómo interactúan los factores clínicos con los hábitos de vida, pude ayudar al personal médico e instituciones de salud, a tomar decisiones al contar con herramientas de apoyo, como la puntuación de riesgo cardiovascular (`heart_disease_prob`), para centrar sus recursos en los pacientes que mas lo necesiten.


## 4. Pregunta de investigación

¿Cual es el factor de riesgo de sufrir una enfermedad cardiaca, evaluando el impacto diferenciado entre factores clínicos y de estilo de vida?

## 5. Objetivo general

Desarrollar, evaluar y comparar modelos de regresión utilizando datos clínicos, demográficos, de estilo de vida y de dispositivos vestibles, para predecir la presencia de enfermedades cardíacas y estimar la puntuación de riesgo cardiovascular.

## 6. Objetivos específicos

1. Preprocesar y analizar exploratoriamente el dataset de 100,000 registros de salud cardiovascular, verificando distribuciones y calidad de las variables.
2. Entrenar y optimizar modelos de clasificación binaria para predecir la presencia de enfermedad cardíaca (`heart_disease = 0 o 1`).
3. Generar y calibrar una puntuación de riesgo (`heart_disease_prob`) mediante el ajuste de un modelo de regresión.
4. Realizar un análisis de importancia de características  para determinar las de mayor influencia.
5. Efectuar un análisis de subgrupos para evaluar si las métricas de desempeño se mantienen según el sexo, grupos de edad y niveles de ingresos.



## 7. Justificación

Este proyecto es relevante tanto desde una perspectiva clínica. En el ámbito práctico, el desarrollo de herramientas predictivas permite priorizar la atención de pacientes en riesgo y diseñar planes de prevención personalizados. Desde el punto de vista del aprendizaje automatico, el dataset ofrece una prevalencia equilibrada (46.1% positivos), lo que elimina la necesidad de técnicas complejas de *overfitting*. Además, permite aplicar una gama diversa de algoritmos (modelos lineales, árboles de decisión), estudiar la interpretabilidad de los modelos y evaluar sesgos en subgrupos socioeconómicos y demográficos.

## 8. Dataset

[Heart Disease Prediction Dataset 2026 @ Kaggle](https://www.kaggle.com/datasets/mohankrishnathalla/heart-disease-prediction-dataset-2026/data)

Este conjunto de datos contiene 100,000 registros de pacientes sintetizados para reflejar los datos de salud cardiovascular del mundo real para la población adulta de los Estados Unidos (de 29 a 80 años). Cada distribución de características, prevalencia de comorbilidad y medición clínica se ha calibrado cuidadosamente, lo que lo hace adecuado para una investigación rigurosa de aprendizaje automático, cursos académicos y modelado de riesgos clínicos.

La variable objetivo `heart_disease` (binario: 0 = No, 1 = Sí) logra una tasa positiva del **46.1%**, proporcionando un desafío de clasificación casi equilibrado sin sobremuestreo artificial.

### Grupos de Características

| Grupo 	        | Características 	        | Conteo |
|-------------------|-------------------|-------|
|Demografía         |age, sex, bmi, region, income_level 	|5|
|Estilo de vida     |smoking_status, alcohol_intake, physical_activity, diet_quality, sleep_hours, stress_level 	|6|
|Historial médico   |diabetes, hypertension, family_history, previous_heart_event 	|4|
|Marcadores clínicos| 	resting_bp, cholesterol, fasting_blood_sugar, max_heart_rate, resting_ecg, exercise_angina, st_depression, st_slope 	|8|
|Dispositivos vestibles| 	avg_resting_hr, hrv_score 	|2|
|Obejetivo       	|heart_disease 	|1|


## 9. Tipo de problema de Machine Learning
El problema se formula dentro del **Aprendizaje Supervisado**:

- **Clasificación binaria:** Tarea central orientada a discriminar entre pacientes sanos y con patología cardíaca (`heart_disease`).
-  **Estimación probabilística / Regresión de riesgo:** Estimación del valor continuo $P(Y=1 \mid X) \in [0, 1]$.

### Algoritmos propuestos:
- **Regresión Logística:** Modelo base interpretable, útil para coeficientes directos y calibración.
- **Random Forest:** Ensamble por agregación que modela no linealidades complejas y reduce varianza.
- **Support Vector Machines (SVM):** Con kernel

## 10. Marco teórico

**Machine Learning Supervisado**: Subcampo de la Inteligencia Artificial donde el algoritmo aprende un mapeo de las variables de entrada $X$ a una etiqueta de salida $Y$ utilizando un conjunto de datos etiquetado.

**Clasificación Binaria**: Tarea predictiva donde la variable objetivo toma únicamente dos valores discretos (p. ej., presencia o ausencia de enfermedad). 

**Calibración de Probabilidades y Regresión Logística**: Función sigmoide que transforma combinaciones lineales de variables en un rango $[0, 1]$, interpretable como la probabilidad clínica del evento.

## 11. Metodología propuesta

Debe explicar cómo piensan realizar el proyecto.
Una estructura sencilla que pueden seguir:

### Paso 1. Obtención de datos

Carga del conjunto de datos sintético estructurado de 100,000 registros clínicos. (`Python: Pandas`)

### Paso 2. Preparación de datos

- Verificación de consistencia y tratamiento de posibles valores atípicos o inconsistencias de formato.
- Codificación de variables categóricas nominales (`region`, `sex`) mediante *One-Hot Encoding* y ordinales (`income_level`, `diet_quality`) mediante *Ordinal Encoding*.
- Estandarización / Normalización ($Z$-score o *Min-Max*) de variables numéricas continuas para modelos sensibles a la escala (SVM, Regresión Logística).

### Paso 3. Análisis exploratorio

- Análisis de estadísticas descriptivas (media, mediana, desviación estándar, rangos intercuartílicos).
- Evaluación de correlaciones (matrices de Pearson y Spearman).
- Comparación preliminar entre variables clínicas y de estilo de vida.   

### Paso 4. División de los datos

Partición estratificada en tres subconjuntos para preservar la tasa de positivos (46.1%):
- **Entrenamiento (70%)**
- **Validación (15%)**
- **Prueba (15%)**

### Paso 5. Entrenamiento

- Entrenamiento del modelo base (Regresión Logística).
- Ajuste de hiperparámetros (profundidad, tasa de aprendizaje, regularización) en Random Forest y XGBoost

### Paso 6. Evaluación

- **Métricas globales:** Curva ROC y valor AUC-ROC, Precision-Recall Curve (PR-AUC), Precisión, Sensibilidad (*Recall*), Especificidad y puntuación $F_1$. En aplicaciones médicas se priorizará un alto *Recall* para minimizar falsos negativos.
- **Evaluación de importancia:** Desglose del impacto de variables clínicas vs. estilo de vida mediante valores SHAP.
- **Auditoría de subgrupos:** Desglose de métricas segmentando por `sex`, deciles de `age` y niveles de `income_level`.
- **Emparejamiento / Supervivencia:** Análisis exploratorio de riesgo condicional respecto a la edad del primer evento (`previous_heart_event`).

### Paso 7. Comparación

Selección del modelo final evaluando el mejor equilibrio entre capacidad de discriminación (AUC-ROC), calibración probabilística e interpretabilidad clínica.

## 12. Resultados esperados
Un modelo de ensamble (*XGBoost* o *Random Forest*) que alcance un alto desempeño en la clasificación binaria de enfermedad cardíaca

Obtención de un estimador probabilístico (`heart_disease_prob`) utilizable como puntuación de riesgo

## 13. Referencias preliminares

1. Mohan Krishna Thalla. (2026). *Heart Disease Prediction Dataset 2026*. [Kaggle]. (https://www.kaggle.com/datasets/mohankrishnathalla/heart-disease-prediction-dataset-2026/data)

2. World Health Organization. (2025). *Cardiovascular diseases (CVDs)*. [WHO Fact Sheets](https://www.who.int/news-room/fact-sheets/detail/cardiovascular-diseases-(cvds)).

3. Eda Kavlakoglu. (2021). *What is random forest?*. [IBM Think](https://www.ibm.com/think/topics/random-forest)