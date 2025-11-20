# Reporte técnico – Modelo de Flakiness E2E

## 1. Objetivo

El objetivo de este proyecto es **analizar y predecir la flakiness** en ejecuciones end-to-end (E2E) de la app, a partir de un histórico de corridas almacenado en el archivo:

- `data/e2e_runs_meli_fake.csv`

El modelo produce, para cada corrida E2E:

- `p_fail`: probabilidad de que la corrida tenga **al menos un escenario fallido**.
- Una predicción binaria `predicted_failed` a partir de un umbral (`threshold`, típicamente 0.5).
- Explicaciones locales por corrida (importancia de features), expuestas vía API.

---

## 2. Dataset y definición del problema

### 2.1. Origen y granularidad

Cada fila del dataset representa una **corrida E2E** (ejecución de un conjunto de escenarios) asociada a:

- un **squad**,
- un **device**,
- un **environment** (`test_app` / `release_app`),
- una **plataforma** (`android`, `ios`, `web`, etc.),
- un **timestamp** (fecha/hora de la ejecución).

### 2.2. Variables clave

Campos relevantes utilizados en el pipeline:

- `timestamp`: fecha/hora de la corrida.
- `squad`: equipo responsable (checkout, search, etc.).
- `platform`: plataforma (`android`, `ios`, `web`…).
- `environment`: ambiente (`test_app`, `release_app`).
- `device_id`: identificador de dispositivo/emulador.
- `memory_mb`: memoria disponible del device (MB).
- `scenarios_total`: cantidad total de escenarios ejecutados.
- `scenarios_failed`: cantidad de escenarios fallidos.

A partir de estos se derivan:

- `fail_rate = scenarios_failed / scenarios_total`
- `run_failed = 1 si scenarios_failed > 0, en caso contrario 0`
- Features de tiempo:
  - `hour`: hora del día (0–23).
  - `dayofweek`: día de la semana (0=Lunes…6=Domingo).

### 2.3. Formulación del problema

Problema formulado como **clasificación binaria**:

- **Target**: `run_failed` (1 = corrida con al menos un escenario fallido, 0 = totalmente verde).
- **Salida del modelo**: `p_fail = P(run_failed=1 | features)`.

---

## 3. Metodología

### 3.1. EDA (Exploratory Data Analysis)

Script: `src/eda_flakiness.py`  
Objetivos del EDA:

- Entender la **distribución de fail_rate**.
- Identificar **patrones de flakiness**:
  - por `squad`,
  - por `device_id`,
  - por `platform` + `environment`,
  - por `iteration` (si existe).
- Detectar grupos potencialmente **flaky** basados en:
  - `min_runs` por grupo (e.g. ≥ 5 ejecuciones),
  - **alta variabilidad** de `fail_rate` (desviación estándar elevada).

Salidas:

- Tablas agregadas (`reports/by_squad.csv`, `reports/by_device.csv`, etc.).
- Gráficos (`reports/plots/*.png`) con:
  - boxplots de `fail_rate` por squad/device/plataforma/ambiente,
  - serie temporal de `fail_rate` promedio diario.
- Resumen de texto en `reports/summary.txt`.

### 3.2. Ingeniería de características

Features usadas para el modelo:

- **Categóricas**:
  - `squad`
  - `platform`
  - `environment`
  - `device_id`
- **Numéricas**:
  - `memory_mb`
  - `scenarios_total`
  - `hour` (derivada de `timestamp`)
  - `dayofweek` (derivada de `timestamp`)

Pipeline de preprocesamiento:

- `OneHotEncoder` para categóricas (con `handle_unknown="ignore"`).
- `StandardScaler` para numéricas.

Todo se integra vía `ColumnTransformer` + `Pipeline` de scikit-learn.

### 3.3. Partición Train/Test

- Se realiza un **train/test split**:
  - `test_size = 0.2` (20% para test).
  - `stratify=y` para preservar la proporción de clases.
  - `random_state = 42` para reproducibilidad.

### 3.4. Modelo

Modelo principal:

- `GradientBoostingClassifier` de scikit-learn.

Motivación:

- Funciona bien en datos tabulares.
- Entrega probabilidades (`predict_proba`).
- Captura interacciones no lineales sin requerir mucho feature engineering manual.

### 3.5. Entrenamiento y validación

Script: `src/train_flakiness_model.py`

Pasos:

1. Carga del CSV y construcción de `X` (features) e `y` (`run_failed`).
2. Split en train/test.
3. Entrenamiento del pipeline completo (preprocesamiento + modelo).
4. Evaluación en test set con las métricas descritas en la sección 4.
5. Persistencia:
   - `models/flakiness_model.joblib` → pipeline completo.
   - `models/flakiness_model_metrics.txt` → reporte de métricas.

---

## 4. Métricas de desempeño

Las métricas se guardan en el archivo:

- `models/flakiness_model_metrics.txt`

Incluye:

- `classification_report` (precision, recall, F1 por clase).
- `ROC-AUC`.
- `confusion_matrix`.
- Distribución de la variable objetivo en test.

### 4.1. Métricas principales

| Métrica                | Clase / Global | Valor  | Comentario breve                                    |
|------------------------|----------------|--------|-----------------------------------------------------|
| F1-score               | Clase 1 (fail) | 0.95   | Balance entre precision y recall para corridas fail |
| F1-score               | Clase 0 (ok)   | 0.89   |                                                     |
| ROC-AUC                | Global         | 0.97   | Capacidad de rankear corridas fail vs ok            |
| Precision (fail)       | Clase 1        | 0.95   | Proporción de predicciones fail que realmente fallan|
| Recall (fail)          | Clase 1        | 0.94   | Cobertura de corridas fail                          |
| Support (fail)         | Clase 1        | 1506   | Cantidad de corridas fail en test                  |
| Support (ok)           | Clase 0        | 728    | Cantidad de corridas ok en test                    |

### 4.2. Interpretación de ROC-AUC

- ROC-AUC mide la capacidad del modelo de **ordenar corridas** desde más propensas a fallar hasta menos.
- Un valor de:
  - ~0.5 → modelo no mejor que azar.
  - 0.7–0.8 → razonable en problemas ruidosos como flakiness.
  - 0.8–0.9 → muy bueno.
- En contexto E2E, un ROC-AUC decente permite **priorizar corridas de alto riesgo** (por ejemplo, para re-ejecución, debugging o alertas tempranas).

### 4.3. Interpretación de F1

- F1 combina **precision** y **recall** en una única métrica.
- Especialmente relevante para la clase 1 (corridas fallidas), porque:
  - Queremos **detectar la mayoría de corridas problemáticas** (buen recall),
  - Sin disparar demasiados falsos positivos (mantener precision razonable).

---

## 5. Explainability (SHAP/LIME) y enfoque actual

### 5.1. Enfoque liviano implementado en la API

Endpoint: `POST /explain` (`src/flakiness_api.py`)

En lugar de integrar directamente librerías pesadas como SHAP o LIME en runtime, la API implementa un enfoque **ligero de sensibilidad local**:

1. Se calcula `p_fail` original para cada corrida (`p0`).
2. Se define un **baseline por feature** usando las corridas enviadas en el batch:
   - Para variables **categóricas** (`squad`, `platform`, `environment`, `device_id`):
     - Baseline = **moda** (valor más frecuente).
   - Para variables **numéricas** (`memory_mb`, `scenarios_total`, `hour`, `dayofweek`):
     - Baseline = **mediana**.
3. Para cada corrida y para cada feature `f`:
   - Se crea una copia de la corrida reemplazando `f` por su baseline.
   - Se vuelve a predecir la probabilidad: `p_new`.
   - Se calcula:  
     `delta_p_fail = p0 - p_new`
4. Interpretación de `delta_p_fail`:
   - `delta_p_fail > 0`:
     - La feature actual **incrementa** el riesgo de fallo frente a la referencia.
   - `delta_p_fail < 0`:
     - La feature actual **disminuye** el riesgo de fallo frente a la referencia.
5. El endpoint ordena las features por `|delta_p_fail|` y permite limitar el número de features devueltas por corrida (`top_k`).

Este enfoque se inspira en la lógica de **“what-if”** de LIME/SHAP (perturbar inputs y observar el cambio en la predicción), pero sin necesidad de librerías externas.

### 5.2. Concepto de SHAP

**SHAP (SHapley Additive exPlanations)**:

- Basado en valores de **Shapley** de teoría de juegos.
- Descompone la predicción del modelo en contribuciones de cada feature, garantizando propiedades deseables (aditividad, consistencia).
- Permite:
  - Explicaciones locales (por observación).
  - Importancias globales agregando valores por feature.
- Extensión futura:
  - Integrar `shap` fuera del path de producción (offline) para dashboards de análisis más ricos.

### 5.3. Concepto de LIME

**LIME (Local Interpretable Model-agnostic Explanations)**:

- Entrena modelos simples (e.g. regresión lineal) alrededor de un punto de interés, perturbando el input.
- Aproxima localmente la función compleja del modelo original.
- Logra una explicación intuitiva:
  - “En la vecindad de esta corrida, estas features pesan así”.

El enfoque actual con perturbaciones y baselines va en una dirección similar, pero **sin entrenamiento auxiliar** de modelos locales, lo que lo hace más liviano para usar en la API.

### 5.4. Relación con la API

- `/predict`:
  - Devuelve sólo:
    - `p_fail`
    - `predicted_failed`
- `/explain`:
  - Devuelve:
    - `p_fail`
    - `predicted_failed`
    - `contributions`: lista de `{ feature, delta_p_fail }`, ordenada por importancia.
  - Parámetro `top_k`:
    - `0` → devuelve todas las features.
    - `> 0` → devuelve las `top_k` más influyentes.

---

## 6. Uso práctico

### 6.1. Casos de uso recomendados

- **Priorizar debugging**:
  - Ordenar corridas por `p_fail` descendente para decidir dónde invertir esfuerzo.
- **Alertas tempranas**:
  - En pipelines de CI/CD, disparar una alerta si `p_fail` supera cierto umbral, incluso antes de ver resultados completos.
- **Gestión de deuda técnica**:
  - Monitorear en el tiempo `p_fail` por `squad`, `platform` o `device_id` para detectar hotspots de flakiness.
- **Feedback a squads**:
  - Usar `/explain` para mostrar:
    - qué factores están empujando el riesgo (e.g. cierto device, environment),
    - y priorizar acciones (migrar devices, depurar suites, ajustar data de test).

### 6.2. Limitaciones

- El modelo está limitado por las features disponibles:
  - No ve causas de negocio o de infraestructura que no estén en el CSV.
- La flakiness es un fenómeno ruidoso:
  - No todas las fuentes de inestabilidad son capturables de manera determinista.
- El enfoque de explainability es una aproximación:
  - No sustituye un análisis completo con SHAP/LIME offline cuando se necesitan explicaciones muy finas.

---
