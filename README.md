# E2E Flakiness Analyzer & Predictor

Proyecto para analizar y predecir **flakiness** en ejecuciones end-to-end (E2E), basado en un dataset histórico (`e2e_runs_meli_fake.csv`) y empaquetado completamente con **Docker** y **docker-compose**.

Incluye:

1. **EDA (Exploratory Data Analysis)** de flakiness por squad, device, plataforma, etc.
2. **Modelo predictivo** de probabilidad de fallo por corrida (`p_fail`).
3. **Microservicio FastAPI** que expone:
   - `/predict`: predicción de `p_fail`.
   - `/explain`: explicación tipo feature-importance por corrida.
   - `/schema`: ejemplos de payload para integración.


---



## 1. Estructura del proyecto

```text
.
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
├── README.md
├── data/
│   └── e2e_runs_meli_fake.csv        # dataset de ejecuciones E2E
├── models/
│   └── flakiness_model.joblib        # se genera tras el entrenamiento
│   └── flakiness_model_metrics.txt   # métricas del modelo
├── reports/
│   ├── summary.txt                   # resumen del EDA
│   ├── by_iteration.csv
│   ├── by_squad.csv
│   ├── by_platform.csv
│   ├── by_device.csv
│   ├── flaky_full_combo.csv
│   ├── flaky_squad_device.csv
│   └── plots/
│       ├── boxplot_fail_rate_by_*.png
│       └── time_series_fail_rate_daily.png
└── src/
    ├── eda_flakiness.py              # actividad 1: EDA
    ├── train_flakiness_model.py      # actividad 2: entrenamiento modelo
    └── flakiness_api.py              # actividad 3: microservicio FastAPI

```


Algunas carpetas (models/, reports/) se crean automáticamente la primera vez que corres los servicios.


---


## 2. Dataset esperado

El archivo principal es `data/e2e_runs_meli_fake.csv`.
Las columnas mínimas que el pipeline espera (y/o sabe aprovechar) son:

   * `timestamp`: fecha y hora de la ejecución (parseable como datetime).

   * `squad`: squad dueño de la suite.

   * `platform`: plataforma (ej. android, ios, web).

   * `environment`: ambiente de ejecución (ej. test_app, release_app).

   * `device_id`: identificador del dispositivo / emulador.

   * `memory_mb`: memoria disponible del device (en MB).

   * `scenarios_total`: número de escenarios ejecutados en esa corrida.

   * `scenarios_failed`: número de escenarios fallidos en esa corrida.

A partir de esto, los scripts calculan:

   * `fail_rate = scenarios_failed / scenarios_total`
   * `run_failed = 1 si scenarios_failed > 0, si no 0`
   * Features de tiempo:
      * `hour (hora del día)`
      * `dayofweek (día de la semana: 0=Lunes, 6=Domingo)`

---


## 3. Requisitos

   * Docker (20+ recomendado)
   * docker-compose (v2+)

No necesitas instalar Python ni dependencias localmente: todo corre dentro de los contenedores.

---

## 4. Instalación y primeros pasos

 1. Clonar el repo
 2. Construir la imagen Docker

    - **docker-compose build**



  Esto instalará todas las dependencias definidas en `requirements.txt` dentro de la imagen:



 * `pandas`, `numpy`

 * `matplotlib`, `seaborn`

 * `scikit-learn`, `joblib`

 * `fastapi`, `uvicorn`


---


## 5. Servicios en docker-compose

El `docker-compose.yml` define tres servicios principales:

   * `flakiness-eda`: corre el EDA sobre el CSV.
   * `flakiness-train`: entrena el modelo predictivo y lo guarda en /models.
   * `flakiness-api`: levanta el microservicio FastAPI para predicción y explicación.

Puedes ver la definición completa en docker-compose.yml.

---

## 6. Actividad 1 – EDA de flakiness

Script: src/eda_flakiness.py
Servicio: flakiness-eda

**¿Qué hace?**

 * Lee `data/e2e_runs_meli_fake.csv`.

 * Normaliza columnas a `lower_snake_case`.

 * Calcula:
   * `fail_rate` por corrida.
   * agregados por:
      * `squad`
      * `device_id`
      * `platform + environment`
      * `iteration` (si existe)

 * Detecta potenciales **grupos flaky** basados en:
   * `min_runs` por grupo (ej. >= 5)
   * `fail_rate_std` alto (alta variabilidad).

* Genera:
   * CSV de agregados (por squad, device, etc.).
   * CSV de grupos marcados como flaky.
   * Gráficas:
      * boxplots de `fail_rate` por squad/device/platform/env.
      * serie temporal de `fail_rate` diario.

* `reports/summary.txt` con:
   * resumen del dataset,
   * estadísticas de `fail_rate`,
   * top 10 grupos más flaky.


**Cómo ejecutarlo**

   docker-compose run --rm flakiness-eda

Outputs principales:
   * `reports/summary.txt`
   * `reports/by_squad.csv`
   * `reports/by_device.csv`
   * `reports/by_platform_env.csv`
   * `reports/flaky_*.csv`
   * `reports/plots/*.png`


---


## 7. Actividad 2 – Modelo predictivo de p_fail

Script: `src/train_flakiness_model.py`
Servicio: `flakiness-train`

**Objetivo del modelo**

Predecir:

 * `run_failed = 1` si la corrida tiene al menos un escenario fallido.

 * Se obtiene `p_fail = P(run_failed = 1 | features)` con un modelo de clasificación.

Features utilizadas

 * Categóricas:
   * `squad`
   * `platform`
   * `environment`
   * `device_id`
 * Numéricas:
   * `scenarios_total`
   * `memory_mb`
   * `hour` (derivada de `timestamp`)

dayofweek (derivada de timestamp)

**Target**

 * `run_failed = (scenarios_failed > 0).astype(int)`

**Modelo y pipeline**

Se construye un `Pipeline` de scikit-learn:

1. **Preprocesamiento (`ColumnTransformer`)**
     * OneHotEncoder para features categóricas.
     * StandardScaler para numéricas.

2. **Modelo**
     * GradientBoostingClassifier (bueno para datos tabulares y devuelve probabilidades).


## Cómo entrenarlo ##

   `docker-compose run --rm flakiness-train`

Esto:

   * Lee data/e2e_runs_meli_fake.csv.
   * Separa train/test (stratify por target).
   * Entrena el modelo.
   * Evalúa y guarda métricas.
   * Guarda el modelo en:
      * models/flakiness_model.joblib
      * models/flakiness_model_metrics.txt

El archivo de métricas incluye:
   * classification_report (precision/recall/f1).
   * ROC-AUC.
   * confusion_matrix.
   * distribución de la variable objetivo en test.


---


## 8. API de predicción y explicabilidad

Script: src/flakiness_api.py
Servicio: flakiness-api

La API usa el modelo entrenado en `models/flakiness_model.joblib`.
Asegúrate de haber corrido antes `flakiness-train`.

**Levantar la API**

   `docker compose up flakiness-api`

La API quedará disponible en:

   * http://localhost:8000

FastAPI genera además documentación interactiva:

   * Swagger UI: http://localhost:8000/docs

   * ReDoc: http://localhost:8000/redoc


---


## 9. Endpoints disponibles
   
   **9.1.** `GET` /health
  
      Simple healthcheck.

   **9.2.** `POST` /predict

      Predice p_fail para una o varias corridas.  

   **9.3.** POST /explain

      Explica p_fail por corrida, devolviendo importancia local por feature.

      La técnica es una aproximación ligera tipo “sensibilidad”:

       * Para cada corrida:
         * Se calcula p_fail original (p0).
         * Para cada feature:
            * Se reemplaza su valor por un baseline:
               * Categóricas: moda (más frecuente) en el batch.
               * Numéricas: mediana en el batch.
            * Se vuelve a predecir (p_new).
            * delta_p_fail = p0 - p_new:
               * > 0: esa feature aumenta el riesgo frente al baseline.
               * < 0: esa feature reduce el riesgo frente al baseline.

      Además, se puede limitar cuántas features se devuelven por corrida con top_k.

      Body de ejemplo:

      {
         "threshold": 0.5,
         "top_k": 3,
         "runs": [
            {
               "timestamp": "2025-11-18T10:30:00Z",
               "squad": "checkout",
               "platform": "android",
               "environment": "test_app",
               "device_id": "pixel_6_emulator",
               "memory_mb": 4096,
               "scenarios_total": 120
            }
         ]
      }


      Respuesta de ejemplo:

      {
         "explanations": [
            {
               "p_fail": 0.65,
               "predicted_failed": true,
               "contributions": [
               {"feature": "environment", "delta_p_fail": 0.18},
               {"feature": "platform", "delta_p_fail": 0.12},
               {"feature": "squad", "delta_p_fail": 0.09}
               ]
            }
         ]
      }

      * Si top_k = 0, se devuelven todas las features ordenadas por |delta_p_fail|.

      * Si top_k > 0, se devuelven solo las top_k más influyentes.


   **9.4.** GET /schema

   Devuelve ejemplos de payload para /predict y /explain, para facilitar la integración con otros equipos.

   Respuesta de ejemplo:

      {
      "predict": {
         "method": "POST",
         "path": "/predict",
         "body": {
            "threshold": 0.5,
            "runs": [
            {
               "timestamp": "2025-11-18T10:30:00Z",
               "squad": "checkout",
               "platform": "android",
               "environment": "test_app",
               "device_id": "pixel_6_emulator",
               "memory_mb": 4096,
               "scenarios_total": 120
            }
            ]
         }
      },
      "explain": {
         "method": "POST",
         "path": "/explain",
         "body": {
            "threshold": 0.5,
            "top_k": 3,
            "runs": [
            {
               "timestamp": "2025-11-18T10:30:00Z",
               "squad": "checkout",
               "platform": "android",
               "environment": "test_app",
               "device_id": "pixel_6_emulator",
               "memory_mb": 4096,
               "scenarios_total": 120
            }
            ]
         }
      }
      }


     Esto permite copiar/pegar y modificar solo los valores.

---


## 10. Flujo end-to-end

 **1.Actualizar dataset**
    * Dejar el CSV actualizado en `data/e2e_runs_meli_fake.csv`.

 **2. Construir las imagenes Docker**
    * docker-compose build    

 **3. Correr EDA**
    * docker-compose run --rm flakiness-eda
    * Revisar `reports/summary.txt`, CSVs y plots para entender flakiness por squad/device/etc.

 **4.Entrenar/actualizar modelo**
    * docker-compose run --rm flakiness-train
    * Revisar `models/flakiness_model_metrics.txt` para validar calidad del modelo.

 **5. Levantar la API**
    * docker-compose up flakiness-api

 **6. Integración con pipelines**
    * Desde tu pipeline de E2E: llamar a /predict para obtener p_fail por corrida.