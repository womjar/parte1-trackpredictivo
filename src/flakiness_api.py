from pathlib import Path
from typing import List
from datetime import datetime

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL_PATH = Path("models/flakiness_model.joblib")

app = FastAPI(
    title="Flakiness Prediction API",
    description="Microservicio para predecir y explicar p_fail de ejecuciones E2E.",
    version="1.3.0",
)

model = None  # pipeline de scikit-learn cargado en startup

# Features que usamos en entrenamiento / inferencia
CAT_FEATURES = ["squad", "platform", "environment", "device_id"]
NUM_FEATURES = ["memory_mb", "scenarios_total", "hour", "dayofweek"]
ALL_FEATURES = CAT_FEATURES + NUM_FEATURES


# ========= Esquemas de entrada/salida =========


class RunFeatures(BaseModel):
    timestamp: datetime = Field(..., description="Fecha/hora de la ejecución")
    squad: str = Field(..., description="Squad dueño de la suite")
    platform: str = Field(..., description="Plataforma (android, ios, web, etc)")
    environment: str = Field(..., description="Ambiente (test_app / release_app)")
    device_id: str = Field(..., description="Identificador del dispositivo")
    memory_mb: float = Field(..., description="Memoria del dispositivo en MB")
    scenarios_total: int = Field(
        ..., description="Número total de escenarios ejecutados"
    )


class Prediction(BaseModel):
    p_fail: float = Field(
        ..., description="Probabilidad de corrida fallida (≥1 escenario fallido)"
    )
    predicted_failed: bool = Field(..., description="True si p_fail >= threshold")


class BatchPredictionRequest(BaseModel):
    runs: List[RunFeatures]
    threshold: float = Field(
        0.5,
        ge=0.0,
        le=1.0,
        description="Umbral para marcar una corrida como fallida (default: 0.5)",
    )


class BatchPredictionResponse(BaseModel):
    predictions: List[Prediction]


class FeatureContribution(BaseModel):
    feature: str = Field(..., description="Nombre de la feature original")
    delta_p_fail: float = Field(
        ...,
        description=(
            "Cambio en p_fail al reemplazar la feature por un valor baseline. "
            "Positivo: esa feature aumenta el riesgo de fallo frente al baseline."
        ),
    )


class Explanation(BaseModel):
    p_fail: float = Field(..., description="Probabilidad de corrida fallida")
    predicted_failed: bool = Field(
        ..., description="Predicción binaria según threshold"
    )
    contributions: List[FeatureContribution] = Field(
        ..., description="Importancia local por feature para esta corrida"
    )


class BatchExplanationRequest(BaseModel):
    runs: List[RunFeatures]
    threshold: float = Field(
        0.5,
        ge=0.0,
        le=1.0,
        description="Umbral para marcar corrida como fallida (default: 0.5)",
    )
    top_k: int = Field(
        0,
        ge=0,
        description=(
            "Número máximo de features a devolver por corrida. "
            "Si es 0, se devuelven todas las features ordenadas por |delta_p_fail|."
        ),
    )


class BatchExplanationResponse(BaseModel):
    explanations: List[Explanation]


# ========= Hooks de ciclo de vida =========


@app.on_event("startup")
def load_model():
    global model
    if not MODEL_PATH.exists():
        raise RuntimeError(
            f"No se encontró el archivo de modelo en {MODEL_PATH}. "
            "Asegúrate de haber entrenado con el servicio flakiness-train."
        )
    model = joblib.load(MODEL_PATH)
    print(f"Modelo cargado desde {MODEL_PATH.resolve()}")


# ========= Helpers internos =========


def build_features_dataframe(runs: List[RunFeatures]) -> pd.DataFrame:
    """
    Construye el DataFrame de features X a partir de la lista de runs,
    usando exactamente las columnas que el modelo espera.
    """
    records = []
    for r in runs:
        hour = r.timestamp.hour
        dayofweek = r.timestamp.weekday()  # 0=Lunes, 6=Domingo
        records.append(
            {
                "squad": r.squad,
                "platform": r.platform,
                "environment": r.environment,
                "device_id": r.device_id,
                "memory_mb": r.memory_mb,
                "scenarios_total": r.scenarios_total,
                "hour": hour,
                "dayofweek": dayofweek,
            }
        )
    X = pd.DataFrame.from_records(records)
    # Asegurarnos de que las columnas estén en el orden esperado
    X = X[ALL_FEATURES]
    return X


def predict_p_fail(X: pd.DataFrame):
    """
    Devuelve p_fail para cada fila de X usando el pipeline cargado.
    """
    if not hasattr(model, "predict_proba"):
        raise RuntimeError("El modelo no soporta predict_proba.")
    proba = model.predict_proba(X)[:, 1]
    return proba


# ========= Endpoints =========


@app.get("/health")
def health():
    if model is None:
        return {"status": "error", "detail": "Modelo no cargado"}
    return {"status": "ok"}


@app.post("/predict", response_model=BatchPredictionResponse)
def predict(batch: BatchPredictionRequest):
    if model is None:
        raise HTTPException(status_code=500, detail="Modelo no cargado")
    if not batch.runs:
        raise HTTPException(
            status_code=400, detail="La lista de 'runs' no puede estar vacía"
        )

    X = build_features_dataframe(batch.runs)
    proba = predict_p_fail(X)
    preds_bool = proba >= batch.threshold

    predictions = [
        Prediction(p_fail=float(p), predicted_failed=bool(flag))
        for p, flag in zip(proba, preds_bool)
    ]
    return BatchPredictionResponse(predictions=predictions)


@app.post("/explain", response_model=BatchExplanationResponse)
def explain(batch: BatchExplanationRequest):
    """
    Explica p_fail por corrida usando una aproximación de sensibilidad:
    para cada feature, medimos cuánto cambia p_fail si la reemplazamos
    por un valor baseline (mediana/moda del batch).

    top_k permite limitar cuántas features se devuelven por corrida:
    - top_k = 0  -> todas las features ordenadas por |delta_p_fail|
    - top_k > 0  -> solo las top_k features más influyentes
    """
    if model is None:
        raise HTTPException(status_code=500, detail="Modelo no cargado")
    if not batch.runs:
        raise HTTPException(
            status_code=400, detail="La lista de 'runs' no puede estar vacía"
        )

    # 1) Construimos X con todas las features
    X = build_features_dataframe(batch.runs)

    # 2) Calculamos baseline por feature (a partir de las corridas del request)
    baseline = {}

    # Categóricas: moda
    for f in CAT_FEATURES:
        if f in X.columns and not X[f].dropna().empty:
            baseline[f] = X[f].mode(dropna=True).iloc[0]
        else:
            baseline[f] = None

    # Numéricas: mediana
    for f in NUM_FEATURES:
        if f in X.columns and not X[f].dropna().empty:
            baseline[f] = float(X[f].median())
        else:
            baseline[f] = 0.0  # fallback neutro

    # 3) Predicción base p_fail por corrida
    base_proba = predict_p_fail(X)

    explanations: List[Explanation] = []
    top_k = batch.top_k

    # 4) Para cada corrida, hacemos perturbación de features
    for i in range(len(batch.runs)):
        X_row = X.iloc[[i]].copy()
        p0 = float(base_proba[i])
        predicted_failed = p0 >= batch.threshold

        contributions: List[FeatureContribution] = []

        for f in ALL_FEATURES:
            # Si no tenemos baseline para esta feature categórica, la saltamos
            if f in CAT_FEATURES and baseline[f] is None:
                continue

            X_pert = X_row.copy()
            X_pert[f] = baseline[f]
            p_new = float(predict_p_fail(X_pert)[0])
            delta = p0 - p_new  # impacto de esa feature frente al baseline

            contributions.append(FeatureContribution(feature=f, delta_p_fail=delta))

        # Ordenar por importancia absoluta desc
        contributions_sorted = sorted(
            contributions,
            key=lambda c: abs(c.delta_p_fail),
            reverse=True,
        )

        # Aplicar top_k si corresponde
        if top_k > 0:
            contributions_sorted = contributions_sorted[:top_k]

        explanations.append(
            Explanation(
                p_fail=p0,
                predicted_failed=predicted_failed,
                contributions=contributions_sorted,
            )
        )

    return BatchExplanationResponse(explanations=explanations)


@app.get("/schema")
def schema():
    """
    Devuelve ejemplos de payload para /predict y /explain
    para facilitar la integración de otros equipos.
    """
    example_run = {
        "timestamp": "2025-11-18T10:30:00Z",
        "squad": "checkout",
        "platform": "android",
        "environment": "test_app",
        "device_id": "pixel_6_emulator",
        "memory_mb": 4096,
        "scenarios_total": 120,
    }

    predict_example = {
        "method": "POST",
        "path": "/predict",
        "body": {
            "threshold": 0.5,
            "runs": [example_run],
        },
    }

    explain_example = {
        "method": "POST",
        "path": "/explain",
        "body": {
            "threshold": 0.5,
            "top_k": 3,
            "runs": [example_run],
        },
    }

    return {
        "predict": predict_example,
        "explain": explain_example,
    }
