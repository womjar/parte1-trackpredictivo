import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    confusion_matrix,
)
import joblib


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Entrenar un modelo predictivo para p_fail "
            "(probabilidad de que una corrida tenga al menos un escenario fallido)."
        )
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Ruta al CSV con las ejecuciones E2E (e2e_runs_meli_fake.csv).",
    )
    parser.add_argument(
        "--output-model",
        required=True,
        help="Ruta donde se guardará el modelo entrenado (.joblib).",
    )
    parser.add_argument(
        "--metrics-output",
        required=False,
        help="Archivo de texto donde se guardarán métricas (opcional).",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Proporción del dataset para test (default: 0.2).",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random state para reproducibilidad.",
    )
    return parser.parse_args()


def load_data(csv_path: str) -> pd.DataFrame:
    # Intentar parsear timestamp si existe
    parse_dates = []
    preview_cols = pd.read_csv(csv_path, nrows=0).columns
    if "timestamp" in preview_cols:
        parse_dates = ["timestamp"]

    df = pd.read_csv(csv_path, parse_dates=parse_dates)
    df.columns = [c.strip().lower() for c in df.columns]
    return df


def add_target_and_features(df: pd.DataFrame):
    # Verificación de columnas mínimas
    required_cols = {"scenarios_total", "scenarios_failed"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas requeridas: {missing}")

    # Evitar división por cero
    df["scenarios_total"] = df["scenarios_total"].replace(0, np.nan)
    df = df.dropna(subset=["scenarios_total"])

    # fail_rate por corrida
    df["fail_rate"] = df["scenarios_failed"] / df["scenarios_total"]

    # Objetivo binario: corrida fallida si hay al menos un escenario fallido
    df["run_failed"] = (df["scenarios_failed"] > 0).astype(int)

    # Features base
    cat_features = []
    num_features = []

    for col in ["squad", "platform", "environment", "device_id"]:
        if col in df.columns:
            cat_features.append(col)

    for col in ["memory_mb", "scenarios_total"]:
        if col in df.columns:
            num_features.append(col)

    # Features de tiempo, si tenemos timestamp
    if "timestamp" in df.columns and np.issubdtype(
        df["timestamp"].dtype, np.datetime64
    ):
        df["hour"] = df["timestamp"].dt.hour
        df["dayofweek"] = df["timestamp"].dt.dayofweek
        num_features.extend(["hour", "dayofweek"])

    feature_cols = cat_features + num_features

    if not feature_cols:
        raise ValueError(
            "No se encontraron columnas de features para entrenar el modelo."
        )

    X = df[feature_cols].copy()
    y = df["run_failed"].copy()

    return X, y, cat_features, num_features


def build_model_pipeline(cat_features, num_features) -> Pipeline:
    # Preprocesador: OneHot para categóricas, StandardScaler para numéricas
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features),
            ("num", StandardScaler(), num_features),
        ]
    )

    # Modelo: Gradient Boosting (bueno para tabular, da probabilidades)
    model = GradientBoostingClassifier(random_state=42)

    pipeline = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", model),
        ]
    )

    return pipeline


def format_metrics_report(y_test, y_pred, y_proba) -> str:
    lines = []
    lines.append("===== Métricas modelo p_fail =====\n")
    lines.append("Reporte de clasificación (clase 1 = corrida fallida):")
    lines.append(classification_report(y_test, y_pred, digits=4))

    try:
        roc_auc = roc_auc_score(y_test, y_proba)
        lines.append(f"ROC-AUC: {roc_auc:.4f}")
    except Exception as e:
        lines.append(f"ROC-AUC: no se pudo calcular ({e})")

    cm = confusion_matrix(y_test, y_pred)
    lines.append("Matriz de confusión [ [TN FP] [FN TP] ]:")
    lines.append(str(cm))

    lines.append("\nDistribución de la variable objetivo (y_test):")
    unique, counts = np.unique(y_test, return_counts=True)
    dist = dict(zip(unique, counts))
    lines.append(str(dist))

    return "\n".join(lines)


def main():
    args = parse_args()

    input_path = args.input
    output_model_path = Path(args.output_model)
    metrics_output_path = Path(args.metrics_output) if args.metrics_output else None

    print(f"Cargando datos desde {input_path} ...")
    df = load_data(input_path)

    print("Construyendo features y variable objetivo (run_failed)...")
    X, y, cat_features, num_features = add_target_and_features(df)

    print("Columnas categóricas usadas:", cat_features)
    print("Columnas numéricas usadas:", num_features)
    print("Distribución de la variable objetivo (0 = OK, 1 = failed):")
    print(y.value_counts())

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y,
    )

    pipeline = build_model_pipeline(cat_features, num_features)

    print("Entrenando modelo...")
    pipeline.fit(X_train, y_train)

    print("Evaluando en el set de test...")
    y_pred = pipeline.predict(X_test)
    # Probabilidad de clase positiva (corrida fallida)
    if hasattr(pipeline, "predict_proba"):
        y_proba = pipeline.predict_proba(X_test)[:, 1]
    else:
        # Algunos modelos no tienen predict_proba; como fallback usamos decision_function
        if hasattr(pipeline, "decision_function"):
            scores = pipeline.decision_function(X_test)
            # Normalización simple a [0,1] para pseudo-proba
            y_proba = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
        else:
            y_proba = np.zeros_like(y_test, dtype=float)

    metrics_report = format_metrics_report(y_test, y_pred, y_proba)
    print(metrics_report)

    # Guardar modelo
    output_model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, output_model_path)
    print(f"\nModelo guardado en: {output_model_path.resolve()}")

    # Guardar métricas
    if metrics_output_path is not None:
        metrics_output_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_output_path.write_text(metrics_report, encoding="utf-8")
        print(f"Métricas guardadas en: {metrics_output_path.resolve()}")


if __name__ == "__main__":
    main()
