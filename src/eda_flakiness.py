import argparse
import os
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def parse_args():
    parser = argparse.ArgumentParser(
        description="EDA para flakiness de ejecuciones E2E."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Ruta al CSV con las ejecuciones E2E (e2e_runs_meli_fake.csv).",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directorio donde se guardarán reportes y gráficas.",
    )
    return parser.parse_args()


def load_data(csv_path: str) -> pd.DataFrame:
    parse_dates = []
    if "timestamp" in pd.read_csv(csv_path, nrows=0).columns:
        parse_dates = ["timestamp"]

    df = pd.read_csv(csv_path, parse_dates=parse_dates)

    # Normalizar nombres de columnas a lower_snake_case por si acaso
    df.columns = [c.strip().lower() for c in df.columns]

    return df


def add_flakiness_metrics(df: pd.DataFrame) -> pd.DataFrame:
    required_cols = ["scenarios_total", "scenarios_failed"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Falta la columna requerida: {col}")

    df["scenarios_total"] = df["scenarios_total"].replace(0, np.nan)
    df["fail_rate"] = df["scenarios_failed"] / df["scenarios_total"]

    # Métrica simple: passed
    df["scenarios_passed"] = df["scenarios_total"] - df["scenarios_failed"]

    return df


def describe_basic(df: pd.DataFrame) -> str:
    lines = []
    lines.append("========== RESUMEN BÁSICO DEL DATASET ==========\n")
    lines.append(f"Filas: {len(df)}")
    lines.append(f"Columnas: {len(df.columns)}")
    lines.append(f"Columnas: {list(df.columns)}\n")

    if "timestamp" in df.columns:
        lines.append("Rango de fechas:")
        lines.append(f"  min: {df['timestamp'].min()}")
        lines.append(f"  max: {df['timestamp'].max()}\n")

    for col in ["squad", "platform", "environment", "device_id"]:
        if col in df.columns:
            uniq = df[col].nunique()
            top = df[col].value_counts().head(5)
            lines.append(f"=== {col.upper()} ===")
            lines.append(f"Valores únicos: {uniq}")
            lines.append("Top 5 más frecuentes:")
            lines.append(str(top))
            lines.append("")

    lines.append("=== Estadísticas de fail_rate ===")
    lines.append(str(df["fail_rate"].describe()))
    lines.append("")

    return "\n".join(lines)


def group_stats(df: pd.DataFrame, group_cols, metric="fail_rate"):
    grouped = (
        df.groupby(group_cols)[metric]
        .agg(["mean", "std", "count"])
        .rename(
            columns={"mean": "fail_rate_mean", "std": "fail_rate_std", "count": "runs"}
        )
        .sort_values("fail_rate_mean", ascending=False)
    )
    return grouped


def detect_flaky_groups(
    df: pd.DataFrame,
    group_cols,
    metric="fail_rate",
    min_runs: int = 5,
    std_threshold: float = 0.25,
):
    """
    Flaky = grupos con suficiente número de ejecuciones y alta variabilidad en fail_rate.

    - min_runs: mínimo de ejecuciones para considerar el grupo.
    - std_threshold: desviación estándar mínima de fail_rate para marcarlo como flaky.
    """
    grp = (
        df.groupby(group_cols)[metric]
        .agg(["mean", "std", "count"])
        .rename(
            columns={"mean": "fail_rate_mean", "std": "fail_rate_std", "count": "runs"}
        )
    )
    flaky = grp[(grp["runs"] >= min_runs) & (grp["fail_rate_std"] >= std_threshold)]
    flaky = flaky.sort_values("fail_rate_std", ascending=False)
    return flaky


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def plot_boxplot_fail_rate(df: pd.DataFrame, by: str, output_path: Path):
    if by not in df.columns:
        return

    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x=by, y="fail_rate")
    plt.title(f"Distribución de fail_rate por {by}")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_time_series(df: pd.DataFrame, output_path: Path):
    if "timestamp" not in df.columns:
        return

    # Resampleo diario de fail_rate promedio
    ts = df.set_index("timestamp").sort_index().resample("D")["fail_rate"].mean()

    plt.figure(figsize=(12, 5))
    plt.plot(ts.index, ts.values)
    plt.title("Evolución diaria del fail_rate promedio")
    plt.xlabel("Fecha")
    plt.ylabel("fail_rate medio")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def main():
    args = parse_args()

    input_path = args.input
    output_dir = Path(args.output_dir)
    plots_dir = output_dir / "plots"

    ensure_dir(output_dir)
    ensure_dir(plots_dir)

    # 1) Cargar datos
    df = load_data(input_path)

    # 2) Añadir métricas de flakiness
    df = add_flakiness_metrics(df)

    # 3) Resumen básico
    summary_text = describe_basic(df)

    # 4) Agregaciones por squad, device, platform/env, iteration (si existe)
    results = {}

    if "squad" in df.columns:
        results["by_squad"] = group_stats(df, ["squad"])
    if "device_id" in df.columns:
        results["by_device"] = group_stats(df, ["device_id"])
    if {"platform", "environment"}.issubset(df.columns):
        results["by_platform_env"] = group_stats(df, ["platform", "environment"])
    if "iteration" in df.columns:
        results["by_iteration"] = group_stats(df, ["iteration"])

    # 5) Detección de grupos flaky
    flaky_results = {}

    if {"squad", "device_id"}.issubset(df.columns):
        flaky_results["flaky_squad_device"] = detect_flaky_groups(
            df,
            ["squad", "device_id"],
            min_runs=5,
            std_threshold=0.25,
        )

    if {"squad", "device_id", "platform", "environment"}.issubset(df.columns):
        flaky_results["flaky_full_combo"] = detect_flaky_groups(
            df,
            ["squad", "device_id", "platform", "environment"],
            min_runs=5,
            std_threshold=0.25,
        )

    # 6) Guardar resultados en CSV
    for name, table in results.items():
        table.to_csv(output_dir / f"{name}.csv")

    for name, table in flaky_results.items():
        table.to_csv(output_dir / f"{name}.csv")

    # 7) Plots
    plot_boxplot_fail_rate(df, "squad", plots_dir / "boxplot_fail_rate_by_squad.png")
    plot_boxplot_fail_rate(
        df, "device_id", plots_dir / "boxplot_fail_rate_by_device.png"
    )
    if "platform" in df.columns:
        plot_boxplot_fail_rate(
            df, "platform", plots_dir / "boxplot_fail_rate_by_platform.png"
        )
    if "environment" in df.columns:
        plot_boxplot_fail_rate(
            df, "environment", plots_dir / "boxplot_fail_rate_by_env.png"
        )

    plot_time_series(df, plots_dir / "time_series_fail_rate_daily.png")

    # 8) Guardar resumen en summary.txt (incluyendo top flaky)
    summary_lines = [
        summary_text,
        "\n========== GRUPOS MÁS FLAKY (TOP 10) ==========\n",
    ]

    for name, table in flaky_results.items():
        summary_lines.append(f"--- {name} ---")
        summary_lines.append(str(table.head(10)))
        summary_lines.append("")

    (output_dir / "summary.txt").write_text("\n".join(summary_lines), encoding="utf-8")

    print("EDA de flakiness completado.")
    print(f"Reportes guardados en: {output_dir.resolve()}")
    print(f"Gráficas guardadas en: {plots_dir.resolve()}")


if __name__ == "__main__":
    main()
