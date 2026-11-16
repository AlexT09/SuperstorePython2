"""
src/data.py — Carga (una sola vez) de los datos, los resultados de los experimentos y los modelos guardados.

El dashboard no entrena ni optimiza nada: solo lee estos archivos.
    data/                       datos procesados
    experiments/resultados/     tablas de la validación anidada y de las pruebas (experiments/analizar.py)
    artifacts/                  modelos serializados y metadata.json (experiments/artefactos.py)
"""
import json
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from model.features import FEATURES, TARGET, agregar_variables

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "experiments" / "resultados"
ART = ROOT / "artifacts"
ORIGINAL = "Perfil jerárquico (original)"
META = ["Category", "Region", "Mes", "Sales", "prom_12m"]


def listo() -> bool:
    """True si ya existen los resultados y los modelos guardados."""
    return (ART / "metadata.json").exists() and (RES / "lideres.csv").exists()


@lru_cache(maxsize=None)
def resultado(nombre: str):
    """Lee un archivo de experiments/resultados/ (CSV -> DataFrame, JSON -> dict). None si no existe."""
    ruta = RES / nombre
    if not ruta.exists():
        return None
    if ruta.suffix == ".json":
        return json.loads(ruta.read_text(encoding="utf-8"))
    df = pd.read_csv(ruta)
    for col in ("Mes", "origen"):
        if col in df:
            df[col] = pd.to_datetime(df[col])
    return df


@lru_cache(maxsize=None)
def artefacto(nombre: str):
    """Lee un archivo de artifacts/ (CSV, JSON o modelo .joblib)."""
    ruta = ART / nombre
    if not ruta.exists():
        return None
    if ruta.suffix == ".json":
        return json.loads(ruta.read_text(encoding="utf-8"))
    if ruta.suffix == ".joblib":
        return joblib.load(ruta)
    df = pd.read_csv(ruta)
    if "Mes" in df:
        df["Mes"] = pd.to_datetime(df["Mes"])
    return df


def metadata() -> dict:
    return artefacto("metadata.json")


def modelo(nombre: str):
    """Modelo entrenado (se carga la primera vez que se pide y queda en memoria)."""
    return artefacto(metadata()["modelos"][nombre]["archivo"])


def entrada(nombre: str, tabla: pd.DataFrame) -> pd.DataFrame:
    """Columnas que recibe cada modelo: el original lee la tabla completa; los demás, sus variables."""
    return tabla if nombre == ORIGINAL else tabla[FEATURES]


@lru_cache(maxsize=1)
def panel() -> pd.DataFrame:
    return pd.read_csv(ROOT / "data" / "ventas_mensuales.csv", parse_dates=["Mes"])


@lru_cache(maxsize=1)
def historia() -> pd.DataFrame:
    """Variables del modelo para cada serie y mes, incluido el primer mes futuro (sin ventas reales)."""
    m = panel()[["Category", "Region", "Mes", "Sales"]]
    futuro = m[["Category", "Region"]].drop_duplicates().assign(Mes=m.Mes.max() + pd.DateOffset(months=1), Sales=np.nan)
    h = agregar_variables(pd.concat([m, futuro], ignore_index=True))
    return h.dropna(subset=["prom_12m", "lag_12", "lag_1", "prom_3m"]).reset_index(drop=True)


@lru_cache(maxsize=1)
def fondo() -> pd.DataFrame:
    """Muestra de referencia para los valores de Shapley (30 filas con historia real)."""
    h = historia().dropna(subset=[TARGET])
    return h.sample(30, random_state=metadata()["semilla"]).reset_index(drop=True)


@lru_cache(maxsize=1)
def predicciones() -> pd.DataFrame:
    """Pronósticos del bucle externo (origen móvil) de todas las combinaciones, en Sample Superstore."""
    return resultado("pred_superstore.csv")


def columna(nombre: str) -> str:
    """Columna de predicciones() de la mejor combinación de un modelo."""
    return f"{nombre}|{metadata()['modelos'][nombre]['optimizador']}"
