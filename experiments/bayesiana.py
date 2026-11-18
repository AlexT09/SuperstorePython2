"""
experiments/bayesiana.py — Experimento complementario: regresión bayesiana (BayesianRidge) como referencia.

La guía de resultados del curso incluye la regresión bayesiana entre los modelos que deben compararse,
y el protocolo registrado (protocolo-v1) no la tenía. Se añade DESPUÉS del registro, como desviación
declarada en el artículo, con exactamente el mismo procedimiento de ejecutar.py: los mismos 24 orígenes,
la misma validación cruzada temporal interna, el mismo Pipeline (one-hot + estandarización ajustados en
cada pliegue), el mismo presupuesto (rejilla completa o 16 evaluaciones) y la misma semilla.

No modifica los resultados registrados: escribe archivos aparte, que lee experiments/paper_tablas.py.
    pred_bayesiana_<conjunto>.csv     predicciones (mismas filas y columnas META que pred_<conjunto>.csv)
    tabla_maestra_bayesiana.csv       hiperparámetros, tiempos y semilla por origen y optimizador

Ejecutar desde la raíz:  python experiments/bayesiana.py
"""
import json
import time
import warnings

import pandas as pd
from joblib import Parallel, delayed
from sklearn.linear_model import BayesianRidge

import config as C
import ejecutar as EJ
from model.features import FEATURES, TARGET, a_ventas
from src.optimizacion import optimizar

NOMBRE = "Regresión bayesiana"
# Previas Gamma de la precisión del ruido (alpha_1, alpha_2) y de los pesos (lambda_1, lambda_2)
REJILLA = {"alpha_1": [1e-6, 1e-4, 1e-2], "lambda_1": [1e-6, 1e-4, 1e-2]}
ESPACIO = {"alpha_1": ("log", 1e-7, 1e-1), "alpha_2": ("log", 1e-7, 1e-1),
           "lambda_1": ("log", 1e-7, 1e-1), "lambda_2": ("log", 1e-7, 1e-1)}


def un_origen(conjunto, datos, origen):
    warnings.filterwarnings("ignore")
    tr, te = datos[datos.Mes < origen], datos[datos.Mes == origen]
    cv = EJ.cv_interna(tr)
    pred = te[["Category", "Region", "Mes", "Sales", "prom_12m"]].copy()
    maestra = []
    for metodo in C.OPTIMIZADORES[conjunto]:
        pipe = EJ.con_pipeline(BayesianRidge())
        r = optimizar(metodo, pipe, EJ.prefijo(REJILLA), EJ.prefijo(ESPACIO), tr[FEATURES], tr[TARGET], cv,
                      n=C.PRESUPUESTO, semilla=C.SEMILLA)
        final = pipe.set_params(**r["params"])
        t0 = time.perf_counter(); final.fit(tr[FEATURES], tr[TARGET]); t_fit = time.perf_counter() - t0
        t0 = time.perf_counter(); p = final.predict(te[FEATURES]); t_inf = time.perf_counter() - t0
        pred[f"{NOMBRE}|{metodo}"] = a_ventas(p, te)
        maestra.append({"conjunto": conjunto, "origen": pd.Timestamp(origen), "modelo": NOMBRE, "optimizador": metodo,
                        "hiperparametros": json.dumps({k.replace("modelo__", ""): v for k, v in r["params"].items()}, default=str),
                        "puntaje_interno": r["puntaje"], "evaluaciones": len(r["historial"]),
                        "busqueda_s": r["segundos"], "ajuste_ms": t_fit * 1e3, "inferencia_ms": t_inf * 1e3,
                        "historial": json.dumps([round(v, 6) for v in r["historial"]]), "semilla": C.SEMILLA})
    return pred, maestra


def main():
    filas = []
    for conjunto in C.CONJUNTOS:
        t0 = time.perf_counter()
        _, datos, origenes = EJ.cargar(conjunto)
        salida = Parallel(n_jobs=-1)(delayed(un_origen)(conjunto, datos, o) for o in origenes)
        pd.concat([p for p, _ in salida], ignore_index=True).round(4).to_csv(C.RES / f"pred_bayesiana_{conjunto}.csv", index=False)
        filas += [f for _, m in salida for f in m]
        print(f"{conjunto}: {len(origenes)} orígenes en {time.perf_counter() - t0:.0f} s", flush=True)
    pd.DataFrame(filas).to_csv(C.RES / "tabla_maestra_bayesiana.csv", index=False)


if __name__ == "__main__":
    main()
