"""
experiments/paper_tablas.py — Tablas y cifras del artículo, con la regresión bayesiana incluida.

Une las predicciones registradas (pred_<conjunto>.csv, de ejecutar.py) con las del experimento
complementario (pred_bayesiana_<conjunto>.csv, de bayesiana.py) y repite el análisis de analizar.py con
11 referencias en lugar de 10: la corrección de Holm usa m = 11, el Model Confidence Set incluye las
cuatro combinaciones de la regresión bayesiana y la prueba entre conjuntos compara 12 modelos.

Añade lo que pide la guía de resultados y no estaba en el análisis registrado (análisis exploratorio,
declarado en el artículo): R², los estadísticos de cada prueba, la prueba t bayesiana correlacionada con
región de equivalencia práctica (ROPE = ±1 % de la venta media por serie y mes) y la tabla de costo.

Salidas en experiments/resultados/:  paper_resultados.csv · paper_lideres.csv · paper_pruebas.csv ·
paper_conjuntos.csv · paper_costo.csv · paper_numeros.json
Ejecutar desde la raíz, después de analizar.py y bayesiana.py:  python experiments/paper_tablas.py
"""
import io
import json
import warnings

import joblib
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import clone

import analizar as A
import config as C
import ejecutar as EJ
from bayesiana import NOMBRE as BAYES
from model.features import FEATURES, TARGET
from src import estadistica as E

BASE = C.BASE + [BAYES]
ORIG = A.ORIG
CLAVE = ["Category", "Region", "Mes"]
ROPE_REL = 0.01                       # ±1 % de la venta media por serie y mes


def cargar(conjunto):
    p = pd.read_csv(C.RES / f"pred_{conjunto}.csv", parse_dates=["Mes"])
    b = pd.read_csv(C.RES / f"pred_bayesiana_{conjunto}.csv", parse_dates=["Mes"])
    b = b[CLAVE + [c for c in b.columns if c.startswith(BAYES)]]
    out = p.merge(b, on=CLAVE, how="left", validate="one_to_one")
    assert out[[c for c in b.columns if c.startswith(BAYES)]].notna().all().all(), conjunto
    return out


def r2(p, col):
    e = p.Sales - p[col]
    return float(1 - (e ** 2).sum() / ((p.Sales - p.Sales.mean()) ** 2).sum())


def n_parametros(est):
    """Cantidad de números que el modelo ajustado guarda para predecir (no cuenta hiperparámetros)."""
    nombre = type(est).__name__
    if nombre == "PerfilEstacionalJerarquico":
        return int(np.size(est.perfil_) + len(est.pesos_) + 1)           # 12 meses + un peso por serie + w0
    if nombre in ("Ridge", "Lasso"):
        return int(est.coef_.size + 1)
    if nombre == "BayesianRidge":
        return int(est.coef_.size + 3)                                   # coeficientes, intercepto, alpha_, lambda_
    if nombre == "KNeighborsRegressor":
        return int(est.n_samples_fit_ * (est.n_features_in_ + 1))        # guarda el entrenamiento completo
    if nombre == "DecisionTreeRegressor":
        return int(est.tree_.node_count)
    if nombre == "RandomForestRegressor":
        return int(sum(t.tree_.node_count for t in est.estimators_))
    if nombre == "XGBRegressor":
        return int(len(est.get_booster().trees_to_dataframe()))
    if nombre == "SVR":
        return int(est.support_vectors_.size + est.dual_coef_.size + 1)
    if nombre == "HistGradientBoostingRegressor":
        return int(sum(len(pr[0].nodes) for pr in est._predictors))
    raise ValueError(nombre)


def tamano_kb(obj):
    buf = io.BytesIO()
    joblib.dump(obj, buf)
    return buf.tell() / 1024


def main():
    warnings.filterwarnings("ignore")
    rng = np.random.default_rng(C.SEMILLA)
    maestra = pd.concat([pd.read_csv(C.RES / "tabla_maestra.csv", parse_dates=["origen"]),
                         pd.read_csv(C.RES / "tabla_maestra_bayesiana.csv", parse_dates=["origen"])], ignore_index=True)
    costo = maestra.groupby(["conjunto", "modelo", "optimizador"]).agg(
        busqueda_s=("busqueda_s", "mean"), ajuste_ms=("ajuste_ms", "median"),
        inferencia_ms=("inferencia_ms", "median")).reset_index()
    N, todos_mej, todas_pr, conj, rangos, todas = {}, [], [], [], {}, []

    for conjunto in C.CONJUNTOS:
        p = cargar(conjunto)
        escala = A.escala_mase(conjunto, p.Mes.min())
        rope = ROPE_REL * float(p.Sales.mean())
        cols = [c for c in p.columns if c not in A.META and not c.startswith(("abl|", "lam|"))]
        tabla = pd.DataFrame([{"conjunto": conjunto, "modelo": c.split("|")[0], "optimizador": c.split("|")[1], "columna": c,
                               **A.metricas(p, c, escala, rng), "R2": r2(p, c)} for c in cols])
        todas.append(tabla)
        base = tabla[tabla.modelo.isin(BASE)]
        refs = pd.concat([base.loc[base.groupby("modelo").MAE.idxmin()], tabla[tabla.columna.isin(A.REFS_FIJAS)]])
        orig = tabla[tabla.columna == ORIG]
        mej = pd.concat([orig, refs]).sort_values("MAE").merge(costo, on=["conjunto", "modelo", "optimizador"], how="left")
        todos_mej.append(mej)

        # Pruebas por pares: original frente a las 11 referencias
        m_orig = A.mensual(p, ORIG)
        filas = []
        for r in refs.itertuples():
            m_ref = A.mensual(p, r.columna)
            d = m_orig - m_ref
            w = stats.wilcoxon(d)
            filas.append({"conjunto": conjunto, "referencia": r.modelo, "optimizador": r.optimizador,
                          "MAE_referencia": r.MAE, **E.comparar(m_orig, m_ref, rng), "W": float(w.statistic),
                          **E.bayes_rope(d, rope), "rope": rope})
        pr = pd.DataFrame(filas)
        for c in ["p_DM", "p_GW", "p_wilcoxon"]:
            pr[c + "_holm"] = E.holm(pr[c])
        todas_pr.append(pr)

        # MCS con las 39 (o 15) combinaciones
        L = np.column_stack([A.mensual(p, c) for c in cols])
        pm = E.mcs(L, cols, rng)
        dentro = [c for c in cols if pm[c] >= C.ALFA_MCS]
        rangos[conjunto] = {c.split("|")[0]: float(A.mensual(p, c).mean()) / escala
                            for c in cols if c.split("|")[1] in ("bayes", "-")}
        mejor_ref = refs.loc[refs.MAE.idxmin()]
        pos = int(mej.MAE.rank(method="min")[mej.modelo == C.ORIGINAL].iloc[0])
        conj.append({"conjunto": conjunto, "posicion_original": pos, "modelos": len(mej),
                     "MASE_original": float(orig.MASE.iloc[0]), "mejor_referencia": mejor_ref.modelo,
                     "MASE_mejor_referencia": float(mejor_ref.MASE),
                     "posicion_bayesiana": int(mej.MAE.rank(method="min")[mej.modelo == BAYES].iloc[0]),
                     "MASE_bayesiana": float(mej[mej.modelo == BAYES].MASE.iloc[0]),
                     "mcs_tamano": len(dentro), "mcs_total": len(cols), "p_mcs_original": float(pm[ORIG]),
                     "mcs_fuera": [c for c in cols if c not in dentro]})

    mej, pr, conj = pd.concat(todos_mej), pd.concat(todas_pr), pd.DataFrame(conj)
    mej.to_csv(C.RES / "paper_resultados.csv", index=False)
    pd.concat(todas).to_csv(C.RES / "paper_lideres.csv", index=False)
    pr.to_csv(C.RES / "paper_pruebas.csv", index=False)
    conj.to_csv(C.RES / "paper_conjuntos.csv", index=False)

    ec = pd.DataFrame(rangos).T
    fid = E.friedman_iman_davenport(ec.to_numpy())
    N["entre_conjuntos"] = {"modelos": list(ec.columns), "rango_medio": dict(zip(ec.columns, map(float, fid["rangos"]))),
                            "F": fid["F"], "p": fid["p"]}

    # Veredicto con 11 referencias (misma regla del protocolo)
    ps = pr[pr.conjunto == C.PRINCIPAL]
    mejor = ps.loc[ps.MAE_referencia.idxmin()]
    sup = conj.set_index("conjunto")
    N["veredicto"] = {"c1_menor_MAE_que_todas": bool((ps.dif < 0).all()),
                      "rechazos_frente_mejor": int((mejor[["p_DM_holm", "p_GW_holm", "p_wilcoxon_holm"]] < C.ALFA).sum()),
                      "d_cohen": float(mejor.d_cohen), "delta_cliff": float(mejor.delta_cliff),
                      "gana_en_adicionales": int((sup.drop(C.PRINCIPAL).posicion_original == 1).sum())}

    # Costo computacional en el conjunto principal: modelos finales entrenados con todos los datos
    _, datos, _ = EJ.cargar(C.PRINCIPAL)
    archivos = {"KNN": "knn", "Ridge": "ridge", "Lasso": "lasso", "Árbol de decisión": "arbol_de_decision",
                "Random Forest": "random_forest", "XGBoost": "xgboost", "SVR": "svr", C.ORIGINAL: "perfil_jerarquico_original"}
    filas = []
    for r in mej[mej.conjunto == C.PRINCIPAL].itertuples():
        if r.modelo in archivos:
            obj = joblib.load(C.ROOT / "artifacts" / f"{archivos[r.modelo]}.joblib")
            kb = (C.ROOT / "artifacts" / f"{archivos[r.modelo]}.joblib").stat().st_size / 1024
        elif r.modelo in (BAYES, C.HGB):              # sin artefacto: se reentrena con los hiperparámetros del último origen
            ult = maestra[(maestra.conjunto == C.PRINCIPAL) & (maestra.modelo == r.modelo) & (maestra.optimizador == r.optimizador)]
            params = json.loads(ult.sort_values("origen").hiperparametros.iloc[-1])
            from sklearn.linear_model import BayesianRidge
            est = BayesianRidge() if r.modelo == BAYES else clone(C.EST_HGB)
            obj = EJ.con_pipeline(est).set_params(**EJ.prefijo(params)).fit(datos[FEATURES], datos[TARGET])
            kb = tamano_kb(obj)
        else:
            obj, kb = None, np.nan
        est = getattr(obj, "named_steps", {}).get("modelo", obj) if obj is not None else None
        filas.append({"modelo": r.modelo, "optimizador": r.optimizador, "busqueda_s": r.busqueda_s,
                      "ajuste_ms": r.ajuste_ms, "inferencia_ms": r.inferencia_ms, "tamano_kb": kb,
                      "parametros": n_parametros(est) if est is not None else np.nan})
    pd.DataFrame(filas).to_csv(C.RES / "paper_costo.csv", index=False)
    (C.RES / "paper_numeros.json").write_text(json.dumps(N, indent=2, ensure_ascii=False, default=float), encoding="utf-8")

    cols = ["modelo", "optimizador", "MAE", "MAE_sd", "RMSE", "WAPE", "MASE", "R2", "desvio_total_pct"]
    print(mej[mej.conjunto == C.PRINCIPAL][cols].round(3).to_string(index=False))
    print(ps[["referencia", "dif", "ic_lo", "ic_hi", "DM", "p_DM", "p_DM_holm", "GW", "p_GW", "p_GW_holm", "W", "p_wilcoxon",
              "p_wilcoxon_holm", "p_bayes_mejor", "p_bayes_rope", "p_bayes_peor", "d_cohen", "delta_cliff", "meses_mejor"]].round(3).to_string(index=False))
    print(conj.drop(columns="mcs_fuera").round(3).to_string(index=False))
    print(pd.DataFrame(filas).round(2).to_string(index=False))
    print(json.dumps(N, indent=1, ensure_ascii=False, default=float))


if __name__ == "__main__":
    main()
