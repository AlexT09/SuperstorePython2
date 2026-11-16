"""
pages/modelos.py — Sección 3: Modelos.

Tabla de líderes, predicción en vivo (con intervalo y explicación local), diagnóstico del modelo
elegido, comparación estadística, el modelo original, su ficha y la proyección de 2018.
Todo se lee de experiments/resultados/ y artifacts/: aquí no se entrena nada.
"""
import dash
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
from dash import Input, Output, callback, dcc, html

from common import ROOT, insight, kpi, page_header, soft_card, tabla_datos, usd
from model.features import FEATURES, NOMBRES
from src import data as D
from src import plots as P
from src.explicacion import shapley_exacto
from tabs import pronostico, proyeccion  # al importar se registran sus callbacks (antes de arrancar el servidor)

dash.register_page(__name__, path="/modelos", name="Modelos", title="Modelos · Superstore")

CFG = {"displayModeBar": False}
MESES = ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto", "septiembre", "octubre",
         "noviembre", "diciembre"]
OPT = {"grid": "Grid Search", "random": "Random Search", "bayes": "Bayesiana (Optuna)", "genetico": "Genética (DEAP)",
       "-": "sin ajuste"}
CAMPOS = [("pv-prom12", "prom_12m", "Promedio mensual de los últimos 12 meses"),
          ("pv-lag12", "lag_12", "Ventas del mismo mes del año anterior"),
          ("pv-lag1", "lag_1", "Ventas del mes anterior"),
          ("pv-prom3", "prom_3m", "Promedio de los últimos 3 meses")]
PSEUDOCODIGO = """\
Entrada: índices z(s,t), rezagos r(s,t), mes m(t), fuerza de agrupamiento lam, tolerancia tol
Salida:  perfil P[1..12], pesos w_s de cada serie, peso común w0

 1. P[m]  <- media de z(s,t) en los meses con m(t) = m, sobre todas las series
 2. a_st  <- z(s,t) - P[m(t)]          b_st <- r(s,t) - P[m(t)]
 3. S_ab[s] <- suma_t a_st * b_st      S_bb[s] <- suma_t b_st^2
 4. w0 <- recortar( suma_s S_ab[s] / suma_s S_bb[s] )      w_s <- w0
 5. repetir
 6.     w_s <- recortar( (S_ab[s] + lam * w0) / (S_bb[s] + lam) )   para cada serie s
 7.     w0  <- promedio de los w_s
 8. hasta que J baje menos de tol * (1 + J)
 9. devolver P, w, w0

Pronóstico:  z_hat(s,t) = P[m(t)] + w_s * ( r(s,t) - P[m(t)] )
             ventas(s,t) = z_hat(s,t) * promedio de los 12 meses anteriores de la serie
"""
CERCANOS = [
    ("Bunn y Vassilopoulos (1993, 1999)", "Índices estacionales de grupo para muchos productos.",
     "Usan el índice del grupo para todas las series; no combinan con la información propia de cada una."),
    ("Miller y Williams (2003)", "Contracción de los índices estacionales de una serie hacia 1.",
     "Contraen el índice propio hacia la ausencia de estacionalidad, no hacia un perfil común."),
    ("Chen y Boylan (2007, 2008)", "Reglas para elegir entre índice individual, de grupo o contraído.",
     "Eligen una opción por serie; aquí se mezclan las dos con un peso continuo."),
    ("Zhang, Chen, Boylan y Scarf (2013)", "Estimador que combina agrupamiento y contracción.",
     "Es el antecedente más cercano. Sus pesos salen de fórmulas de error cuadrático; aquí se aprenden "
     "minimizando el error de pronóstico, y los pesos de las series se agrupan entre sí."),
    ("Montero-Manso y Hyndman (2021)", "Modelos globales: un solo modelo para todas las series.",
     "Principio general; no proponen esta regla de combinación."),
]


def _sin_resultados():
    return soft_card("Resultados no disponibles", html.P([
        "Ejecute ", html.Code("python experiments/ejecutar.py"), ", ", html.Code("analizar.py"), " y ",
        html.Code("artefactos.py"), " para generar los resultados."], className="mb-0"))


def _nombres_modelos():
    return list(D.metadata()["modelos"])


# ================================================================ 1. líderes
def _lideres():
    m = D.resultado("mejores.csv")
    m = m[m.conjunto == "superstore"].sort_values("MAE")
    t = pd.DataFrame({
        "Modelo": m.modelo, "Optimizador": m.optimizador.map(OPT),
        "MAE (USD), media ± sd": [f"{a:,.0f} ± {b:,.0f}" for a, b in zip(m.MAE, m.MAE_sd)],
        "IC 95 % del MAE": [f"[{a:,.0f}, {b:,.0f}]" for a, b in zip(m.MAE_ic_lo, m.MAE_ic_hi)],
        "RMSE (USD)": m.RMSE.map("{:,.0f}".format), "MASE": m.MASE.map("{:.3f}".format),
        "Desvío del total": m.desvio_total_pct.map("{:+.1f} %".format),
        "Inferencia (ms)": m.inferencia_ms.map(lambda v: "—" if pd.isna(v) else f"{v:.2f}"),
    })
    todas = D.resultado("lideres.csv")
    todas = todas[todas.conjunto == "superstore"].sort_values("MAE")
    t2 = pd.DataFrame({
        "Modelo": todas.modelo, "Optimizador": todas.optimizador.map(OPT),
        "MAE (USD)": todas.MAE.round(0), "sd": todas.MAE_sd.round(0), "RMSE (USD)": todas.RMSE.round(0),
        "Evaluaciones": todas.evaluaciones.round(0), "Búsqueda (s)": todas.busqueda_s.round(2),
        "Inferencia (ms)": todas.inferencia_ms.round(2)})
    mejor, seg = m.iloc[0], m.iloc[1]
    opt = D.resultado("optimizadores.csv")
    curvas = D.resultado("anytime.csv")
    bloques = [
        dbc.Row([
            dbc.Col(kpi("Menor MAE", mejor.modelo, f"{usd(mejor.MAE)} por serie y mes"), md=4, className="mb-3"),
            dbc.Col(kpi("Mejor referencia", seg.modelo, f"{usd(seg.MAE)} · {OPT[seg.optimizador]}"), md=4, className="mb-3"),
            dbc.Col(kpi("Evaluación", "24 meses", "origen móvil, 288 pronósticos por modelo"), md=4, className="mb-3"),
        ]),
        soft_card("Mejor combinación de cada modelo", [
            tabla_datos(t, filtro=True),
            insight("Cada fila es la combinación de modelo y método de optimización con menor MAE en el bucle externo "
                    "(24 meses, reentrenando cada mes). La desviación estándar es la del MAE entre meses. Un MASE "
                    "menor que 1 significa que el modelo mejora al ingenuo estacional de referencia."),
        ], height_full=False),
        html.H4("Todas las combinaciones", className="section-title"),
        soft_card("Modelo × método de optimización", [
            tabla_datos(t2, filtro=True, pagina=12),
            insight("Los 7 modelos base y el modelo original se ajustaron con los 4 métodos, con el mismo "
                    "presupuesto de 16 evaluaciones (Grid Search recorre su rejilla completa)."),
        ], height_full=False),
    ]
    if opt is not None and curvas is not None:
        c = curvas[curvas.modelo == "Random Forest"]
        n = int(c.groupby("optimizador").evaluacion.max().min())
        series = {OPT[o]: g.sort_values("evaluacion").mejor_MAE_interno.to_numpy()[:n] for o, g in c.groupby("optimizador")}
        to = pd.DataFrame({"Método": opt.optimizador.map(OPT), "MAE medio de los 8 modelos (USD)": opt.MAE_medio.round(0),
                           "Evaluaciones por búsqueda": opt.evaluaciones.round(1), "Segundos por búsqueda": opt.busqueda_s.round(2),
                           "Veces mejor": opt.veces_mejor.astype(int)})
        bloques += [
            html.H4("Métodos de optimización", className="section-title"),
            dbc.Row([
                dbc.Col(soft_card("Calidad y costo por método", [
                    tabla_datos(to),
                    insight("«Veces mejor» cuenta en cuántos de los 8 modelos ese método dio el menor MAE del bucle externo."),
                ]), lg=6, className="mb-3"),
                dbc.Col(soft_card("Convergencia de la búsqueda (Random Forest)", [
                    dcc.Graph(figure=P.fig_lineas(list(range(1, n + 1)), series, None, "Evaluación",
                                                  "Mejor MAE interno hasta el momento (índice)"), config=CFG),
                    insight("Mejor valor encontrado hasta cada evaluación, promediado sobre los 24 orígenes. Una curva "
                            "que baja antes aprovecha mejor el presupuesto."),
                ]), lg=6, className="mb-3"),
            ]),
        ]
    return html.Div(bloques)


# ================================================================ 2. predicción en vivo
def _prediccion():
    h = D.historia()
    meses = sorted(h.Mes.unique(), reverse=True)
    futuro = h.Mes.max()
    op_mes = [{"label": f"{MESES[pd.Timestamp(m).month - 1].capitalize()} {pd.Timestamp(m).year}"
                        + (" (futuro)" if m == futuro else ""), "value": pd.Timestamp(m).strftime("%Y-%m-%d")} for m in meses]
    esquema = D.metadata()["esquema"]
    sel = lambda id_, opts, valor: dcc.Dropdown(opts, valor, id=id_, clearable=False)        # noqa: E731

    def campo(id_, etiqueta):
        return dbc.Col([html.Label(etiqueta, className="small fw-bold", htmlFor=id_),
                        dbc.InputGroup([dbc.InputGroupText("$"),
                                        dbc.Input(id=id_, type="number", min=0, max=1e6, step=1, debounce=True)])],
                       md=6, className="mb-3")

    return html.Div([
        html.P("Elija la serie, el mes y el modelo. El formulario se llena con la historia real de la serie; puede "
               "cambiar cualquier valor para simular un escenario.", className="page-subtitle mb-3"),
        dbc.Row([
            dbc.Col(soft_card("Datos de entrada", [
                dbc.Row([
                    dbc.Col([html.Label("Modelo", className="small fw-bold"),
                             sel("pv-modelo", _nombres_modelos(), D.ORIGINAL)], md=12, className="mb-3"),
                    dbc.Col([html.Label("Categoría", className="small fw-bold"),
                             sel("pv-cat", esquema["Category"]["valores"], "Technology")], md=6, className="mb-3"),
                    dbc.Col([html.Label("Región", className="small fw-bold"),
                             sel("pv-reg", esquema["Region"]["valores"], "West")], md=6, className="mb-3"),
                    dbc.Col([html.Label("Mes a pronosticar", className="small fw-bold"),
                             sel("pv-mes", op_mes, op_mes[0]["value"])], md=6, className="mb-3"),
                    dbc.Col([html.Label("Confianza del intervalo", className="small fw-bold"),
                             dbc.RadioItems([{"label": "80 %", "value": "0.8"}, {"label": "90 %", "value": "0.9"}], "0.8",
                                            id="pv-nivel", inline=True)], md=6, className="mb-3"),
                    *[campo(i, e) for i, _, e in CAMPOS],
                ]),
                insight("Valores en dólares, entre 0 y 1.000.000. El promedio de 12 meses debe ser mayor que cero: "
                        "es el nivel por el que se multiplica el índice pronosticado."),
            ]), lg=5, className="mb-3"),
            dbc.Col(html.Div(id="pv-salida"), lg=7, className="mb-3"),
        ]),
    ])


@callback([Output(i, "value") for i, _, _ in CAMPOS],
          Input("pv-cat", "value"), Input("pv-reg", "value"), Input("pv-mes", "value"))
def llenar_formulario(cat, reg, mes):
    """Llena los campos con la historia real de la serie en el mes elegido."""
    h = D.historia()
    f = h[(h.Category == cat) & (h.Region == reg) & (h.Mes == pd.Timestamp(mes))]
    if f.empty:
        return [None] * len(CAMPOS)
    return [round(float(f.iloc[0][col]), 0) for _, col, _ in CAMPOS]


@callback(Output("pv-salida", "children"),
          Input("pv-modelo", "value"), Input("pv-cat", "value"), Input("pv-reg", "value"), Input("pv-mes", "value"),
          Input("pv-nivel", "value"), *[Input(i, "value") for i, _, _ in CAMPOS])
def predecir(nombre, cat, reg, mes, nivel, prom12, lag12, lag1, prom3):
    valores = [prom12, lag12, lag1, prom3]
    if any(v is None for v in valores):
        return dbc.Alert("Complete los cuatro valores de la historia de la serie.", color="warning")
    if any(v < 0 or v > 1e6 for v in valores):
        return dbc.Alert("Los valores deben estar entre 0 y 1.000.000 dólares.", color="danger")
    if prom12 <= 0:
        return dbc.Alert("El promedio de los últimos 12 meses debe ser mayor que cero.", color="danger")
    try:
        m = pd.Timestamp(mes).month
        fila = pd.DataFrame([{"Category": cat, "Region": reg, "r_lag_12": lag12 / prom12, "r_lag_1": lag1 / prom12,
                              "r_prom_3m": prom3 / prom12, "mes_sin": np.sin(2 * np.pi * m / 12),
                              "mes_cos": np.cos(2 * np.pi * m / 12), "prom_12m": prom12}])
        modelo, meta = D.modelo(nombre), D.metadata()["modelos"][nombre]
        indice = float(modelo.predict(D.entrada(nombre, fila))[0])
        pred = max(0.0, indice * prom12)
        radio = meta["radio_conforme_indice"][nivel] * prom12
        if nombre == D.ORIGINAL:
            e = modelo.explicar(fila)
            w = modelo.pesos_.get(f"{cat}|{reg}", modelo.w_global_)
            etiquetas = ["Mes del año (perfil común a todas las series)", f"Mismo mes del año anterior (peso de la serie: {w:.2f})"]
            aportes, base = [float(e["estacional"][0]) * prom12, float(e["rezago"][0]) * prom12], float(e["base"][0]) * prom12
            nota = ("Descomposición exacta del modelo: la predicción es la base más lo que aporta el mes del año y lo "
                    "que aporta la historia propia de la serie.")
        else:
            phi, b = shapley_exacto(lambda t: modelo.predict(t[FEATURES]), fila, D.fondo(), FEATURES)
            orden = np.argsort(-np.abs(phi))
            etiquetas, aportes, base = [NOMBRES[FEATURES[i]] for i in orden], [float(phi[i]) * prom12 for i in orden], b * prom12
            nota = ("Valores de Shapley exactos (128 coaliciones de las 7 variables): cuánto mueve cada variable la "
                    "predicción respecto de la predicción media.")
    except Exception as error:                                   # una entrada inesperada no detiene la app
        return dbc.Alert(f"No se pudo calcular la predicción: {error}", color="danger")
    cobertura = meta["cobertura_verificada"].get(nivel)
    return html.Div([
        dbc.Row([
            dbc.Col(kpi("Pronóstico del mes", usd(pred), f"{nombre} · índice {indice:.2f}"), md=6, className="mb-3"),
            dbc.Col(kpi(f"Intervalo al {float(nivel):.0%}", f"{usd(max(0, pred - radio))} – {usd(pred + radio)}",
                        "cobertura medida: " + (f"{cobertura:.0%}" if cobertura is not None else "no disponible")),
                    md=6, className="mb-3"),
        ]),
        soft_card("¿Por qué este valor?", [
            dcc.Graph(figure=P.fig_cascada(etiquetas, aportes, base, None), config=CFG),
            insight(nota),
            insight("El intervalo es conforme: su radio es el cuantil de los errores absolutos del índice en los 24 meses "
                    "evaluados, multiplicado por el nivel de la serie. La cobertura medida es la proporción de meses "
                    "posteriores que cayeron dentro del intervalo calculado solo con meses anteriores."),
        ], height_full=False),
    ])


# ================================================================ 3. diagnóstico
def _diagnostico():
    return html.Div([
        dbc.Row(dbc.Col([html.Label("Modelo", className="small fw-bold"),
                         dcc.Dropdown(_nombres_modelos(), D.ORIGINAL, id="dg-modelo", clearable=False)], md=4), className="mb-3"),
        html.Div(id="dg-salida"),
    ])


@callback(Output("dg-salida", "children"), Input("dg-modelo", "value"))
def diagnosticar(nombre):
    p, col = D.predicciones(), D.columna(nombre)
    e = p.Sales - p[col]
    total = p.assign(e=e).groupby("Mes").e.sum()
    r = D.resultado("residuos.csv")
    r = r[(r.conjunto == "superstore") & (r.modelo == nombre)].iloc[0]
    sg = D.artefacto("shapley_global.csv")
    sg = sg[sg.modelo == nombre]
    lectura = lambda pv, si, no: si if pv < 0.05 else no           # noqa: E731
    pruebas = pd.DataFrame([
        ("White", "varianza constante de los residuos", f"{r.p_white:.3f}",
         lectura(r.p_white, "se rechaza: el error crece con el tamaño de la venta", "no se rechaza")),
        ("Ljung–Box (6 rezagos)", "residuos sin autocorrelación", f"{r.p_ljung_box:.3f}",
         lectura(r.p_ljung_box, "se rechaza: queda estructura temporal sin explicar", "no se rechaza")),
        ("BDS", "residuos independientes", "—" if pd.isna(r.p_bds) else f"{r.p_bds:.3f}",
         "no calculable" if pd.isna(r.p_bds) else lectura(r.p_bds, "se rechaza", "no se rechaza")),
        ("Shapiro–Wilk", "residuos normales", f"{r.p_shapiro:.3f}", lectura(r.p_shapiro, "se rechaza", "no se rechaza")),
    ], columns=["Prueba", "Hipótesis nula", "p-valor", "Lectura (α = 0.05)"])
    mae = float(e.abs().mean())
    return html.Div([
        dbc.Row([
            dbc.Col(soft_card("Real frente a pronóstico", [
                dcc.Graph(figure=P.fig_real_vs_pred(p.Sales, p[col], None), config=CFG),
                insight(f"288 pronósticos del bucle externo. Los puntos sobre la diagonal son aciertos; el error medio "
                        f"es de {usd(mae)}. Los meses de ventas muy altas quedan por debajo de la diagonal: el modelo "
                        "se queda corto en los picos."),
            ]), lg=6, className="mb-3"),
            dbc.Col(soft_card("Residuos en el tiempo", [
                dcc.Graph(figure=P.fig_residuos(total.index, total.values), config=CFG),
                insight(f"Suma de las 12 series por mes. El sesgo medio por pronóstico es de {usd(r.sesgo_medio)} "
                        "(positivo: el modelo subestima)."),
            ]), lg=6, className="mb-3"),
        ]),
        dbc.Row([
            dbc.Col(soft_card("Autocorrelación de los residuos", [
                dcc.Graph(figure=P.fig_acf(total.values, None), config=CFG),
                insight("Barras fuera de la banda punteada (en ámbar) indican meses cuyo error se parece al de meses anteriores."),
            ]), lg=6, className="mb-3"),
            dbc.Col(soft_card("Distribución de los residuos", [
                dcc.Graph(figure=P.fig_distribucion(e, None, "Real − pronóstico (USD)", "$"), config=CFG),
                insight(f"Asimetría de {r.asimetria:.2f}: la cola derecha son los meses de ventas excepcionales que ningún modelo anticipa."),
            ]), lg=6, className="mb-3"),
        ]),
        dbc.Row([
            dbc.Col(soft_card("Supuestos de los residuos", [tabla_datos(pruebas),
                    insight("Pruebas sobre los residuos del bucle externo. No invalidan el pronóstico puntual, pero "
                            "indican qué tan fiables son los intervalos que suponen errores homogéneos.")]),
                    lg=6, className="mb-3"),
            dbc.Col(soft_card("Importancia global de las variables", [
                dcc.Graph(figure=P.fig_barras_h([NOMBRES[v] for v in sg.variable], sg.importancia.to_numpy(), None,
                                                "Media del valor absoluto de Shapley (índice)"), config=CFG),
                insight("Promedio, sobre los pronósticos del último año, de cuánto mueve cada variable la predicción."),
            ]), lg=6, className="mb-3"),
        ]),
    ])


# ================================================================ 4. comparación estadística
def _tabla_pruebas(conjunto="superstore"):
    c = D.resultado("pruebas.csv")
    c = c[c.conjunto == conjunto].sort_values("MAE_referencia")
    return pd.DataFrame({
        "Referencia": [f"{a} ({OPT[b]})" for a, b in zip(c.referencia, c.optimizador)],
        "Diferencia de MAE [IC 95 %]": [f"{d:+,.0f} [{lo:+,.0f}, {hi:+,.0f}]" for d, lo, hi in zip(c.dif, c.ic_lo, c.ic_hi)],
        "Diebold–Mariano p / p Holm": [f"{a:.3f} / {b:.3f}" for a, b in zip(c.p_DM, c.p_DM_holm)],
        "Giacomini–White p / p Holm": [f"{a:.3f} / {b:.3f}" for a, b in zip(c.p_GW, c.p_GW_holm)],
        "Wilcoxon p / p Holm": [f"{a:.3f} / {b:.3f}" for a, b in zip(c.p_wilcoxon, c.p_wilcoxon_holm)],
        "d de Cohen": c.d_cohen.map("{:+.2f}".format), "δ de Cliff": c.delta_cliff.map("{:+.2f}".format),
        "Meses en que gana": [f"{n} de 24" for n in c.meses_mejor]})


def _comparacion():
    mcs = D.resultado("mcs.csv")
    mcs = mcs[mcs.conjunto == "superstore"].sort_values("MAE")
    t = pd.DataFrame({"Modelo": mcs.modelo, "Optimizador": mcs.optimizador.map(OPT), "MAE (USD)": mcs.MAE.round(0),
                      "p-valor MCS": mcs.p_mcs.round(3), "En el conjunto de confianza": np.where(mcs.en_conjunto, "Sí", "No")})
    n = D.resultado("numeros.json")["superstore"]
    return html.Div([
        soft_card("Model Confidence Set (α = 0.10)", [
            tabla_datos(t, filtro=True, pagina=12, condicional=[
                {"if": {"filter_query": '{En el conjunto de confianza} = "No"'}, "color": "rgba(255,255,255,.5)"}]),
            insight(f"El procedimiento parte de las {n['mcs_total']} combinaciones y elimina, una a una, las que son "
                    f"significativamente peores. Quedan {n['mcs_tamano']}: con 24 meses de evaluación, los datos no "
                    "alcanzan para separar a la mayoría de los modelos. El modelo original "
                    + ("pertenece" if n["original_en_mcs"] else "no pertenece") + " al conjunto."),
        ], height_full=False),
        html.H4("Modelo original frente a cada referencia", className="section-title"),
        soft_card("Pruebas por pares en Sample Superstore", [
            tabla_datos(_tabla_pruebas()),
            insight("Diferencia = MAE del original − MAE de la referencia (negativa: gana el original). Cada prueba se "
                    "corrige con Holm por las 10 comparaciones. Un intervalo que contiene el cero no permite afirmar "
                    "que haya diferencia."),
        ], height_full=False),
    ])


# ================================================================ 5. modelo original
def _tabla_adicionales():
    m, n = D.resultado("mejores.csv"), D.resultado("numeros.json")
    filas = []
    for c, nombre in [("superstore", "Sample Superstore"), ("turismo", "Tourism Monthly"), ("hospital", "Hospital"), ("m3", "M3 Monthly")]:
        if c not in n:
            continue
        g = m[m.conjunto == c]
        o = g[g.modelo == D.ORIGINAL].iloc[0]
        filas.append({"Conjunto": nombre, "MASE del original": f"{o.MASE:.3f}", "Mejor referencia": n[c]["mejor_referencia"],
                      "MASE de la mejor referencia": f"{n[c]['MAE_mejor_referencia'] / n[c]['escala_mase']:.3f}",
                      "Posición del original": f"{n[c]['posicion_original']} de 11",
                      "¿En el conjunto de confianza?": "Sí" if n[c]["original_en_mcs"] else "No"})
    return pd.DataFrame(filas)


def _original():
    meta = D.metadata()["modelos"][D.ORIGINAL]
    conv, pesos = D.artefacto("convergencia.csv"), D.artefacto("pesos_original.csv")
    v, sens = D.resultado("veredicto.json"), D.resultado("sensibilidad_lam.csv")
    sens = sens[sens.conjunto == "superstore"].sort_values("lam")
    abl = D.resultado("ablacion.csv")
    abl = abl[abl.conjunto == "superstore"]
    ta = pd.DataFrame({"Variante": abl.variante, "MAE (USD)": abl.MAE.map("{:,.0f}".format),
                       "Diferencia frente al completo [IC 95 %]": ["—" if pd.isna(d) else f"{-d:+,.0f} [{-hi:+,.0f}, {-lo:+,.0f}]"
                                                                 for d, lo, hi in zip(abl.dif, abl.ic_lo, abl.ic_hi)],
                       "p Holm (Diebold–Mariano)": abl.p_DM_holm.map(lambda x: "—" if pd.isna(x) else f"{x:.3f}"),
                       "Desvío del total": abl.desvio_total_pct.map("{:+.1f} %".format)})
    conf = D.resultado("conformal.csv")
    conf = conf[conf.conjunto == "superstore"]
    tc = pd.DataFrame({"Modelo": conf.modelo, "Nivel nominal": conf.nivel.map("{:.0%}".format),
                       "Cobertura medida": conf.cobertura.map("{:.1%}".format), "Ancho medio (USD)": conf.ancho_medio.map("{:,.0f}".format)})
    ruido, muestra, semillas = D.resultado("robustez_ruido.csv"), D.resultado("robustez_muestra.csv"), D.resultado("robustez_semillas.csv")
    cumple = lambda b: "Sí" if b else "No"                               # noqa: E731
    tv = pd.DataFrame([
        ("1. Menor MAE que las 10 referencias en Superstore", cumple(v["c1_menor_MAE_que_todas"])),
        ("2. Al menos tres pruebas rechazan la igualdad frente a la mejor referencia",
         cumple(v["c2_tres_pruebas"]) + f" ({sum(v['rechazos'].values())} de 4)"),
        ("3. Tamaño del efecto no despreciable", cumple(v["c3_efecto"]) + f" (d = {v['d_cohen']:+.2f}, δ = {v['delta_cliff']:+.2f})"),
        ("4. Gana en al menos 2 de los 3 conjuntos adicionales y no es peor en ninguno",
         cumple(v["c4_consistencia"]) + f" (gana en {sum(v['gana_en'].values())} de {len(v['gana_en'])})"),
    ], columns=["Condición fijada en el protocolo", "¿Se cumple?"])
    robustez = []
    if ruido is not None:
        r = ruido.pivot(index="nivel", columns="modelo", values="MAE").round(0).reset_index().rename(columns={"nivel": "Contaminación"})
        robustez.append(dbc.Col(soft_card("Ruido y atípicos en el entrenamiento (MAE, USD)", [tabla_datos(r),
                        insight("Se contamina la variable objetivo de entrenamiento y se pronostica el mes real. "
                                "Un modelo robusto cambia poco de una fila a otra.")]), lg=6, className="mb-3"))
    if muestra is not None:
        r = muestra.assign(Estudio=muestra.estudio.map({"meses": "Meses de entrenamiento", "series": "Número de series"}) + ": "
                           + muestra.nivel.astype(str)).pivot(index="Estudio", columns="modelo", values="MAE").round(0).reset_index()
        robustez.append(dbc.Col(soft_card("Menos datos (MAE, USD)", [tabla_datos(r),
                        insight("Con menos series el perfil común se estima con menos datos; con menos meses, todos los modelos pierden.")]),
                        lg=6, className="mb-3"))
    if semillas is not None:
        s = semillas.groupby("modelo").MAE.agg(["mean", "std", "min", "max"]).round(0).reset_index()
        s.columns = ["Modelo", "MAE medio (USD)", "sd", "mínimo", "máximo"]
        robustez.append(dbc.Col(soft_card("Estabilidad por semilla (10 semillas)", [tabla_datos(s),
                        insight("El modelo original es determinista: no tiene semilla y siempre da el mismo resultado.")]),
                        lg=6, className="mb-3"))
    return html.Div([
        dbc.Alert([html.B("Veredicto: " + ("superior" if v["superior"] else "no superior") + ". "),
                   "El modelo original obtuvo el menor error en Sample Superstore, pero la diferencia frente a la mejor "
                   f"referencia ({v['mejor_referencia']}) no es estadísticamente significativa."
                   if not v["superior"] else "Se cumplen las cuatro condiciones fijadas en el protocolo."],
                  color="success" if v["superior"] else "warning"),
        dbc.Row([
            dbc.Col(soft_card("La idea", [
                html.P(["Con 48 meses por serie hay tres o cuatro datos de cada mes del año: la estacionalidad de una serie "
                        "sola se estima con mucho ruido. El modelo estima un ", html.B("perfil estacional común"),
                        " con todas las series y lo combina con lo que hizo cada serie un año antes."]),
                html.P(["La combinación usa ", html.B("un peso por serie"), ". Los pesos se aprenden minimizando el error, "
                        "con un término que acerca cada peso al peso común del grupo: una serie con poca evidencia "
                        "hereda el comportamiento del grupo."], className="mb-0"),
            ]), lg=5, className="mb-3"),
            dbc.Col(soft_card("Formulación", [
                dcc.Markdown(r"""
$$\hat z_{s,t} = P_{m(t)} + w_s\,\bigl(r_{s,t} - P_{m(t)}\bigr)$$

$$J(w, w_0) = \sum_s\sum_t \bigl(a_{s,t} - w_s\, b_{s,t}\bigr)^2 + \lambda \sum_s (w_s - w_0)^2,\qquad 0\le w_s\le 1$$

$$w_s \leftarrow \mathrm{recortar}\!\left(\frac{S_{ab,s} + \lambda w_0}{S_{bb,s} + \lambda}\right),\qquad w_0 \leftarrow \frac{1}{S}\sum_s w_s$$
""", mathjax=True),
                insight("z: índice (ventas ÷ promedio de 12 meses); P: perfil común; r: índice del mismo mes del año anterior; "
                        "a = z − P; b = r − P. Cada actualización es el mínimo exacto de J en su bloque, por eso J nunca aumenta."),
            ]), lg=7, className="mb-3"),
        ]),
        dbc.Row([
            dbc.Col(soft_card("Algoritmo", html.Pre(PSEUDOCODIGO, className="mb-0", style={"fontSize": "12px", "whiteSpace": "pre-wrap"})),
                    lg=7, className="mb-3"),
            dbc.Col(soft_card("Convergencia", [
                dcc.Graph(figure=P.fig_lineas(conv.iteracion, {"J": conv.J}, None, "Iteración", "Objetivo J", alto=260), config=CFG),
                insight(f"Con todos los datos converge en {len(conv) - 1} iteraciones. λ elegido: {meta['hiperparametros']['lam']:.2f}. "
                        "Costo: O(n) para las sumas y O(S) por iteración."),
            ]), lg=5, className="mb-3"),
        ]),
        dbc.Row([
            dbc.Col(soft_card("Pesos aprendidos por serie", [
                dcc.Graph(figure=P.fig_barras_h(pesos.serie.str.replace("|", " · ", regex=False).to_numpy(), pesos.peso.to_numpy(),
                                                None, "Peso del rezago propio (0 = solo perfil común)", alto=360), config=CFG),
                insight("Un peso cercano a cero indica que la serie se pronostica casi solo con el perfil común."),
            ]), lg=6, className="mb-3"),
            dbc.Col(soft_card("Sensibilidad a λ", [
                dcc.Graph(figure=P.fig_lineas(sens.lam, {"Modelo original": sens.MAE}, None, "λ (escala logarítmica)", "MAE (USD)",
                                              log_x=True, alto=360, referencia=("mejor referencia", float(sens.MAE_mejor_referencia.iloc[0]))),
                          config=CFG),
                insight("MAE del bucle externo con λ fijo. El modelo queda por debajo de la mejor referencia en todo el rango; "
                        "el error baja al crecer λ, es decir, cuando los pesos se acercan a un valor común."),
            ]), lg=6, className="mb-3"),
        ]),
        html.H4("Diferencias con los métodos más cercanos", className="section-title"),
        soft_card("Qué se conserva y qué cambia", [
            tabla_datos(pd.DataFrame(CERCANOS, columns=["Trabajo", "Qué propone", "Diferencia con este modelo"])),
            insight("Los índices estacionales de grupo y su contracción son conocidos. No se encontró, en la búsqueda "
                    "documentada en el libro, la regla de pesos por serie con agrupamiento parcial ni su aplicación a este conjunto."),
        ], height_full=False),
        html.H4("Pruebas de hipótesis y veredicto", className="section-title"),
        dbc.Row([
            dbc.Col(soft_card("Regla de decisión", [tabla_datos(tv),
                    insight("Las cuatro condiciones se fijaron antes de ejecutar los experimentos (experiments/protocolo.md, etiqueta protocolo-v1).")]),
                    lg=6, className="mb-3"),
            dbc.Col(soft_card("Conjuntos adicionales", [tabla_datos(_tabla_adicionales()),
                    insight("MASE: error dividido por el del ingenuo estacional; permite comparar conjuntos de escalas distintas.")]),
                    lg=6, className="mb-3"),
        ]),
        soft_card("Original frente a cada referencia (Sample Superstore)", tabla_datos(_tabla_pruebas()), height_full=False),
        html.H4("Robustez", className="section-title"),
        dbc.Row([
            dbc.Col(soft_card("Aporte de cada componente", [tabla_datos(ta),
                    insight("Cada fila cambia una pieza del modelo. Diferencia positiva: la variante es peor que el modelo completo.")]),
                    lg=7, className="mb-3"),
            dbc.Col(soft_card("Intervalos de predicción conformes", [tabla_datos(tc),
                    insight("Cobertura medida en los últimos 12 meses, con intervalos calibrados solo con meses anteriores.")]),
                    lg=5, className="mb-3"),
            *robustez,
        ]),
    ])


# ================================================================ 6 y 7. ficha y proyección
def _ficha():
    ruta = ROOT / "MODEL_CARD.md"
    texto = ruta.read_text(encoding="utf-8") if ruta.exists() else "La ficha del modelo no está disponible."
    return soft_card("Ficha del modelo original", dcc.Markdown(texto, className="ficha-modelo"), height_full=False)


def _proyeccion():
    return html.Div([proyeccion.layout(), html.Hr(), pronostico.layout()])


SECCIONES = [("lideres", "Líderes", _lideres), ("prediccion", "Predicción en vivo", _prediccion),
             ("diagnostico", "Diagnóstico", _diagnostico), ("comparacion", "Comparación estadística", _comparacion),
             ("original", "Modelo original", _original), ("ficha", "Ficha del modelo", _ficha),
             ("proyeccion", "Proyección 2018", _proyeccion)]


def layout():
    if not D.listo():
        return html.Div([page_header("Modelos"), _sin_resultados()])
    return html.Div([
        page_header("Modelos", "Qué modelo pronostica mejor, con qué certeza y por qué"),
        dbc.Tabs([dbc.Tab(label=etiqueta, tab_id=clave) for clave, etiqueta, _ in SECCIONES],
                 id="mod-tabs", active_tab="lideres", className="mb-3"),
        dcc.Loading(html.Div(id="mod-contenido"), type="dot", color="#f0b851"),
    ])


@callback(Output("mod-contenido", "children"), Input("mod-tabs", "active_tab"))
def mostrar(clave):
    """Construye solo la subsección abierta (las demás no se calculan)."""
    return html.Div(dict((c, f) for c, _, f in SECCIONES)[clave](), className="fade-in")
