"""
pages/eda.py — Sección 2: EDA.

Explorador interactivo: los filtros de arriba (categoría, región, fechas y descuento) actualizan
todos los gráficos con un solo callback. Las otras pestañas muestran el análisis completo
(tabs/eda.py) y la preparación de los datos (tabs/etl.py).
"""
import dash
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
from dash import Input, Output, callback, dcc, html

from common import insight, kpi, load_data, page_header, soft_card, tabla_datos, usd
from src import plots as P
from tabs import eda as eda_completo
from tabs import etl

dash.register_page(__name__, path="/eda", name="EDA", title="EDA · Superstore")

CFG = {"displayModeBar": False}
NUMERICAS = ["Sales", "Quantity", "Discount", "Profit"]
COLORES = ["Category", "Region", "Segment"]


def _explorador():
    d = load_data()
    sel = lambda id_, opts, valor, **k: dcc.Dropdown(opts, valor, id=id_, **k)          # noqa: E731
    return html.Div([
        soft_card("Filtros (se aplican a todos los gráficos de esta pestaña)", dbc.Row([
            dbc.Col([html.Label("Categoría", className="small fw-bold"),
                     sel("ex-cat", sorted(d.Category.unique()), sorted(d.Category.unique()), multi=True)], lg=4, className="mb-2"),
            dbc.Col([html.Label("Región", className="small fw-bold"),
                     sel("ex-reg", sorted(d.Region.unique()), sorted(d.Region.unique()), multi=True)], lg=3, className="mb-2"),
            dbc.Col([html.Label("Fechas del pedido", className="small fw-bold"), html.Br(),
                     dcc.DatePickerRange(id="ex-fechas", min_date_allowed=d.Order_Date.min(), max_date_allowed=d.Order_Date.max(),
                                         start_date=d.Order_Date.min(), end_date=d.Order_Date.max(), display_format="DD/MM/YYYY")],
                    lg=3, className="mb-2"),
            dbc.Col([html.Label("Descuento", className="small fw-bold"),
                     dcc.RangeSlider(0, 0.8, 0.1, value=[0, 0.8], id="ex-desc",
                                     marks={0: "0 %", 0.4: "40 %", 0.8: "80 %"})], lg=2, className="mb-2"),
        ]), height_full=False),
        html.Div(className="mb-3"),
        dbc.Row([
            dbc.Col([html.Label("Explorador: eje X", className="small fw-bold"), sel("ex-x", NUMERICAS, "Discount", clearable=False)], md=3),
            dbc.Col([html.Label("Eje Y", className="small fw-bold"), sel("ex-y", NUMERICAS, "Profit", clearable=False)], md=3),
            dbc.Col([html.Label("Color", className="small fw-bold"), sel("ex-color", COLORES, "Category")], md=3),
        ], className="mb-3"),
        dcc.Loading(html.Div(id="ex-salida"), type="dot", color="#f0b851"),
    ])


@callback(Output("ex-salida", "children"),
          Input("ex-cat", "value"), Input("ex-reg", "value"), Input("ex-fechas", "start_date"), Input("ex-fechas", "end_date"),
          Input("ex-desc", "value"), Input("ex-x", "value"), Input("ex-y", "value"), Input("ex-color", "value"))
def actualizar(cats, regs, inicio, fin, desc, x, y, color):
    d = load_data()
    if not cats or not regs or not inicio or not fin:
        return dbc.Alert("Elija al menos una categoría, una región y un rango de fechas.", color="warning")
    d = d[d.Category.isin(cats) & d.Region.isin(regs) & (d.Order_Date >= inicio) & (d.Order_Date <= fin)
          & (d.Discount >= desc[0] - 1e-9) & (d.Discount <= desc[1] + 1e-9)]
    if len(d) < 30:
        return dbc.Alert("Con esos filtros quedan menos de 30 líneas de pedido: amplíe la selección.", color="warning")

    # variable objetivo: ventas mensuales de cada serie
    mensual = d.assign(Mes=d.Order_Date.dt.to_period("M").dt.to_timestamp()).groupby(["Category", "Region", "Mes"]).Sales.sum()
    total = mensual.groupby("Mes").sum().asfreq("MS", fill_value=0.0)
    asim = float(mensual.skew()) if len(mensual) > 2 else 0.0
    por_mes = total.groupby(total.index.month).mean()
    pico, valle = int(por_mes.idxmax()), int(por_mes.idxmin())
    nombres = ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto", "septiembre", "octubre",
               "noviembre", "diciembre"]

    # calidad: faltantes y atípicos (regla de 1.5 veces el rango intercuartílico)
    filas = []
    for c in NUMERICAS:
        q1, q3 = d[c].quantile([0.25, 0.75])
        fuera = ((d[c] < q1 - 1.5 * (q3 - q1)) | (d[c] > q3 + 1.5 * (q3 - q1))).sum()
        filas.append({"Variable": c, "Faltantes": int(d[c].isna().sum()), "Atípicos": int(fuera),
                      "% de atípicos": f"{fuera / len(d) * 100:.1f} %", "Mediana": f"{d[c].median():,.2f}",
                      "Media": f"{d[c].mean():,.2f}"})
    calidad = pd.DataFrame(filas)
    at = d.Sales > d.Sales.quantile(0.75) + 1.5 * (d.Sales.quantile(0.75) - d.Sales.quantile(0.25))
    corr = d[NUMERICAS].corr(method="spearman")
    muestra = d.sample(min(len(d), 2500), random_state=42)

    bloques = [
        dbc.Row([
            dbc.Col(kpi("Líneas de pedido", f"{len(d):,}".replace(",", "."), "con los filtros actuales"), md=3, className="mb-3"),
            dbc.Col(kpi("Ventas", usd(d.Sales.sum()), f"{total.size} meses"), md=3, className="mb-3"),
            dbc.Col(kpi("Venta mediana por línea", usd(d.Sales.median(), 2) if d.Sales.median() < 100 else usd(d.Sales.median()),
                        f"media {usd(d.Sales.mean())}"), md=3, className="mb-3"),
            dbc.Col(kpi("Mes pico", nombres[pico - 1].capitalize(), f"mes más bajo: {nombres[valle - 1]}"), md=3, className="mb-3"),
        ]),
        dbc.Row([
            dbc.Col(soft_card("Variable objetivo: ventas mensuales por serie", [
                dcc.Graph(figure=P.fig_distribucion(mensual.values, None, "Ventas del mes por categoría y región (USD)", "$"), config=CFG),
                insight(f"Asimetría de {asim:.2f}: la mayoría de los meses vende poco y unos pocos venden mucho. Por eso el "
                        "modelo predice un índice relativo al nivel de cada serie y no dólares directamente."),
            ]), lg=6, className="mb-3"),
            dbc.Col(soft_card("Correlación entre variables numéricas", [
                dcc.Graph(figure=P.fig_correlacion(d[NUMERICAS]), config=CFG),
                insight(f"El descuento se asocia con menor ganancia (ρ = {corr.loc['Discount', 'Profit']:.2f}) y casi no "
                        f"cambia el valor de la venta (ρ = {corr.loc['Discount', 'Sales']:.2f})."),
            ]), lg=6, className="mb-3"),
        ]),
        dbc.Row([
            dbc.Col(soft_card(f"Explorador: {y} frente a {x}", [
                dcc.Graph(figure=P.fig_explorador(muestra, x, y, color), config=CFG),
                insight(f"Correlación de Spearman entre {x} y {y}: {corr.loc[x, y]:.2f}. "
                        + (f"Se muestran {len(muestra):,} líneas al azar de {len(d):,}.".replace(",", ".") if len(d) > len(muestra) else
                           "Se muestran todas las líneas de la selección.")),
            ]), lg=7, className="mb-3"),
            dbc.Col(soft_card("Valores faltantes y atípicos", [
                tabla_datos(calidad),
                insight(f"No hay valores faltantes. Los atípicos de Sales son el {at.mean() * 100:.1f} % de las líneas pero el "
                        f"{d.Sales[at].sum() / d.Sales.sum() * 100:.0f} % de las ventas: son pedidos grandes reales y no se eliminan."),
            ]), lg=5, className="mb-3"),
        ]),
    ]

    if total.size >= 24:
        from statsmodels.tsa.stattools import adfuller
        try:
            p_adf = float(adfuller(total.values, maxlag=min(12, total.size // 3))[1])
            lectura = (f"Prueba de Dickey–Fuller aumentada: p = {p_adf:.3f}. " +
                       ("Se rechaza la raíz unitaria: la serie es estacionaria alrededor de su media." if p_adf < 0.05 else
                        "No se rechaza la raíz unitaria: la serie tiene tendencia o estacionalidad y no es estacionaria. "
                        "Dividir por el promedio de 12 meses (el índice) quita la tendencia."))
        except Exception:
            lectura = "La prueba de estacionariedad no se pudo calcular con esta selección."
        ac = P.acf(total.values)
        r12 = f"{ac[11]:.2f}" if len(ac) >= 12 else "no disponible"
        bloques += [
            html.H4("Serie de tiempo del total mensual", className="section-title"),
            soft_card("Descomposición: tendencia, estacionalidad y resto", [
                dcc.Graph(figure=P.fig_descomposicion(total), config=CFG),
                insight(f"El componente estacional alcanza su máximo en {nombres[pico - 1]} y su mínimo en {nombres[valle - 1]}; "
                        "la tendencia es la media móvil centrada de 12 meses."),
            ], height_full=False),
            html.Div(className="mb-3"),
            dbc.Row([
                dbc.Col(soft_card("Autocorrelación (ACF)", [
                    dcc.Graph(figure=P.fig_acf(total.values, None), config=CFG),
                    insight(f"Correlación con el mismo mes del año anterior (rezago 12): {r12}. Es la señal que usan los "
                            "modelos: lo que pasó hace un año se repite."),
                ]), lg=6, className="mb-3"),
                dbc.Col(soft_card("Autocorrelación parcial (PACF)", [
                    dcc.Graph(figure=P.fig_pacf(total.values, None), config=CFG),
                    insight(lectura),
                ]), lg=6, className="mb-3"),
            ]),
        ]
    else:
        bloques.append(dbc.Alert("La descomposición y las autocorrelaciones necesitan al menos 24 meses: amplíe el rango de fechas.",
                                 color="info"))
    return html.Div(bloques, className="fade-in")


SECCIONES = [("explorador", "Explorador", _explorador), ("completo", "Análisis completo", eda_completo.layout),
             ("etl", "ETL", etl.layout)]


def layout():
    return html.Div([
        page_header("Análisis exploratorio", "Qué muestran los datos antes de modelar"),
        dbc.Tabs([dbc.Tab(label=etiqueta, tab_id=clave) for clave, etiqueta, _ in SECCIONES],
                 id="eda-tabs", active_tab="explorador", className="mb-3"),
        dcc.Loading(html.Div(id="eda-contenido"), type="dot", color="#f0b851"),
    ])


@callback(Output("eda-contenido", "children"), Input("eda-tabs", "active_tab"))
def mostrar(clave):
    return html.Div(dict((c, f) for c, _, f in SECCIONES)[clave](), className="fade-in")
