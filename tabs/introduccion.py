"""tabs/introduccion.py — Portada: héroe, cifras clave con mini gráficos, la serie completa
(2014–2017 real + proyección 2018) filtrable por categoría, accesos rápidos y el contexto del proyecto."""
from functools import lru_cache

import dash_bootstrap_components as dbc
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, callback, dcc, html

from common import (CATEGORY_COLORS, PROY, REAL, ROOT, insight, kpi, load_artifact, load_data,
                    modelo_listo, soft_card, style_fig, tabla_datos, usd)

TODAS = "Todas"
MESES = ["ene", "feb", "mar", "abr", "may", "jun", "jul", "ago", "sep", "oct", "nov", "dic"]

# accesos rápidos: (ruta, icono, título, descripción, color)
ACCESOS = [
    ("/eda", "bi-bar-chart-line", "Explorar los datos", "Distribuciones, tiempo, atípicos y mapa.", "#4a93e6"),
    ("/modelos", "bi-cpu", "Comparar modelos", "7 modelos, 4 optimizadores y un modelo propio.", "#b89cf0"),
    ("/modelos", "bi-graph-up-arrow", "Predecir y proyectar 2018", "Predicción en vivo y proyección mes a mes.", "#f0b851"),
]


# Enlaces de acceso al conjunto de datos (se repiten en Contexto › Datos y en el libro)
URL_DATOS_REPO = "https://github.com/AlexT09/SuperstorePython2/blob/main/data/raw/Sample%20-%20Superstore.csv"
URL_DATOS_ORIGEN = "https://www.kaggle.com/datasets/bravehart101/sample-supermarket-dataset"


@lru_cache(maxsize=1)
def _vista_previa():
    """Primeras 5 filas del archivo original, sin el nombre del cliente (el proyecto no lo usa)."""
    crudo = pd.read_csv(ROOT / "data" / "raw" / "Sample - Superstore.csv", encoding="latin-1")
    cols = ["Order ID", "Order Date", "Segment", "State", "Region", "Category", "Sub-Category", "Sales", "Quantity",
            "Discount", "Profit"]
    return crudo[cols].head(5), crudo.shape


def _dataset():
    """Ficha del conjunto de datos para abrir la presentación: fuente, tamaño, unidad, vista previa y enlaces."""
    df = load_data()
    muestra, (filas, columnas) = _vista_previa()
    return soft_card("El conjunto de datos: Sample Superstore", [
        dbc.Row([
            dbc.Col(html.Ul([
                html.Li([html.B("Fuente: "), "Sample Superstore, un conjunto público descargado de Kaggle; datos "
                         "ficticios de una cadena minorista de Estados Unidos."]),
                html.Li([html.B("Tamaño: "), f"{filas:,} líneas de pedido × {columnas} columnas".replace(",", ".")
                         + f", del {df.Order_Date.min():%d/%m/%Y} al {df.Order_Date.max():%d/%m/%Y}."]),
                html.Li([html.B("Cada fila: "), "un producto dentro de un pedido, con su venta (Sales), cantidad, "
                         "descuento y ganancia."]),
                html.Li([html.B("Lo que se pronostica: "), "las ventas mensuales de 3 categorías × 4 regiones = "
                         "12 series de 48 meses."]),
            ], className="mb-0"), lg=7),
            dbc.Col(html.Ul([
                html.Li(html.A("Archivo CSV en el repositorio del proyecto", href=URL_DATOS_REPO, target="_blank")),
                html.Li(html.A("Fuente de descarga: Kaggle, Sample Superstore Dataset", href=URL_DATOS_ORIGEN, target="_blank")),
                html.Li(html.A("Declaración de datos (procedencia, licencia, limpieza)", href=URL_DATOS_REPO.replace(
                    "raw/Sample%20-%20Superstore.csv", "DATA_STATEMENT.md"), target="_blank")),
            ], className="mb-0"), lg=5),
        ], className="mb-3"),
        tabla_datos(muestra),
        insight("Primeras 5 filas del archivo original (11 de las 21 columnas; se omite el nombre del cliente). "
                "Variables completas en Contexto › Datos."),
    ], "blue", height_full=False)


@lru_cache(maxsize=1)
def _cifras():
    """Totales por año y series mensuales para los KPI de la portada."""
    df = load_data()
    anio = df.Order_Date.dt.year
    por_anio = df.groupby(anio).agg(ventas=("Sales", "sum"), ganancia=("Profit", "sum"),
                                    pedidos=("Order ID", "nunique"), clientes=("Customer ID", "nunique"))
    mes = df.set_index("Order_Date").resample("MS")
    mensual = mes.agg({"Sales": "sum", "Profit": "sum", "Order ID": "nunique"})
    return df, por_anio, mensual


def _sparkline(serie, color):
    """Mini gráfico de área sin ejes para el pie de un KPI."""
    fig = go.Figure(go.Scatter(
        x=serie.index, y=serie.values, mode="lines", line=dict(color=color, width=2, shape="spline", smoothing=.6),
        fill="tozeroy", fillcolor=_hex_alpha(color, .16),
        hovertemplate="%{x|%b %Y}<br><b>%{y:,.0f}</b><extra></extra>"))
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=56, paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", showlegend=False,
                      hoverlabel=dict(bgcolor="rgba(11,26,38,.95)", font=dict(color="#fff", size=11)))
    fig.update_xaxes(visible=False, fixedrange=True)
    fig.update_yaxes(visible=False, fixedrange=True, rangemode="tozero")
    return dcc.Graph(figure=fig, config={"displayModeBar": False}, className="spark")


def _hex_alpha(hex_color, a):
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{a})"


def _delta(actual, previo):
    v = (actual / previo - 1) * 100
    clase, icono = ("sube", "bi-arrow-up-right") if v >= 0 else ("baja", "bi-arrow-down-right")
    return html.Span([html.I(className=f"bi {icono}"), f"{v:+.1f} %"], className=f"kpi-delta {clase}")


def _kpis():
    df, a, m = _cifras()
    margen = df.Profit.sum() / df.Sales.sum() * 100
    tarjetas = [
        ("Ventas 2014–2017", usd(df.Sales.sum()), [_delta(a.ventas[2017], a.ventas[2016]), " 2017 vs 2016"],
         "bi-cash-stack", m.Sales, "#4a93e6"),
        ("Ganancia total", usd(df.Profit.sum()), [_delta(a.ganancia[2017], a.ganancia[2016]), " 2017 vs 2016"],
         "bi-piggy-bank", m.Profit, "#8fd3b0"),
        ("Pedidos", f"{df['Order ID'].nunique():,}", [_delta(a.pedidos[2017], a.pedidos[2016]), " 2017 vs 2016"],
         "bi-bag-check", m["Order ID"], "#b89cf0"),
        ("Margen de ganancia", f"{margen:.1f} %", f"{df['Customer ID'].nunique():,} clientes únicos",
         "bi-percent", (m.Profit / m.Sales * 100).clip(lower=-20), "#f0b851"),
    ]
    return dbc.Row([
        dbc.Col(html.Div(kpi(label, valor, detalle, icon=icono, extra=_sparkline(serie, color)),
                         className="h-100 kpi-spark-wrap"), md=6, xl=3, className="mb-3")
        for label, valor, detalle, icono, serie, color in tarjetas
    ], className="kpi-fila")


def _hero():
    proy = ""
    if modelo_listo():
        p18 = load_artifact("proyeccion_2018.csv").Proyeccion.sum()
        v17 = _cifras()[1].ventas[2017]
        proy = f" El modelo proyecta {usd(p18)} para 2018 ({(p18 / v17 - 1) * 100:+.1f} % frente a 2017)."
    return html.Div([
        html.Span([html.I(className="bi bi-stars"), "Analítica de ventas · Machine learning"], className="hero-eyebrow"),
        html.H2(["Del histórico de ventas a ", html.Span("decisiones para 2018", className="grad")],
                className="hero-titulo"),
        html.P("Cuatro años de pedidos de una cadena minorista de EE. UU., analizados para entender qué mueve "
               "las ventas y proyectar cuánto venderá cada categoría y región, mes a mes." + proy,
               className="hero-texto"),
        html.Div([
            dcc.Link([html.I(className="bi bi-graph-up-arrow"), "Ver la proyección"], href="/modelos",
                     className="btn-brillo"),
            dcc.Link([html.I(className="bi bi-compass"), "Empezar por el EDA"], href="/eda", className="btn-vidrio"),
        ], className="hero-acciones"),
    ], className="hero mb-4")


def _accesos():
    return dbc.Row([
        dbc.Col(dcc.Link([
            html.I(className=f"bi {icono} principal", style={"--c": color}),
            html.B(titulo), html.Small(texto), html.I(className="bi bi-arrow-up-right flecha"),
        ], href=href, className="nav-card"), md=4, className="mb-3")
        for href, icono, titulo, texto, color in ACCESOS
    ])


def layout():
    df = load_data()
    return html.Div([
        _hero(),
        _kpis(),
        _dataset(),
        html.Div(className="mb-3"),

        dbc.Row([
            dbc.Col(soft_card("Ventas mensuales: real y proyección", [
                html.Div([
                    dbc.RadioItems(id="intro-cat", options=[TODAS] + sorted(df.Category.unique()), value=TODAS,
                                   inline=True, className="segmentado", inputClassName="btn-check",
                                   labelClassName="seg-btn", labelCheckedClassName="activo"),
                ], className="mb-2"),
                dcc.Graph(id="intro-serie", config={"displayModeBar": False}),
                html.Div(id="intro-lectura"),
            ], "yellow"), lg=12, className="mb-3"),
        ]),

        html.H5("Recorrer el proyecto", className="section-title"),
        _accesos(),

        html.H5("Contexto", className="section-title"),
        dbc.Row([
            dbc.Col(soft_card("El proyecto", [
                html.P("Sample Superstore es un conjunto de datos de muestra de una cadena minorista de "
                       "Estados Unidos (2014–2017) con cerca de 10.000 "
                       "líneas de pedido de muebles, artículos de oficina y tecnología."),
                html.P("El proyecto tiene dos partes: un análisis exploratorio (EDA) que identifica qué "
                       "explica el nivel de ventas, y un modelo de machine learning que usa esos hallazgos "
                       "para proyectar cuánto venderá la empresa cada mes, por categoría y región.",
                       className="mb-0"),
            ], "blue"), md=7, className="mb-3"),
            dbc.Col(soft_card("Preguntas del proyecto", [
                html.P(["1. ¿Qué factores (categoría, región, segmento, tiempo) explican el nivel de ventas (",
                        html.Code("Sales"), ")?"], className="fw-bold"),
                html.P("2. ¿Cuánto venderá cada categoría en cada región, mes a mes, durante 2018?",
                       className="fw-bold"),
                html.P("El tablero tiene tres secciones: Contexto (planteamiento y método), EDA (los datos) y "
                       "Modelos (comparación, predicción en vivo y proyección).", className="text-muted mb-0"),
            ], "lavender"), md=5, className="mb-3"),
        ]),
        dbc.Row([
            dbc.Col(soft_card("Por qué importa", html.P(
                "Una cadena minorista vende a través de varias categorías de producto, regiones y "
                "segmentos de cliente, y sus ventas cambian a lo largo del año. Saber qué combinación de "
                "factores mueve las ventas, y anticipar cuánto se venderá, es un insumo directo para "
                "planificar: cuánto inventario comprar, qué metas fijar a cada región y cuándo reforzar "
                "la operación.", className="mb-0"), "peach"), md=6, className="mb-3"),
            dbc.Col(soft_card("Impacto para el negocio", html.Ul([
                html.Li("Metas comerciales: fijar la meta de cada región según lo que se espera vender."),
                html.Li("Inventario y compras: abastecer cada categoría antes de su temporada alta."),
                html.Li("Alerta temprana: detectar dónde se proyecta una caída frente al año anterior."),
                html.Li("Presupuesto: proyectar los ingresos del año siguiente."),
            ], className="mb-0"), "green"), md=6, className="mb-3"),
        ]),
    ])


@callback(Output("intro-serie", "figure"), Output("intro-lectura", "children"), Input("intro-cat", "value"))
def actualizar_serie(cat):
    df = load_data()
    if cat != TODAS:
        df = df[df.Category == cat]
    real = df.set_index("Order_Date").Sales.resample("MS").sum()
    color = CATEGORY_COLORS.get(cat, REAL)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=real.index, y=real.values, name="Ventas reales", mode="lines",
        line=dict(color=color, width=2.6, shape="spline", smoothing=.5),
        fill="tozeroy", fillcolor=_hex_alpha(color, .12),
        hovertemplate="%{x|%b %Y}<br>Real: <b>$%{y:,.0f}</b><extra></extra>"))

    lectura = None
    if modelo_listo():
        p = load_artifact("proyeccion_2018.csv")
        if cat != TODAS:
            p = p[p.Category == cat]
        p = p.groupby("Mes").Proyeccion.sum()
        # la proyección arranca en el último mes real para que la línea no quede cortada
        x = [real.index[-1], *p.index]
        y = [real.values[-1], *p.values]
        fig.add_trace(go.Scatter(
            x=x, y=y, name="Proyección 2018", mode="lines+markers",
            line=dict(color=PROY, width=2.6, dash="dot", shape="spline", smoothing=.5),
            marker=dict(size=6, color=PROY, line=dict(color="#0b1a26", width=1.5)),
            hovertemplate="%{x|%b %Y}<br>Proyección: <b>$%{y:,.0f}</b><extra></extra>"))
        fig.add_vrect(x0=p.index[0], x1=p.index[-1], fillcolor="rgba(240,184,81,.06)", line_width=0,
                      annotation_text="2018 proyectado", annotation_position="top left",
                      annotation_font=dict(color=PROY, size=11))
        v17 = real[real.index.year == 2017].sum()
        pico = p.idxmax()
        lectura = insight([
            html.B(cat if cat != TODAS else "Todas las categorías"), f": {usd(p.sum())} proyectados para 2018 "
            f"({(p.sum() / v17 - 1) * 100:+.1f} % frente a 2017). El mes más fuerte sería ",
            html.B(f"{MESES[pico.month - 1]} ({usd(p.max())})"),
            ", en línea con la temporada alta de fin de año que se repite en los cuatro años reales."])

    style_fig(fig, 380)
    fig.update_layout(hovermode="x unified", legend=dict(orientation="h", y=1.08, x=0, yanchor="bottom"),
                      margin=dict(l=50, r=20, t=30, b=30))
    fig.update_xaxes(showspikes=False, rangeslider=dict(visible=True, thickness=.06, bgcolor="rgba(255,255,255,.03)"),
                     rangeselector=dict(
                         buttons=[dict(count=1, label="1 año", step="year", stepmode="backward"),
                                  dict(count=2, label="2 años", step="year", stepmode="backward"),
                                  dict(step="all", label="Todo")],
                         bgcolor="rgba(255,255,255,.06)", activecolor="rgba(240,184,81,.35)",
                         font=dict(color="#fff", size=11), x=1, xanchor="right", y=1.12))
    fig.update_yaxes(tickprefix="$", tickformat="~s", showspikes=False)
    return fig, lectura
