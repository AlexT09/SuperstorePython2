"""
app.py — Pronóstico de ventas Sample Superstore (Dash + Bootstrap), aplicación multipágina.

Tres secciones, una por archivo de pages/: Contexto ("/"), EDA ("/eda") y Modelos ("/modelos").
El tablero no entrena nada: lee los datos procesados (data/), los resultados de los experimentos
(experiments/resultados/) y los modelos guardados (artifacts/).

Ejecutar:  python app.py   ->   http://127.0.0.1:8061        Producción:  gunicorn app:server
"""
import os
from pathlib import Path

import dash
import dash_bootstrap_components as dbc
from dash import Dash, Input, Output, dcc, html
from dotenv import load_dotenv

load_dotenv(Path(__file__).with_name(".env"))  # clave y modelo de la IA (chatbot flotante)

from tabs import asistente  # noqa: E402

app = Dash(__name__, use_pages=True, external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.BOOTSTRAP],
           suppress_callback_exceptions=True, title="Pronóstico de ventas Sample Superstore")
server = app.server

# Menú lateral: (ruta, etiqueta, icono de Bootstrap Icons, descripción, clave para las sugerencias del asistente)
MENU = [
    ("/", "Contexto", "bi-house-door", "Problema, datos y método", "introduccion"),
    ("/eda", "EDA", "bi-bar-chart-line", "Análisis exploratorio", "eda"),
    ("/modelos", "Modelos", "bi-cpu", "Comparación, predicción y modelo propio", "modelos"),
]
RUTAS = {ruta: (etiqueta, descripcion, clave) for ruta, etiqueta, _, descripcion, clave in MENU}


def sidebar() -> html.Div:
    enlaces = [dbc.NavLink([html.I(className=f"bi {icono}", title=etiqueta), html.Span(etiqueta)],
                           href=ruta, active="exact", className="sidebar-link")
               for ruta, etiqueta, icono, _, _ in MENU]
    return html.Div([
        html.Div([html.Div(html.I(className="bi bi-shop"), className="brand-logo"),
                  html.Div([html.Div("Superstore", className="brand-nombre"),
                            html.Div("Panel de ventas", className="brand-sub")])],
                 className="sidebar-brand"),
        dbc.Nav([html.Div("Secciones", className="sidebar-grupo"), *enlaces], vertical=True, pills=True,
                className="sidebar-nav"),
        html.Div([html.Div([html.Span("AT", className="avatar", style={"background": "#f0b851"})],
                           className="avatares"),
                  html.Span("Alex Teran", className="nombres")], className="sidebar-pie"),
    ], className="sidebar")


app.layout = html.Div([
    dcc.Location(id="url"),
    dcc.Store(id="pestana-activa", data="introduccion"),  # la lee el chatbot para dar sugerencias por sección
    sidebar(),
    html.Main([
        html.Header([
            html.Div([
                html.Button(html.I(className="bi bi-layout-sidebar"), className="btn-menu",
                            title="Plegar o desplegar el menú (tecla [)", **{"aria-label": "Plegar menú"}),
                html.Div([html.Div(id="topbar-ruta", className="topbar-ruta"),
                          html.H1("Pronóstico de ventas Sample Superstore", className="topbar-titulo")],
                         style={"minWidth": 0}),
            ], className="topbar-izq"),
            html.Div([
                html.Div([html.I(className="bi bi-calendar3"), "2014 – 2017 · proyección 2018"], className="topbar-chip mono"),
                html.Div([html.Span(className="topbar-punto"), "EDA + machine learning"], className="topbar-chip"),
            ], className="topbar-der"),
        ], className="topbar"),
        html.Div(dash.page_container, className="contenido"),
    ], className="main"),
    asistente.widget(),  # chatbot flotante abajo a la derecha, visible en todas las secciones
], className="panel")


@app.callback(Output("topbar-ruta", "children"), Output("pestana-activa", "data"), Input("url", "pathname"))
def actualizar_ruta(pathname):
    """Migas de la barra superior y sección activa (para las sugerencias del asistente)."""
    etiqueta, descripcion, clave = RUTAS.get(pathname or "/", RUTAS["/"])
    migas = [html.Span(etiqueta), html.I(className="bi bi-chevron-right"), html.Span(descripcion, className="actual")]
    return migas, clave


if __name__ == "__main__":
    # Puerto propio para no chocar con otros dashboards que usan el 8050
    port = int(os.environ.get("SUPERSTORE_PORT", 8061))
    print(f"Superstore Dash -> http://127.0.0.1:{port}/")
    app.run(debug=True, port=port)
