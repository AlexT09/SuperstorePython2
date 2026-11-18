"""
pages/contexto.py — Sección 1: Contexto.

Planteamiento del problema, datos, metodología, marco teórico, limitaciones y conclusiones.
Cada subsección reutiliza el layout() de tabs/<nombre>.py; aquí solo se añaden la descripción de
los datos y las limitaciones y consideraciones éticas.
"""
import dash
import dash_bootstrap_components as dbc
from dash import Input, Output, callback, dcc, html

from common import insight, load_data, load_mensual, page_header, soft_card, tabla_datos
from tabs import conclusiones, introduccion, marco_teorico, metodologia, problema

dash.register_page(__name__, path="/", name="Contexto", title="Contexto · Superstore")

VERSION, ACTUALIZADO = "1.0.0", "15 de noviembre de 2026"
DICCIONARIO = [
    ("Order Date", "Fecha", "Fecha del pedido; define el mes de la venta."),
    ("Category", "Categórica (3)", "Furniture, Office Supplies o Technology."),
    ("Region", "Categórica (4)", "Central, East, South o West."),
    ("Segment", "Categórica (3)", "Tipo de cliente: Consumer, Corporate o Home Office."),
    ("Sub-Category", "Categórica (17)", "Subfamilia de producto."),
    ("State", "Categórica", "Estado de Estados Unidos del envío."),
    ("Sales", "Numérica (USD)", "Valor de la línea de pedido. Sumada por mes, es lo que se pronostica."),
    ("Quantity", "Numérica", "Unidades de la línea de pedido."),
    ("Discount", "Numérica (0 a 0.8)", "Descuento aplicado."),
    ("Profit", "Numérica (USD)", "Ganancia de la línea de pedido."),
]
REFERENCIAS = [
    "Sample Superstore Dataset. Kaggle, bravehart101.",
    "Fildes, R., Ma, S. y Kolassa, S. (2022). Retail forecasting: research and practice. International Journal of Forecasting, 38(4).",
    "Godahewa, R., Bergmeir, C., Webb, G., Hyndman, R. y Montero-Manso, P. (2021). Monash Time Series Forecasting Archive. NeurIPS Datasets and Benchmarks.",
    "Hansen, P., Lunde, A. y Nason, J. (2011). The model confidence set. Econometrica, 79(2).",
    "Chen, H. y Boylan, J. (2008). Empirical evidence on individual, group and shrinkage seasonal indices. International Journal of Forecasting, 24(3).",
]


def _datos():
    d, m = load_data(), load_mensual()
    import pandas as pd
    dic = pd.DataFrame(DICCIONARIO, columns=["Variable", "Tipo", "Descripción"])
    return html.Div([
        dbc.Row([
            dbc.Col(soft_card("Fuente y alcance", html.Ul([
                html.Li("Sample Superstore: conjunto público descargado de Kaggle."),
                html.Li(f"Periodo: {d.Order_Date.min():%d/%m/%Y} a {d.Order_Date.max():%d/%m/%Y}."),
                html.Li(f"{len(d):,} líneas de pedido y 21 variables en el archivo original.".replace(",", ".")),
                html.Li("Licencia: la que indique la página de Kaggle del conjunto; se usa con fines académicos."),
                html.Li("Son datos de demostración: no corresponden a una empresa identificada."),
                html.Li(["Acceso: ", html.A("archivo CSV", href=introduccion.URL_DATOS_REPO, target="_blank"), " · ",
                         html.A("descarga en Kaggle", href=introduccion.URL_DATOS_ORIGEN, target="_blank"), "."]),
            ], className="mb-0")), md=6, className="mb-3"),
            dbc.Col(soft_card("Tarea y variable objetivo", html.Ul([
                html.Li([html.B("Tarea: "), "regresión sobre series de tiempo (pronóstico del mes siguiente)."]),
                html.Li(f"Unidad de análisis: categoría × región × mes. {m.groupby(['Category', 'Region']).ngroups} series de "
                        f"{m.Mes.nunique()} meses ({len(m)} filas)."),
                html.Li([html.B("Variable objetivo: "), "ventas del mes en dólares. El modelo predice un índice (ventas ÷ "
                         "promedio de los 12 meses anteriores) y lo convierte a dólares."]),
                html.Li("Variables de entrada: categoría, región, mes del año e historial de ventas de la serie."),
            ], className="mb-0")), md=6, className="mb-3"),
        ]),
        soft_card("Diccionario de variables", [
            tabla_datos(dic),
            insight("El detalle de la limpieza y de la tabla mensual está en la sección EDA, pestaña ETL. La declaración "
                    "completa de los datos está en data/DATA_STATEMENT.md."),
        ], height_full=False),
    ])


def _limitaciones():
    return html.Div([
        dbc.Row([
            dbc.Col(soft_card("Limitaciones", html.Ul([
                html.Li("Datos de ejemplo de un solo país y cuatro años: las conclusiones sirven para comparar métodos, "
                        "no para trasladarlas a una empresa sin validarlas con sus datos."),
                html.Li("Series cortas: 24 meses de evaluación dan poca potencia a las pruebas estadísticas."),
                html.Li("Sin precios, promociones, inventario ni variables externas."),
                html.Li("El error por serie es alto; el pronóstico es más confiable para totales."),
                html.Li("El modelo propio supone que las series comparten el patrón estacional. Donde no sea así, pierde."),
            ], className="mb-0")), md=6, className="mb-3"),
            dbc.Col(soft_card("Consideraciones éticas y de sesgo", html.Ul([
                html.Li("El archivo incluye nombres de clientes ficticios; el proyecto no los usa: solo agrega ventas."),
                html.Li("Las series grandes dominan el error en dólares: un modelo puede verse bien y fallar en las "
                        "series pequeñas. Por eso se reporta también el error por serie."),
                html.Li("Un pronóstico usado para fijar metas afecta a personas. Debe leerse con su intervalo y no "
                        "como una cifra exacta."),
                html.Li("El asistente de IA responde con los datos del proyecto, pero puede equivocarse: sus respuestas "
                        "no reemplazan la lectura de las tablas."),
            ], className="mb-0")), md=6, className="mb-3"),
        ]),
        dbc.Row([
            dbc.Col(soft_card("Autor y versión", html.Ul([
                html.Li("Alex Teran."),
                html.Li(f"Versión {VERSION}. Última actualización: {ACTUALIZADO}."),
                html.Li(["Código: ", html.A("github.com/AlexT09/SuperstorePython2", href="https://github.com/AlexT09/SuperstorePython2",
                                            target="_blank")]),
                html.Li(["Informe: ", html.A("alext09.github.io/SuperstorePython2", href="https://alext09.github.io/SuperstorePython2/",
                                             target="_blank")]),
            ], className="mb-0")), md=5, className="mb-3"),
            dbc.Col(soft_card("Referencias", html.Ol([html.Li(r) for r in REFERENCIAS], className="mb-0 small")),
                    md=7, className="mb-3"),
        ]),
    ])


SECCIONES = [("resumen", "Resumen", introduccion.layout), ("problema", "Problema y objetivos", problema.layout),
             ("datos", "Datos", _datos), ("metodologia", "Metodología", metodologia.layout),
             ("marco", "Marco teórico", marco_teorico.layout), ("limitaciones", "Limitaciones y ética", _limitaciones),
             ("conclusiones", "Conclusiones", conclusiones.layout)]


def layout():
    return html.Div([
        page_header("Contexto", "Qué se quiere resolver, con qué datos y cómo"),
        dbc.Tabs([dbc.Tab(label=etiqueta, tab_id=clave) for clave, etiqueta, _ in SECCIONES],
                 id="ctx-tabs", active_tab="resumen", className="mb-3"),
        dcc.Loading(html.Div(id="ctx-contenido"), type="dot", color="#f0b851"),
    ])


@callback(Output("ctx-contenido", "children"), Input("ctx-tabs", "active_tab"))
def mostrar(clave):
    return html.Div(dict((c, f) for c, _, f in SECCIONES)[clave](), className="fade-in")
