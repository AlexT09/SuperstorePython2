"""tabs/metodologia.py — Flujo del proyecto, validación anidada, optimización y comparación estadística."""
import dash_bootstrap_components as dbc
from dash import html

from common import page_header, soft_card

PASOS = [
    ("ETL", "CSV crudo → dataset limpio y tabla mensual (categoría × región × mes)"),
    ("EDA", "Distribución de las ventas, factores, estacionalidad y tendencia"),
    ("Variables", "Mes del año e historial de ventas, sin información del futuro"),
    ("Modelos", "7 regresores dentro de un Pipeline y un modelo propio"),
    ("Optimización", "Grid, Random, bayesiana y genética, con el mismo presupuesto"),
    ("Validación anidada", "24 meses pronosticados uno a uno; los hiperparámetros se eligen dentro de cada entrenamiento"),
    ("Comparación estadística", "Model Confidence Set, Giacomini–White, Diebold–Mariano y Wilcoxon, con Holm"),
    ("Evaluación ampliada", "Tres conjuntos adicionales, ruido, menos datos e intervalos de predicción"),
    ("Proyección", "Modelo reentrenado con toda la historia → 2018 mes a mes"),
]


def layout():
    flujo = html.Div([
        html.Div([html.Span(str(i), className="flujo-num"), html.B(t), html.Span(d)], className="flujo-paso")
        for i, (t, d) in enumerate(PASOS, 1)], className="flujo")
    return html.Div([
        page_header("Metodología", "Cómo se pasa de los datos crudos a la proyección"),
        soft_card("Flujo del proyecto", flujo, "blue", height_full=False),
        dbc.Row([
            dbc.Col(soft_card("Datos y unidad de análisis", html.Ul([
                html.Li("Se agregan las 9.994 líneas de pedido en ventas mensuales por categoría × región: "
                        "12 series × 48 meses = 576 filas."),
                html.Li("2014 solo aporta historia (las variables necesitan 12 meses previos)."),
                html.Li(["Evaluación con ", html.B("origen móvil"), ": cada uno de los 24 meses de 2016 y 2017 se "
                         "pronostica entrenando solo con los meses anteriores (288 pronósticos por modelo)."]),
                html.Li("Nunca se entrena con meses posteriores al que se pronostica: mezclar meses dejaría al "
                        "modelo ver el futuro (fuga de información)."),
            ], className="mb-0"), "blue"), md=6, className="mt-3"),
            dbc.Col(soft_card("Variable objetivo", [
                html.P("El modelo predice un índice: cuántas veces el promedio de los últimos 12 meses se "
                       "venderá ese mes. Luego se multiplica por ese promedio para volver a dólares."),
                html.P("Así una misma regla sirve para series grandes (Technology en West) y pequeñas "
                       "(Furniture en South), y la proyección sigue el nivel reciente de cada serie.",
                       className="mb-0"),
            ], "lavender"), md=6, className="mt-3"),
        ]),
        dbc.Row([
            dbc.Col(soft_card("Validación anidada", html.Ul([
                html.Li(["Bucle externo: los 24 meses de evaluación. Mide el error que se reporta."]),
                html.Li(["Bucle interno: dentro de cada entrenamiento, ", html.Code("TimeSeriesSplit"),
                         " con pliegues de 3 meses elige los hiperparámetros."]),
                html.Li(["El preprocesamiento (", html.Code("OneHotEncoder"), " y ", html.Code("StandardScaler"),
                         ") vive dentro del Pipeline y se ajusta solo con los datos de entrenamiento de cada pliegue."]),
                html.Li("Semilla fija (42) en todos los componentes con azar."),
            ], className="mb-0"), "peach"), md=6, className="mt-3"),
            dbc.Col(soft_card("Optimización de hiperparámetros", html.Ul([
                html.Li([html.B("Grid Search: "), "todas las combinaciones de una rejilla."]),
                html.Li([html.B("Random Search: "), "16 puntos al azar del espacio de búsqueda."]),
                html.Li([html.B("Bayesiana (Optuna): "), "modelo sustituto TPE y mejora esperada; 16 evaluaciones."]),
                html.Li([html.B("Genética (DEAP): "), "torneo, cruce por mezcla, mutación gaussiana y elitismo; 16 evaluaciones."]),
                html.Li("Criterio: menor MAE de validación cruzada. Todos los modelos reciben el mismo presupuesto."),
            ], className="mb-0"), "green"), md=6, className="mt-3"),
        ]),
        dbc.Row([
            dbc.Col(soft_card("Comparación estadística", html.Ul([
                html.Li("Las pruebas se aplican al error de cada mes (24 valores por modelo), que respeta el orden temporal."),
                html.Li([html.B("Model Confidence Set: "), "de todas las combinaciones, cuáles no se distinguen de la mejor."]),
                html.Li([html.B("Giacomini–White, Diebold–Mariano y Wilcoxon: "), "el modelo propio frente a cada referencia, "
                         "con corrección de Holm, intervalos por bootstrap estacionario y tamaño del efecto."]),
                html.Li("Las hipótesis, métricas, conjuntos y pruebas se registraron antes de ejecutar los experimentos "
                        "(experiments/protocolo.md)."),
            ], className="mb-0"), "blue"), md=6, className="mt-3"),
            dbc.Col(soft_card("Referencias", html.Ul([
                html.Li("Ingenuo estacional: repetir las ventas del mismo mes del año anterior. Un modelo solo aporta si lo supera."),
                html.Li("Método Theta y boosting por histogramas, como referencias adicionales a los modelos del curso."),
                html.Li("Tres conjuntos públicos adicionales (turismo, hospital y M3) para comprobar si los resultados se mantienen."),
            ], className="mb-0"), "lavender"), md=6, className="mt-3"),
        ]),
    ])
