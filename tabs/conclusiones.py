"""tabs/conclusiones.py — Conclusiones del EDA y del modelo, recomendaciones y limitaciones."""
import dash_bootstrap_components as dbc
from dash import html

from common import insight, load_artifact, load_data, modelo_listo, page_header, soft_card, usd
from tabs.eda import hallazgos


def _modelo():
    if not modelo_listo():
        return html.P("Entrene el modelo (python model/train_model.py) para ver sus conclusiones.", className="mb-0")
    r = load_artifact("resumen.json")
    rf, rec = r["referencia"], r["prueba_recursiva"]
    res = load_artifact("resultados_modelos.csv").sort_values("MAE").reset_index(drop=True)
    mejor, segundo = res.iloc[0], res.iloc[1]
    ingenuo = res[res.modelo.str.startswith("Referencia")].iloc[0]
    return html.Ul([
        html.Li(f"El modelo propio ({r['mejor_modelo']}) tuvo el menor error al pronosticar, uno a uno, los 24 meses "
                f"de 2016 y 2017: {usd(mejor.MAE)} por serie y mes, un {(1 - mejor.MAE / ingenuo.MAE) * 100:.0f} % "
                f"menos que repetir el año anterior ({usd(ingenuo.MAE)})."),
        html.Li(f"La ventaja frente al segundo modelo ({segundo.modelo}, {usd(segundo.MAE)}) es pequeña y no es "
                "estadísticamente significativa: con 24 meses de evaluación no se puede afirmar que sea mejor, solo "
                "que es igual de preciso con un modelo mucho más simple."),
        html.Li(f"Al proyectar 2017 completo solo con datos hasta 2016, el total anual quedó a "
                f"{rec['total_desvio_total_pct']:+.1f} % del real (repetir el año anterior, a "
                f"{rf['total_desvio_total_pct']:+.1f} %), con un error típico mensual de ±{rec['total_MAE_pct']:.0f} %."),
        html.Li("Lo que explica el resultado es el patrón estacional común a todas las series: estimarlo con las "
                "12 series juntas es lo que reduce el error. El peso propio de cada serie aporta poco."),
        html.Li(f"Para 2018 se proyectan {usd(r['proyeccion_2018'])}, "
                f"{(r['proyeccion_2018'] / r['ventas_2017'] - 1) * 100:+.0f} % frente a 2017, con el pico entre "
                f"septiembre y diciembre."),
    ], className="mb-0")


def layout():
    return html.Div([
        page_header("Conclusiones", "Lo que aprendimos, qué recomendamos y qué tener en cuenta"),
        dbc.Row([
            dbc.Col(soft_card("Análisis exploratorio", [
                html.P(["La distribución de ", html.Code("Sales"), " es muy sesgada hacia valores bajos, con "
                        "pocos pedidos de alto valor: los atípicos son el 11.7 % de los pedidos pero el 64 % de las "
                        "ventas, así que no se eliminan. ", html.Code("Category"), " es el factor que más explica "
                        "diferencias en el valor de venta: Technology y Furniture tienen pedidos más grandes, "
                        "mientras que Office Supplies domina en volumen con pedidos de menor valor."]),
                html.P([html.Code("Region"), " y ", html.Code("Segment"), " muestran diferencias leves. En el "
                        "tiempo, las ventas tienen un patrón estacional que se repite cada año (bajas en enero y "
                        "febrero, altas de septiembre a diciembre) y crecen desde 2016."], className="mb-0"),
            ], "blue"), lg=6, className="mb-3"),
            dbc.Col(soft_card("Modelo de proyección", _modelo(), "lavender"), lg=6, className="mb-3"),
        ]),
        dbc.Row([
            dbc.Col(soft_card("Recomendaciones para el negocio", [
                html.Ol([html.Li([html.B(f"{h['factor']}: "), h["recomendacion"]]) for h in hallazgos(load_data())],
                        className="mb-0"),
                insight("Salen de los hallazgos del EDA (sección 6 de esa pestaña), donde está el dato que "
                        "respalda cada una."),
            ], "peach"), lg=6, className="mb-3"),
            dbc.Col(soft_card("Limitaciones", html.Ul([
                html.Li("Dataset de ejemplo (2014–2017) de una sola tienda: no generaliza a otros negocios."),
                html.Li("Solo 36 meses aptos para modelar: las métricas tienen incertidumbre y un año atípico "
                        "las cambiaría."),
                html.Li("Sin precios, costos, campañas ni datos externos (economía, competencia): el modelo solo "
                        "conoce el historial de ventas y el calendario."),
                html.Li("La proyección de 2018 extiende el crecimiento reciente; si ese ritmo cambia, el error "
                        "crecerá con cada mes proyectado."),
                html.Li("A nivel de una categoría en una región el error mensual es alto; la proyección es más "
                        "confiable para totales (por región, por categoría o de la empresa)."),
                html.Li("El EDA es asociativo: no demuestra causalidad."),
            ], className="mb-0"), "green"), lg=6, className="mb-3"),
        ]),
    ])
