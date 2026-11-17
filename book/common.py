"""
common.py
Utilidades compartidas por los notebooks del libro: rutas, carga de datos y resultados del modelo,
paleta y estilo de figuras. Es el common.py de la app Dash (carpeta superior), sin los helpers de
interfaz y con tema claro.
"""
import json
import sys
from functools import lru_cache
from pathlib import Path

import pandas as pd
from IPython.display import HTML, Markdown, display

# ---------------------------------------------------------------- rutas
# El libro vive en libro/ y usa los mismos datos y el mismo modelo que la app Dash (raíz del repo)
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # para "from model.features import ..." en los notebooks
DATA_PATH = ROOT / "data" / "superstore_transformado.csv"
MENSUAL_PATH = ROOT / "data" / "ventas_mensuales.csv"
ART = ROOT / "model" / "artifacts"  # resumen y proyección del modelo elegido (experiments/artefactos.py)

# ---------------------------------------------------------------- colores
# Cada color tiene una función fija en todo el libro (los mismos papeles que en la app Dash).
TEXT = "#2f3247"
GRID = "#EEF0F6"
BRAND = "#2E6F95"
BRAND_LIGHT = "#EAF2F7"

REAL = "#2E6F95"         # serie: ventas reales / color principal
PROY = "#E0962B"         # serie: proyección
NEG = "#D1574A"          # caídas
POS = "#2CA58D"          # crecimientos, mejor resultado
REF = "#8A8FA3"          # referencia y líneas auxiliares
MARK = "#153B4E"         # puntos de referencia (p. ej. la media en un boxplot)

SEQ_SCALE = [[0, "#F4F8FB"], [0.5, "#8FBBD1"], [1, "#1F5570"]]
COLORWAY = ["#2E6F95", "#2CA58D", "#8E6CD8", "#E0962B", "#63A0BC", "#D1574A"]
PASTEL = {"blue": "#63A0BC", "green": "#2CA58D", "peach": "#8FBBD1", "pink": "#2E6F95"}

# Un tono distinto por categoría (el mismo en todas las gráficas)
CATEGORY_COLORS = {
    "Furniture": "#2CA58D",
    "Office Supplies": "#4A93E6",
    "Technology": "#8E6CD8",
}

REFERENCIA = "Referencia (año anterior)"


@lru_cache(maxsize=1)
def load_data() -> pd.DataFrame:
    """Carga el dataset transformado (salida de data/generate_data.py)."""
    return pd.read_csv(DATA_PATH, parse_dates=["Order_Date", "Ship Date"])


@lru_cache(maxsize=1)
def load_mensual() -> pd.DataFrame:
    """Ventas por categoría × región × mes (salida de data/generate_data.py)."""
    return pd.read_csv(MENSUAL_PATH, parse_dates=["Mes"])


@lru_cache(maxsize=None)
def load_artifact(name: str):
    """Lee un archivo de model/artifacts/ (CSV como DataFrame, JSON como dict)."""
    path = ART / name
    if path.suffix == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    df = pd.read_csv(path)
    if "Mes" in df:
        df["Mes"] = pd.to_datetime(df["Mes"])
    return df


def mejores_por_modelo() -> pd.DataFrame:
    """Resultados de los 7 modelos (sin la referencia), ordenados por MAE en la prueba."""
    r = load_artifact("resultados_modelos.csv")
    return r[r.modelo != REFERENCIA].sort_values("MAE")


def usd(x, dec=0):
    return f"${x:,.{dec}f}"


def hallazgos(df) -> pd.DataFrame:
    """Hallazgos del EDA convertidos en recomendaciones (la misma función que tabs/eda.py de la app).
    Lo usan los capítulos EDA y Conclusiones. Una fila por factor: hallazgo y recomendación."""
    total = df.Sales.sum()
    # Peso de cada mes en las ventas y crecimiento anual
    mes = df.groupby(df.Order_Date.dt.month).Sales.sum() / total * 100
    crec = df.groupby(df.Order_Date.dt.year).Sales.sum().pct_change() * 100
    # Descuentos de más de 20 %: su pérdida y el margen del resto
    alto = df.Discount > 0.2
    margen_bajo = df[~alto].Profit.sum() / df[~alto].Sales.sum() * 100
    corr = df.Sales.corr(df.Discount, method="spearman")   # Sales no es normal: se usa Spearman
    # Subcategorías con pérdida y margen por categoría
    sub = df.groupby("Sub-Category").agg(p=("Profit", "sum"), d=("Discount", "mean")).sort_values("p")
    neg = sub[sub.p < 0]
    cat = df.groupby("Category").agg(v=("Sales", "sum"), p=("Profit", "sum"))
    cat["m"], cat["share"] = cat.p / cat.v * 100, cat.v / total * 100
    altos = cat.drop(index="Furniture")
    # Región con peor y mejor margen, y estados de la peor que pierden dinero
    reg = df.groupby("Region").agg(v=("Sales", "sum"), p=("Profit", "sum"), d=("Discount", "mean"))
    reg["m"], reg["share"] = reg.p / reg.v * 100, reg.v / total * 100
    peor, mejor = reg.m.idxmin(), reg.m.idxmax()
    est = df[df.Region == peor].groupby("State").agg(p=("Profit", "sum"), d=("Discount", "mean")).sort_values("p")
    est_neg = est[est.p < 0].head(2)
    # Pedidos atípicos por su valor (regla del rango intercuantil)
    q1, q3 = df.Sales.quantile([0.25, 0.75])
    grande = df.Sales > q3 + 1.5 * (q3 - q1)
    # Ventas y clientes por segmento
    seg = df.groupby("Segment").agg(v=("Sales", "sum"), n=("Customer ID", "nunique"))
    seg["vxc"] = seg.v / seg.n
    top_seg = seg.v.idxmax()

    filas = [
        ("Temporada",
         f"Septiembre a diciembre concentran el {mes.loc[9:12].sum():.0f} % de las ventas del año; febrero, solo el "
         f"{mes.loc[2]:.1f} %. Es el factor más fuerte y el que más usa el modelo.",
         "Planificar inventario, personal y campañas para septiembre a diciembre con la proyección mensual. Usar "
         "febrero para promociones controladas o para liquidar inventario."),
        ("Tendencia",
         f"Las ventas crecieron {crec.loc[2016]:+.0f} % en 2016 y {crec.loc[2017]:+.0f} % en 2017.",
         "Fijar las metas de 2018 con la proyección como base, que ya incluye el crecimiento, y revisarlas cada mes "
         "con las ventas reales."),
        ("Descuento",
         f"El descuento casi no se relaciona con las ventas (correlación de Spearman {corr:.2f}), pero las líneas con más de "
         f"20 % de descuento perdieron {usd(-df[alto].Profit.sum())}. Con 20 % o menos, el margen es de "
         f"{margen_bajo:.0f} %.",
         "Poner un tope de 20 % al descuento y pedir aprobación por encima de ese valor: es la acción con mayor "
         "efecto en la ganancia."),
        ("Producto",
         f"{', '.join(f'{s} ({r.d * 100:.0f} % de descuento promedio)' for s, r in neg.iterrows())} pierden dinero. "
         f"Furniture vende el {cat.share['Furniture']:.0f} % pero deja un margen de {cat.m['Furniture']:.1f} %, "
         f"frente a cerca de {altos.m.mean():.0f} % de {' y '.join(altos.index)}.",
         f"Revisar precios y descuentos de {' y '.join(neg.index[:2])}. Dar prioridad a {' y '.join(altos.index)} "
         f"en las campañas: cada venta deja más ganancia."),
        ("Región",
         f"West y East suman el {reg.share[['West', 'East']].sum():.0f} % de las ventas. {peor} tiene el peor "
         f"margen ({reg.m[peor]:.0f} %) y un descuento promedio de {reg.d[peor] * 100:.0f} % "
         f"({mejor}: {reg.d[mejor] * 100:.0f} %)"
         + (f"; {' y '.join(est_neg.index)} pierden dinero con descuentos promedio de "
            f"{est_neg.d.min() * 100:.0f} a {est_neg.d.max() * 100:.0f} %." if len(est_neg) else "."),
         f"Aplicar primero el tope de descuento en {peor}"
         + (f", empezando por {' y '.join(est_neg.index)}" if len(est_neg) else "")
         + f", y replicar ahí la política comercial de {mejor}, la región con mejor margen."),
        ("Pedidos grandes",
         f"El {grande.mean() * 100:.0f} % de las líneas de pedido (las atípicas por su valor) genera el "
         f"{df[grande].Sales.sum() / total * 100:.0f} % de las ventas.",
         "Dar atención y seguimiento dedicados a los clientes de pedidos grandes: perder uno pesa más que perder "
         "muchos pequeños."),
        ("Segmento",
         f"{top_seg} genera el {seg.v[top_seg] / total * 100:.0f} % de las ventas porque tiene más clientes "
         f"({seg.n[top_seg]}). El gasto por cliente es parecido en los tres segmentos ({usd(seg.vxc.min())} a "
         f"{usd(seg.vxc.max())} en 4 años).",
         f"Para crecer, sumar clientes en {' y '.join(seg.drop(index=top_seg).index)}, que gastan lo mismo por "
         f"cliente pero son menos, en lugar de buscar subir el ticket."),
    ]
    return pd.DataFrame(filas, columns=["Factor", "Hallazgo", "Recomendación"])


def md(texto):
    """Muestra texto Markdown calculado (cifras que salen de los resultados del modelo)."""
    display(Markdown(texto.replace("$", r"\$")))


def kpis(items):
    """Fila de cifras destacadas: lista de (etiqueta, valor, detalle)."""
    celdas = "".join(
        f'<div style="flex:1 1 180px;border:1px solid #DCE6EE;border-radius:10px;padding:10px 14px;'
        f'background:{BRAND_LIGHT}">'
        f'<div style="font-size:.8rem;color:#5b6075">{label}</div>'
        f'<div style="font-size:1.45rem;font-weight:700;color:{BRAND}">{valor}</div>'
        f'<div style="font-size:.8rem;color:#5b6075">{detalle}</div></div>'
        for label, valor, detalle in items)
    display(HTML(f'<div style="display:flex;flex-wrap:wrap;gap:10px;margin:8px 0 16px">{celdas}</div>'))


def tabla(df):
    """Tabla sin índice, con un estilo uniforme."""
    return (df.style.hide(axis="index")
              .set_table_styles([{"selector": "th, td", "props": "padding: 5px 12px; text-align: left"}]))


def style_fig(fig, height=360, title=None):
    """Aplica el estilo común a cualquier figura de Plotly."""
    fig.update_layout(
        template="plotly_white",
        height=height,
        title=dict(text=title, font=dict(size=15, color=TEXT)) if title else None,
        font=dict(family="Nunito, Segoe UI, sans-serif", color=TEXT),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        colorway=COLORWAY,
        margin=dict(l=40, r=20, t=50 if title else 20, b=40),
        legend=dict(orientation="h", y=-0.2),
    )
    fig.update_xaxes(gridcolor=GRID, zeroline=False)
    fig.update_yaxes(gridcolor=GRID, zeroline=False)
    return fig


def show(fig, height=None):
    """Muestra una figura de Plotly en el libro.

    Se inserta como HTML sin plotly.js: el libro lo carga una sola vez (_config.yml → html_js_files).
    Así la figura es interactiva en el HTML generado sin repetir ~4 MB de librería por gráfica.
    """
    if height is not None:
        style_fig(fig, height)
    display(HTML(fig.to_html(include_plotlyjs=False, full_html=False, config={"displayModeBar": False})))


# ---------------------------------------------------------------- resultados de los experimentos
RES = ROOT / "experiments" / "resultados"   # tablas de experiments/analizar.py y robustez.py
ARTN = ROOT / "artifacts"                   # modelos guardados y metadata.json (experiments/artefactos.py)
ORIGINAL = "Perfil jerárquico (original)"
OPT = {"grid": "Grid Search", "random": "Random Search", "bayes": "Bayesiana (Optuna)", "genetico": "Genética (DEAP)",
       "-": "sin ajuste"}
CONJUNTOS = {"superstore": "Sample Superstore", "turismo": "Tourism Monthly", "hospital": "Hospital", "m3": "M3 Monthly"}


@lru_cache(maxsize=None)
def res(nombre: str):
    """Lee un archivo de experiments/resultados/ o de artifacts/ (CSV como DataFrame, JSON como dict)."""
    ruta = RES / nombre if (RES / nombre).exists() else ARTN / nombre
    if ruta.suffix == ".json":
        return json.loads(ruta.read_text(encoding="utf-8"))
    df = pd.read_csv(ruta)
    for col in ("Mes", "origen"):
        if col in df:
            df[col] = pd.to_datetime(df[col])
    return df


def lineas(x, series: dict, titulo, eje_x, eje_y, log_x=False, alto=340, referencia=None):
    """Figura de líneas con el estilo del libro. series: nombre -> valores."""
    import plotly.graph_objects as go
    fig = go.Figure()
    for nombre, y in series.items():
        fig.add_trace(go.Scatter(x=list(x), y=list(y), name=nombre, mode="lines+markers"))
    if referencia is not None:
        fig.add_hline(y=referencia[1], line_dash="dot", line_color=REF, annotation_text=referencia[0])
    fig.update_xaxes(title=eje_x, type="log" if log_x else "linear")
    fig.update_yaxes(title=eje_y)
    return style_fig(fig, alto, titulo)


def barras(nombres, valores, titulo, eje, alto=320, horizontal=True):
    """Barras ordenadas con el estilo del libro."""
    import plotly.graph_objects as go
    orden = sorted(range(len(valores)), key=lambda i: valores[i])
    n, v = [nombres[i] for i in orden], [valores[i] for i in orden]
    fig = go.Figure(go.Bar(x=v, y=n, orientation="h", marker_color=REAL) if horizontal else go.Bar(x=n, y=v, marker_color=REAL))
    (fig.update_xaxes if horizontal else fig.update_yaxes)(title=eje)
    return style_fig(fig, alto, titulo).update_layout(margin=dict(l=220 if horizontal else 40))


def diagnostico_regresion(modelo: str, optimizador: str):
    """Métricas y gráficos de un problema de regresión para los pronósticos del bucle externo en Sample Superstore:
    real frente a predicho, residuos frente a predicho, ACF de los residuos e histograma. Cada gráfico lleva
    antes su descripción (qué muestra y cómo se calcula) y después su lectura."""
    import numpy as np
    import plotly.graph_objects as go
    p = res("pred_superstore.csv")
    y, yh = p.Sales.to_numpy(), p[f"{modelo}|{optimizador}"].to_numpy()
    e = y - yh
    mae, rmse = np.abs(e).mean(), np.sqrt((e ** 2).mean())
    r2 = 1 - (e ** 2).sum() / ((y - y.mean()) ** 2).sum()
    wape = np.abs(e).sum() / y.sum() * 100
    kpis([("MAE", f"{mae:,.0f} USD", "error absoluto medio por serie y mes"), ("RMSE", f"{rmse:,.0f} USD", "castiga más los errores grandes"),
          ("R²", f"{r2:.3f}", "parte de la variación mensual que explica"), ("WAPE", f"{wape:.1f} %", "Σ|error| ÷ Σ ventas")])
    md(f"*Métricas sobre los {len(p)} pronósticos del bucle externo (12 series × 24 meses), con e = real − pronóstico: "
       f"MAE = promedio de |e|; RMSE = √(promedio de e²); R² = 1 − Σe² ÷ Σ(real − promedio real)²; WAPE = Σ|e| ÷ Σ real × 100.*")

    # 1. real frente a predicho
    md("**Valores reales frente a predichos.** *Cada punto es una serie en un mes: eje X = ventas reales, eje Y = pronóstico. "
       "La línea punteada es y = x: un pronóstico perfecto cae sobre ella.*")
    lim = [0, float(max(y.max(), yh.max())) * 1.05]
    fig = go.Figure(go.Scatter(x=y, y=yh, mode="markers", marker=dict(color=REAL, size=6, opacity=.6),
                               hovertemplate="real %{x:$,.0f}<br>pronóstico %{y:$,.0f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=lim, y=lim, mode="lines", line=dict(color=REF, dash="dot")))
    fig.update_xaxes(title="Ventas reales del mes (USD)", tickprefix="$", tickformat=",.0f")
    fig.update_yaxes(title="Pronóstico (USD)", tickprefix="$", tickformat=",.0f")
    show(style_fig(fig, 340).update_layout(showlegend=False))
    md(f"El error medio es de {mae:,.0f} USD y R² = {r2:.2f}. Los meses de ventas muy altas quedan por debajo de la "
       f"diagonal: el modelo se queda corto en los picos.")

    # 2. residuos frente a predichos
    md("**Residuos frente a predichos.** *Eje X = pronóstico; eje Y = residuo e = real − pronóstico. Si el modelo es adecuado "
       "la nube se reparte alrededor de 0 sin forma; un abanico indica que el error crece con el tamaño de la venta.*")
    fig = go.Figure(go.Scatter(x=yh, y=e, mode="markers", marker=dict(color=REAL, size=6, opacity=.6),
                               hovertemplate="pronóstico %{x:$,.0f}<br>residuo %{y:$,.0f}<extra></extra>"))
    fig.add_hline(y=0, line_dash="dot", line_color=REF)
    fig.update_xaxes(title="Pronóstico (USD)", tickprefix="$", tickformat=",.0f")
    fig.update_yaxes(title="Residuo = real − pronóstico (USD)", tickprefix="$", tickformat=",.0f")
    show(style_fig(fig, 340).update_layout(showlegend=False))
    rho = pd.Series(yh).rank().corr(pd.Series(np.abs(e)).rank())
    md(f"La correlación de Spearman entre el pronóstico y el tamaño del error (|e|) es {rho:.2f}: "
       + ("el error crece con la escala de la serie, lo que coincide con el rechazo de la prueba de White." if rho > 0.2 else
          "no se ve un abanico: el error casi no depende del tamaño del pronóstico."))

    # 3. ACF de los residuos del total mensual
    md("**Autocorrelación de los residuos (ACF).** *Se suman los residuos de las 12 series de cada mes y se calcula "
       "ρ(k) = Σ(eₜ − ē)(eₜ₋ₖ − ē) ÷ Σ(eₜ − ē)² para los rezagos k = 1…12. Las líneas punteadas son la banda ±1.96/√n: una "
       "barra fuera indica que el error se parece al de meses anteriores.*")
    tot = pd.Series(e).groupby(p.Mes.to_numpy()).sum().to_numpy()
    d = tot - tot.mean()
    k = np.arange(1, 13)
    acf = np.array([np.sum(d[i:] * d[:-i]) / np.sum(d * d) for i in k])
    banda = 1.96 / np.sqrt(len(tot))
    fig = go.Figure(go.Bar(x=k, y=acf, marker_color=[PROY if abs(v) > banda else REAL for v in acf],
                           hovertemplate="rezago %{x}: %{y:.2f}<extra></extra>"))
    fig.add_hline(y=banda, line_dash="dot", line_color=REF)
    fig.add_hline(y=-banda, line_dash="dot", line_color=REF)
    fig.update_xaxes(title="Rezago (meses)", dtick=1)
    fig.update_yaxes(title="Correlación", range=[-1, 1])
    show(style_fig(fig, 300))
    fuera = [int(i) for i, v in zip(k, acf) if abs(v) > banda]
    md(("Hay barras fuera de la banda (en ámbar) en los rezagos " + ", ".join(map(str, fuera)) + ": queda estructura temporal sin explicar."
        if fuera else f"Ninguna barra sale de la banda (±{banda:.2f}): los residuos no tienen autocorrelación visible."))

    # 4. histograma de residuos con la normal de igual media y desviación
    md("**Distribución de los residuos.** *Histograma de los residuos (eje recortado al 98 % central). La curva punteada es la "
       "normal con la misma media y desviación: si el histograma la sigue, los errores son aproximadamente normales.*")
    lo, hi = np.quantile(e, [.01, .99])
    xs = np.linspace(lo, hi, 200)
    mu, sd = e.mean(), e.std(ddof=1)
    fig = go.Figure(go.Histogram(x=e[(e >= lo) & (e <= hi)], nbinsx=30, histnorm="probability density", marker_color=REAL, name="Residuos"))
    fig.add_trace(go.Scatter(x=xs, y=np.exp(-.5 * ((xs - mu) / sd) ** 2) / (sd * np.sqrt(2 * np.pi)), mode="lines",
                             line=dict(color=PROY, dash="dash", width=2), name="Normal (misma media y sd)"))
    fig.update_xaxes(title="Residuo = real − pronóstico (USD)", tickprefix="$", tickformat=",.0f")
    fig.update_yaxes(title="Densidad")
    show(style_fig(fig, 340))
    from scipy import stats
    md(f"Asimetría de {stats.skew(e):.2f} y sesgo medio de {mu:+,.0f} USD (positivo: el modelo subestima). La cola derecha son los "
       f"meses de ventas excepcionales que ningún modelo anticipa.")
