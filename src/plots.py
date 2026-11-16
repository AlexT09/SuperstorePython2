"""
src/plots.py — Funciones que construyen las figuras de las páginas EDA y Modelos.

Todas devuelven una figura de Plotly con el tema común de la app (common.style_fig): título, ejes
con unidades y la misma paleta (un color = un significado).
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from common import CATEGORY_COLORS, DIV_SCALE, NEG, POS, PROY, REAL, REF, style_fig

USD = dict(tickprefix="$", tickformat=",.0f")


def acf(x, n_rezagos=12):
    """Autocorrelación muestral de los rezagos 1..n."""
    x = np.asarray(x, float) - np.mean(x)
    den = np.sum(x * x)
    return np.array([np.sum(x[k:] * x[:-k]) / den for k in range(1, min(n_rezagos, len(x) - 2) + 1)])


def pacf(x, n_rezagos=12):
    """Autocorrelación parcial por Durbin–Levinson."""
    r = np.r_[1.0, acf(x, n_rezagos)]
    n = len(r) - 1
    phi, out = np.zeros((n + 1, n + 1)), []
    for k in range(1, n + 1):
        num = r[k] - np.sum(phi[k - 1, 1:k] * r[1:k][::-1])
        den = 1 - np.sum(phi[k - 1, 1:k] * r[1:k])
        phi[k, k] = num / den if den else 0.0
        phi[k, 1:k] = phi[k - 1, 1:k] - phi[k, k] * phi[k - 1, 1:k][::-1]
        out.append(phi[k, k])
    return np.array(out)


def _barras_correlacion(valores, n, titulo, alto=280):
    banda = 1.96 / np.sqrt(n)
    k = np.arange(1, len(valores) + 1)
    fig = go.Figure(go.Bar(x=k, y=valores, marker_color=[PROY if abs(v) > banda else REAL for v in valores],
                           hovertemplate="rezago %{x}: %{y:.2f}<extra></extra>"))
    fig.add_hline(y=banda, line_dash="dot", line_color=REF)
    fig.add_hline(y=-banda, line_dash="dot", line_color=REF)
    fig.update_xaxes(title="Rezago (meses)", dtick=1)
    fig.update_yaxes(title="Correlación", range=[-1, 1])
    return style_fig(fig, alto, titulo)


def fig_acf(x, titulo="Autocorrelación (ACF)", n_rezagos=12):
    return _barras_correlacion(acf(x, n_rezagos), len(x), titulo)


def fig_pacf(x, titulo="Autocorrelación parcial (PACF)", n_rezagos=12):
    return _barras_correlacion(pacf(x, n_rezagos), len(x), titulo)


def descomponer(serie: pd.Series):
    """Descomposición clásica multiplicativa: tendencia (media móvil centrada de 12), estacional y resto."""
    tendencia = serie.rolling(12, center=True).mean().rolling(2).mean().shift(-1)
    razon = serie / tendencia
    estacional = razon.groupby(razon.index.month).transform("mean")
    estacional = estacional / estacional.groupby(estacional.index.year).transform("mean").mean()
    return tendencia, estacional, serie / (tendencia * estacional)


def fig_descomposicion(serie: pd.Series):
    from plotly.subplots import make_subplots
    t, e, r = descomponer(serie)
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.04,
                        subplot_titles=["Ventas (USD)", "Tendencia (USD)", "Estacional (índice)", "Resto (índice)"])
    for i, (s, c) in enumerate([(serie, REAL), (t, PROY), (e, POS), (r, REF)], 1):
        fig.add_trace(go.Scatter(x=s.index, y=s.values, line=dict(color=c, width=2), showlegend=False), row=i, col=1)
    return style_fig(fig, 560, "Descomposición de las ventas mensuales")


def fig_distribucion(valores, titulo, eje, prefijo=""):
    fig = go.Figure(go.Histogram(x=valores, nbinsx=30, marker_color=REAL))
    fig.add_vline(x=float(np.median(valores)), line_dash="dot", line_color=PROY, annotation_text="mediana")
    fig.update_xaxes(title=eje, tickprefix=prefijo)
    fig.update_yaxes(title="Frecuencia")
    return style_fig(fig, 300, titulo)


def fig_correlacion(df: pd.DataFrame):
    c = df.corr(method="spearman")
    fig = go.Figure(go.Heatmap(z=c.values, x=c.columns, y=c.index, zmin=-1, zmax=1, colorscale=DIV_SCALE,
                               text=c.round(2).values, texttemplate="%{text}", colorbar=dict(title="ρ")))
    return style_fig(fig, 340, "Correlación de Spearman")


def fig_explorador(df: pd.DataFrame, x, y, color):
    fig = go.Figure()
    grupos = df.groupby(color) if color else [("Todos", df)]
    for nombre, g in grupos:
        fig.add_trace(go.Scatter(x=g[x], y=g[y], mode="markers", name=str(nombre),
                                 marker=dict(size=5, opacity=0.55, color=CATEGORY_COLORS.get(nombre))))
    fig.update_xaxes(title=x)
    fig.update_yaxes(title=y)
    return style_fig(fig, 400, f"{y} frente a {x}")


# ---------------------------------------------------------------- modelos
def fig_real_vs_pred(real, pred, titulo):
    lim = [0, float(max(real.max(), pred.max())) * 1.05]
    fig = go.Figure(go.Scatter(x=real, y=pred, mode="markers", marker=dict(color=REAL, size=6, opacity=0.6),
                               hovertemplate="real %{x:$,.0f}<br>pronóstico %{y:$,.0f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=lim, y=lim, mode="lines", line=dict(color=REF, dash="dot"), showlegend=False))
    fig.update_xaxes(title="Ventas reales del mes (USD)", **USD)
    fig.update_yaxes(title="Pronóstico (USD)", **USD)
    return style_fig(fig, 340, titulo).update_layout(showlegend=False)


def fig_residuos(meses, residuo_total):
    fig = go.Figure(go.Bar(x=meses, y=residuo_total, marker_color=[NEG if v < 0 else POS for v in residuo_total],
                           hovertemplate="%{x|%b %Y}: %{y:$,.0f}<extra></extra>"))
    fig.update_yaxes(title="Real − pronóstico, total del mes (USD)", **USD)
    return style_fig(fig, 300, "Residuo del total mensual")


def fig_barras_h(nombres, valores, titulo, eje, color=REAL, alto=300):
    orden = np.argsort(valores)
    fig = go.Figure(go.Bar(x=np.asarray(valores)[orden], y=np.asarray(nombres)[orden], orientation="h", marker_color=color))
    fig.update_xaxes(title=eje)
    return style_fig(fig, alto, titulo).update_layout(margin=dict(l=230))


def fig_cascada(etiquetas, aportes, base, titulo):
    """Explicación local: de la predicción base a la final, sumando el aporte de cada variable."""
    fig = go.Figure(go.Waterfall(
        orientation="h", measure=["absolute"] + ["relative"] * len(aportes) + ["total"],
        y=["Base (predicción media)"] + list(etiquetas) + ["Predicción"], x=[base] + list(aportes) + [0],
        increasing=dict(marker_color=POS), decreasing=dict(marker_color=NEG), totals=dict(marker_color=PROY),
        connector=dict(line=dict(color=REF, width=1)), hovertemplate="%{y}: %{x:$,.0f}<extra></extra>"))
    fig.update_xaxes(title="Ventas del mes (USD)", **USD)
    fig.update_yaxes(autorange="reversed")
    return style_fig(fig, 60 + 34 * (len(aportes) + 2), titulo).update_layout(margin=dict(l=300), showlegend=False)


def fig_lineas(x, series: dict, titulo, eje_x, eje_y, log_x=False, alto=320, referencia=None):
    fig = go.Figure()
    colores = [REAL, PROY, POS, "#b89cf0", NEG, REF]
    for (nombre, y), c in zip(series.items(), colores):
        fig.add_trace(go.Scatter(x=x, y=y, name=nombre, mode="lines+markers", line=dict(color=c, width=2), marker_size=5))
    if referencia is not None:
        fig.add_hline(y=referencia[1], line_dash="dot", line_color=REF, annotation_text=referencia[0])
    fig.update_xaxes(title=eje_x, type="log" if log_x else "linear")
    fig.update_yaxes(title=eje_y)
    return style_fig(fig, alto, titulo)
