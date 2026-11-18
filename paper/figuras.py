"""
paper/figuras.py — Figuras de resultados del artículo, a partir de los experimentos finales.

Lee experiments/resultados/ (validación anidada, pruebas, robustez) y artifacts/ (modelo final) y
escribe en paper/figures/ las figuras en PDF vectorial, con rótulos en inglés y fuente sans-serif.
Las dos figuras del análisis exploratorio (eda_overview.pdf, eda_index.pdf) no cambian y no se tocan.

Ejecutar desde la raíz, después de experiments/analizar.py, robustez.py y artefactos.py:
    python paper/figuras.py
"""
import sys
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
RES, ART, FIG = ROOT / "experiments" / "resultados", ROOT / "artifacts", ROOT / "paper" / "figures"
ORIG = "Perfil jerárquico (original)"
EN = {ORIG: "Hierarchical seasonal profile (proposed)", "Ingenuo estacional": "Seasonal naive", "Theta": "Theta",
      "HistGradientBoosting": "HistGradientBoosting", "KNN": "KNN", "Ridge": "Ridge", "Lasso": "Lasso",
      "Árbol de decisión": "Decision tree", "Random Forest": "Random forest", "XGBoost": "XGBoost", "SVR": "SVR",
      "Regresión bayesiana": "Bayesian ridge"}
ANCHO = 6.3                                                    # ancho útil de la página, en pulgadas
DS = {"superstore": "Superstore", "turismo": "Tourism", "hospital": "Hospital", "m3": "M3"}
AZUL, AMBAR, GRIS, CLARO = "#1F3A5F", "#E0A526", "#7A7A7A", "#C9D6E3"

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Latin Modern Sans", "DejaVu Sans"],
    "font.size": 12, "axes.labelsize": 13, "xtick.labelsize": 12, "ytick.labelsize": 12,
    "legend.fontsize": 12, "lines.linewidth": 2.0, "axes.grid": True, "grid.alpha": 0.3,
    "pdf.fonttype": 42, "savefig.bbox": "tight", "axes.spines.top": False, "axes.spines.right": False,
})


def mae_mensual(p, col):
    return p.assign(e=(p.Sales - p[col]).abs()).groupby("Mes").e.mean()


def main():
    FIG.mkdir(exist_ok=True)
    p = pd.read_csv(RES / "pred_superstore.csv", parse_dates=["Mes"])
    b = pd.read_csv(RES / "pred_bayesiana_superstore.csv", parse_dates=["Mes"])
    p = p.merge(b[["Category", "Region", "Mes"] + [c for c in b.columns if c.startswith("Regresión bayesiana")]],
                on=["Category", "Region", "Mes"], validate="one_to_one")
    mej = pd.read_csv(RES / "paper_resultados.csv")       # 11 referencias (experiments/paper_tablas.py)
    sup = mej[mej.conjunto == "superstore"].sort_values("MAE")
    col = dict(zip(sup.modelo, sup.columna))
    orden = list(sup.modelo)                                   # de menor a mayor MAE

    # 1. Diagrama del método
    fig, ax = plt.subplots(figsize=(ANCHO, 3.7)); ax.axis("off"); ax.set_xlim(0, 100); ax.set_ylim(0, 50)
    cajas = [(1, 31, "Monthly panel\nS series × T months (USD)"),
             (53, 31, "Scale-free index:\nmonthly sales ÷ mean\nof the previous 12 months"),
             (53, 3, "Pooled seasonal profile\n(12 values) + per-series\nweights, partial pooling"),
             (1, 3, "Back-transform to USD\nindex × 12-month mean")]
    for x, y, txt in cajas:
        ax.add_patch(FancyBboxPatch((x, y), 46, 15, boxstyle="round,pad=0.4,rounding_size=1.5", fc="#E6F2F4", ec=AZUL, lw=1.2))
        ax.text(x + 23, y + 7.5, txt, ha="center", va="center", fontsize=10.5)
    flecha = {"arrowstyle": "-|>", "color": AZUL, "lw": 1.8}
    ax.annotate("", xy=(52, 38), xytext=(48, 38), arrowprops=flecha)
    ax.annotate("", xy=(76, 18), xytext=(76, 30), arrowprops=flecha)
    ax.annotate("", xy=(48, 10), xytext=(52, 10), arrowprops=flecha)
    ax.annotate("", xy=(20, 30), xytext=(20, 18), arrowprops={"arrowstyle": "-|>", "color": AMBAR, "lw": 1.8, "ls": "--"})
    ax.text(22, 24, "recursive step: the forecast\nof month t becomes history\nfor month t + 1", ha="left", va="center", fontsize=11)
    fig.savefig(FIG / "method_diagram.pdf"); plt.close(fig)

    # 2. MAE por bloque mensual (mejor combinación de cada modelo)
    ordv = orden[::-1]
    mm = {n: mae_mensual(p, col[n]) / 1000 for n in orden}
    fig, ax = plt.subplots(figsize=(ANCHO, 5.0))
    bp = ax.boxplot([mm[n] for n in ordv], vert=False, widths=0.6, patch_artist=True,
                    medianprops={"color": "black", "linewidth": 1.5}, flierprops={"marker": "o", "markersize": 4})
    for patch, n in zip(bp["boxes"], ordv):
        patch.set_facecolor(AMBAR if n == ORIG else CLARO)
    ax.scatter([mm[n].mean() for n in ordv], range(1, len(ordv) + 1), marker="D", color=AZUL, zorder=3, s=28, label="Mean")
    ax.set_yticks(range(1, len(ordv) + 1)); ax.set_yticklabels([EN[n] for n in ordv])
    ax.set_xlabel("Monthly-block MAE, 24 rolling origins\n(thousand USD)")
    ax.legend(frameon=False, loc="lower right", bbox_to_anchor=(1.0, 1.0), borderaxespad=0.1)
    fig.savefig(FIG / "mae_boxplot.pdf"); plt.close(fig)

    # 3. Ventas totales: observado, original a un paso, original recursivo 2017, ingenuo estacional
    pm = p.groupby("Mes").agg(real=("Sales", "sum"), pred=(col[ORIG], "sum"), ref=(col["Ingenuo estacional"], "sum")).reset_index()
    rec = pd.read_csv(ROOT / "model" / "artifacts" / "prueba_recursiva_2017.csv", parse_dates=["Mes"]).groupby("Mes").Proyeccion.sum()
    fig, ax = plt.subplots(figsize=(ANCHO, 4.2))
    ax.plot(pm.Mes, pm.real / 1000, color="black", marker="o", ms=4, label="Observed")
    ax.plot(pm.Mes, pm.pred / 1000, color=AZUL, ls="--", marker="s", ms=4, label="Proposed, one step ahead")
    ax.plot(rec.index, rec.values / 1000, color=AMBAR, ls="-.", marker="^", ms=4, label="Proposed, 12-month recursive (2017)")
    ax.plot(pm.Mes, pm.ref / 1000, color=GRIS, ls=":", marker="x", ms=4, label="Seasonal naive")
    ax.set_xticks(pm.Mes[::4]); ax.set_xticklabels([f"{m:%b}\n{m:%Y}" for m in pm.Mes[::4]])
    ax.set_xlabel("Month"); ax.set_ylabel("Total sales (thousand USD)")
    ax.set_ylim(0, 190); ax.legend(frameon=False, loc="upper left")
    fig.savefig(FIG / "forecast_total.pdf"); plt.close(fig)

    # 4. Diferencia de MAE frente a cada referencia, con IC del bootstrap estacionario
    s = pd.read_csv(RES / "paper_pruebas.csv")
    s = s[s.conjunto == "superstore"].sort_values("dif", ascending=False)
    fig, ax = plt.subplots(figsize=(ANCHO, 4.6))
    y = np.arange(len(s))
    ax.errorbar(s.dif, y, xerr=[s.dif - s.ic_lo, s.ic_hi - s.dif], fmt="o", color=AZUL, capsize=4, lw=2)
    ax.axvline(0, color="black", lw=1)
    ax.set_yticks(y); ax.set_yticklabels([EN[n] for n in s.referencia])
    ax.set_xlabel("MAE difference, proposed − baseline (USD)\nnegative values favour the proposed model")
    fig.savefig(FIG / "mae_differences.pdf"); plt.close(fig)

    # 5. Predicho frente a observado y residuos del modelo propuesto
    real, pr = p.Sales.to_numpy() / 1000, p[col[ORIG]].to_numpy() / 1000
    fig, ax = plt.subplots(1, 2, figsize=(ANCHO, 3.2))
    ax[0].scatter(real, pr, s=12, color=AZUL, alpha=0.55, lw=0)
    lim = max(real.max(), pr.max()) * 1.05
    ax[0].plot([0, lim], [0, lim], color=GRIS, lw=1, ls="--")
    ax[0].set_xlim(0, lim); ax[0].set_ylim(0, lim)
    ax[0].set_xlabel("Observed (thousand USD)"); ax[0].set_ylabel("Predicted (thousand USD)")
    ax[1].hist(real - pr, bins=24, color=AZUL, alpha=0.85)
    ax[1].axvline(0, color="black", lw=1)
    ax[1].set_xlabel("Residual (thousand USD)"); ax[1].set_ylabel("Series-months")
    fig.tight_layout(); fig.savefig(FIG / "pred_vs_obs.pdf"); plt.close(fig)

    # 6. Sensibilidad a lambda y a la cantidad de datos
    sl = pd.read_csv(RES / "sensibilidad_lam.csv")
    sl = sl[sl.conjunto == "superstore"].sort_values("lam")
    mu = pd.read_csv(RES / "robustez_muestra.csv")
    fig, ax = plt.subplots(1, 2, figsize=(ANCHO, 3.6))
    ax[0].plot(sl.lam, sl.MAE / 1000, color=AZUL, marker="o", ms=4, label="Proposed, fixed λ")
    ax[0].axhline(sl.MAE_mejor_referencia.iloc[0] / 1000, color=GRIS, ls=":", label="Best baseline (SVR)")
    ax[0].set_xscale("log"); ax[0].set_xlabel("Pooling strength λ"); ax[0].set_ylabel("Rolling-origin MAE\n(thousand USD)")
    ax[0].legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    g = mu[mu.estudio == "series"]
    for modelo, c, ls, mk in [(ORIG, AMBAR, "-", "o"), ("SVR", AZUL, "--", "s"), ("Random Forest", GRIS, ":", "^")]:
        h = g[g.modelo == modelo].assign(n=lambda d: d.nivel.astype(int)).sort_values("n")
        ax[1].plot(h.n, h.MAE / 1000, color=c, ls=ls, marker=mk, ms=5, label={ORIG: "Proposed", "SVR": "SVR", "Random Forest": "Random forest"}[modelo])
    ax[1].set_xticks([3, 6, 12]); ax[1].set_xlabel("Series in training")
    ax[1].set_ylabel("MAE on those series\n(thousand USD)")
    ax[1].set_ylim(top=ax[1].get_ylim()[1] + 0.75)               # deja la franja de arriba para la leyenda
    ax[1].legend(frameon=False, loc="upper right")
    fig.tight_layout(); fig.savefig(FIG / "sensitivity.pdf"); plt.close(fig)

    # 7. Interpretación: perfil estacional y error por serie frente a la mejor referencia
    modelo = joblib.load(ART / "perfil_jerarquico_original.joblib")
    perfil = np.r_[modelo.perfil_[1:], modelo.perfil_[0]]      # enero ... diciembre
    ps = pd.read_csv(RES / "por_serie.csv")
    fig, ax = plt.subplots(1, 2, figsize=(ANCHO, 3.4))
    ax[0].bar(range(1, 13), perfil, color=AZUL)
    ax[0].axhline(1, color=GRIS, lw=1, ls="--")
    ax[0].set_xticks(range(1, 13)); ax[0].set_xticklabels(list("JFMAMJJASOND"))
    ax[0].set_xlabel("Month of the year"); ax[0].set_ylabel("Seasonal profile $P_m$")
    ax[1].scatter(ps.venta_media / 1000, ps.MAE / ps.venta_media * 100, color=AMBAR, label="Proposed", zorder=3)
    ax[1].scatter(ps.venta_media / 1000, ps.MAE_ref / ps.venta_media * 100, color=AZUL, marker="s", label="SVR")
    ax[1].set_xlabel("Series mean monthly\nsales (thousand USD)"); ax[1].set_ylabel("Relative MAE (%)")
    ax[1].legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, handletextpad=0.2, columnspacing=0.8)
    fig.tight_layout(); fig.savefig(FIG / "profile_errors.pdf"); plt.close(fig)

    # 8. MASE por conjunto de datos (todos los modelos con búsqueda bayesiana o sin ajuste)
    li = pd.read_csv(RES / "paper_lideres.csv")
    li = li[li.optimizador.isin(["bayes", "-"])]
    fig, ax = plt.subplots(figsize=(ANCHO, 3.8))
    for j, (c, nombre) in enumerate(DS.items()):
        g = li[li.conjunto == c]
        otros = g[g.modelo != ORIG]
        ax.scatter(np.full(len(otros), j) + np.linspace(-0.18, 0.18, len(otros)), otros.MASE, color=CLARO, edgecolor=AZUL, s=36,
                   label="Baselines" if j == 0 else None)
        ax.scatter([j], g[g.modelo == ORIG].MASE, color=AMBAR, edgecolor="black", s=90, marker="D", zorder=3,
                   label="Proposed" if j == 0 else None)
    ax.axhline(1, color=GRIS, ls=":", lw=1)
    ax.set_xticks(range(len(DS))); ax.set_xticklabels(DS.values())
    ax.set_ylabel("MASE (lower is better)"); ax.legend(frameon=False, loc="upper left")
    fig.savefig(FIG / "datasets.pdf"); plt.close(fig)

    viejas = ["ablation.pdf", "importance_errors.pdf"]
    for v in viejas:
        (FIG / v).unlink(missing_ok=True)
    print("figuras:", sorted(f.name for f in FIG.glob("*.pdf")))


if __name__ == "__main__":
    main()
