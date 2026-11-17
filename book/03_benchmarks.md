---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Modelos de referencia

*Siete modelos, cuatro métodos de optimización y una validación que no mira el futuro*

Este capítulo reporta las 28 combinaciones de modelo y método de optimización, cómo se eligió la mejor
de cada modelo y qué tan confiables son. El código que las produce está en `experiments/ejecutar.py`
(cómputo) y `experiments/analizar.py` (tablas); aquí se leen sus resultados guardados, porque
repetirlos tarda unas dos horas.

## Modelos

Los siete modelos predicen el índice de ventas con las mismas siete variables, dentro de un `Pipeline`
con codificación one-hot de la categoría y la región y estandarización de las numéricas.

| Modelo | Idea | Hiperparámetros que se ajustan |
|---|---|---|
| KNN | Promedia los meses más parecidos del historial | número de vecinos (3 a 30), pesos |
| Ridge | Regresión lineal con penalización L2 | `alpha` (0,01 a 100, escala log) |
| Lasso | Regresión lineal con penalización L1 | `alpha` (0,0001 a 0,1, escala log) |
| Árbol de decisión | Reglas sí/no sobre las variables | profundidad (2 a 6), hojas mínimas (1 a 20) |
| Random Forest | Promedio de 300 árboles | profundidad, hojas mínimas, fracción de variables |
| XGBoost | 300 árboles que corrigen el error del anterior | profundidad, tasa de aprendizaje, submuestra, peso mínimo |
| SVR | Margen de tolerancia ε con kernel RBF | `C`, `epsilon`, `gamma` |

Como referencias adicionales se incluyen el **ingenuo estacional** (repetir el mismo mes del año
anterior), el **método Theta**, que es la referencia estándar de las competencias de pronóstico, y
**HistGradientBoosting**, el boosting por histogramas de la familia de LightGBM.

Los rangos se definieron alrededor de los valores habituales de cada modelo y en escala logarítmica
donde el efecto es multiplicativo (`alpha`, `C`, `gamma`, tasa de aprendizaje). Con 144 a 420 filas
de entrenamiento no tiene sentido explorar árboles profundos ni cientos de vecinos.

## Validación anidada

El error que se reporta nunca sale de los mismos datos con los que se eligieron los hiperparámetros.

- **Bucle externo (estima el error).** Origen móvil: cada uno de los 24 meses de 2016 y 2017 se
  pronostica entrenando con todos los meses anteriores. Son 288 pronósticos por modelo.
- **Bucle interno (elige hiperparámetros).** Dentro de cada entrenamiento, validación cruzada temporal
  con pliegues de 3 meses, hasta 4 pliegues. El preprocesamiento se ajusta en cada pliegue.

Con series de tiempo no se pueden barajar los meses: un pliegue aleatorio entrenaría con el futuro.
Por eso los dos bucles respetan el orden temporal.

## Métodos de optimización

| Método | Cómo busca | Configuración |
|---|---|---|
| Grid Search | Todas las combinaciones de una rejilla | entre 4 y 24 combinaciones según el modelo |
| Random Search | Puntos al azar del espacio | 16 evaluaciones |
| Bayesiana (Optuna) | Modelo sustituto **TPE**: estima la densidad de las configuraciones buenas, *l(x)*, y de las malas, *g(x)*. Función de adquisición: **mejora esperada**, que en TPE equivale a elegir el punto con mayor *l(x)/g(x)* | 16 evaluaciones, las 6 primeras al azar (exploración) |
| Genética (DEAP) | Población que evoluciona. Selección por **torneo** de tamaño 2; cruce por mezcla con probabilidad 0,7; mutación gaussiana (σ = 0,2) con probabilidad 0,3; **elitismo** de 1 individuo | población de 4, 4 generaciones: 16 evaluaciones |

Todos los modelos reciben el mismo presupuesto en los tres métodos de 16 evaluaciones; Grid Search
recorre su rejilla completa. La implementación está en `src/optimizacion.py`: los cuatro métodos
comparten una misma función objetivo, el MAE medio de validación cruzada del `Pipeline` completo.

## Resultados de las 28 combinaciones

```{code-cell} ipython3
:tags: [hide-input]
from common import CONJUNTOS, OPT, ORIGINAL, md, res, show, tabla, usd, lineas
import pandas as pd

lid = res("lideres.csv")
lid = lid[lid.conjunto == "superstore"]
base = lid[~lid.modelo.isin([ORIGINAL, "Ingenuo estacional", "Theta", "HistGradientBoosting"])]
# MAE del bucle externo: una fila por modelo, una columna por método de optimización
piv = base.pivot(index="modelo", columns="optimizador", values="MAE")[["grid", "random", "bayes", "genetico"]]
t = piv.round(0).astype(int).rename(columns=OPT)
t["Mejor método"] = piv.idxmin(axis=1).map(OPT)
tabla(t.reset_index().rename(columns={"modelo": "Modelo"}).sort_values(t.columns[0]))
```

MAE en dólares por serie y mes, promedio de los 24 meses evaluados. La diferencia entre métodos dentro
de un mismo modelo es pequeña, casi siempre menor que la diferencia entre modelos.

```{code-cell} ipython3
:tags: [hide-input]
mej = res("mejores.csv")
ms = mej[(mej.conjunto == "superstore") & (mej.modelo != ORIGINAL)].sort_values("MAE")
tabla(pd.DataFrame({
    "Modelo": ms.modelo, "Método": ms.optimizador.map(OPT),
    "MAE (USD), media ± sd": [f"{a:,.0f} ± {b:,.0f}" for a, b in zip(ms.MAE, ms.MAE_sd)],
    "IC 95 %": [f"[{a:,.0f}, {b:,.0f}]" for a, b in zip(ms.MAE_ic_lo, ms.MAE_ic_hi)],
    "RMSE (USD)": ms.RMSE.map("{:,.0f}".format), "MASE": ms.MASE.map("{:.3f}".format),
    "Desvío del total": ms.desvio_total_pct.map("{:+.1f} %".format),
    "Ajuste (ms)": ms.ajuste_ms.map(lambda v: "—" if pd.isna(v) else f"{v:.1f}"),
    "Inferencia (ms)": ms.inferencia_ms.map(lambda v: "—" if pd.isna(v) else f"{v:.2f}")}))
```

La tabla muestra la mejor combinación de cada modelo, elegida por su MAE medio en el bucle externo.
La desviación estándar es la del MAE entre los 24 meses y el intervalo sale de un bootstrap
estacionario. **SVR con Grid Search es la mejor referencia**, pero subestima el total en 18 %: acierta
más meses corrientes a costa de quedarse corto en los picos. Random Forest es casi igual de preciso y
mantiene el total.

## Comparación de los métodos de optimización

```{code-cell} ipython3
:tags: [hide-input]
o = res("optimizadores.csv")
tabla(pd.DataFrame({"Método": o.optimizador.map(OPT), "MAE medio de los 8 modelos (USD)": o.MAE_medio.round(0).astype(int),
                    "Evaluaciones por búsqueda": o.evaluaciones.round(1), "Segundos por búsqueda": o.busqueda_s.round(1),
                    "Modelos en que fue el mejor": o.veces_mejor.astype(int)}))
```

Los cuatro métodos llegan a errores casi iguales: entre el mejor y el peor hay menos de 13 dólares de
MAE medio, frente a una desviación entre meses de unos 850. La búsqueda bayesiana obtiene el menor
error medio con el mismo costo que la aleatoria. La genética tarda ocho veces más por búsqueda, por
el costo fijo de la librería con poblaciones tan pequeñas, sin mejorar el resultado. Con espacios de
uno a cuatro hiperparámetros y 16 evaluaciones, ningún método tiene margen para destacar: **no hay
evidencia de que los métodos guiados sean más eficientes en este problema**.

```{code-cell} ipython3
:tags: [hide-input]
c = res("anytime.csv")
c = c[c.modelo == "Random Forest"]
n = int(c.groupby("optimizador").evaluacion.max().min())
show(lineas(range(1, n + 1), {OPT[k]: g.sort_values("evaluacion").mejor_MAE_interno.to_numpy()[:n] for k, g in c.groupby("optimizador")},
            "Convergencia de la búsqueda en Random Forest", "Evaluación", "Mejor MAE interno hasta el momento (índice)"))
```

La curva muestra el mejor valor encontrado hasta cada evaluación, promediado sobre los 24 orígenes.
Entre la primera y la última evaluación el error interno mejora menos de 2 % en los cuatro métodos;
la búsqueda bayesiana termina con el menor valor, pero por un margen muy pequeño.

## Complejidad y costo

| Modelo | Entrenamiento | Inferencia (por fila) |
|---|---|---|
| KNN | O(1): guarda los datos | O(n·p) |
| Ridge | O(n·p² + p³) | O(p) |
| Lasso | O(n·p) por iteración | O(p) |
| Árbol de decisión | O(n·p·log n) | O(profundidad) |
| Random Forest | O(B·n·p·log n), B = 300 árboles | O(B·profundidad) |
| XGBoost (histogramas) | O(B·n·p) | O(B·profundidad) |
| SVR | entre O(n²·p) y O(n³) | O(vectores de soporte · p) |
| Modelo original (capítulo 5) | O(n) + O(S) por iteración | O(1) |

*n* filas, *p* variables, *S* series. Los tiempos medidos (tabla anterior) siguen ese orden: los
modelos lineales y el árbol se ajustan en pocos milisegundos y Random Forest en cientos. XGBoost usa
el método de histogramas (`tree_method="hist"`). Los orígenes del bucle externo se ejecutan en
paralelo con `joblib` (8 procesos).

## Residuos

```{code-cell} ipython3
:tags: [hide-input]
r = res("residuos.csv")
r = r[r.conjunto == "superstore"].merge(ms[["modelo", "MAE"]], on="modelo", how="inner").sort_values("MAE")
tabla(pd.DataFrame({"Modelo": r.modelo, "White (p)": r.p_white.map("{:.3f}".format), "Ljung–Box (p)": r.p_ljung_box.map("{:.3f}".format),
                    "BDS (p)": r.p_bds.map(lambda v: "—" if pd.isna(v) else f"{v:.3f}"), "Shapiro–Wilk (p)": r.p_shapiro.map("{:.3f}".format),
                    "Sesgo medio (USD)": r.sesgo_medio.map("{:+,.0f}".format), "Asimetría": r.asimetria.map("{:.2f}".format)}))
```

- **White (homocedasticidad):** se rechaza en casi todos los modelos. El error crece con el tamaño de
  la venta, lo esperable cuando las series tienen escalas distintas.
- **Ljung–Box (autocorrelación, 6 rezagos):** no se rechaza en ninguno. No queda estructura lineal en
  el tiempo sin explicar.
- **BDS (independencia):** se rechaza en la mayoría: queda dependencia no lineal, asociada a los
  meses de ventas excepcionales.
- **Sesgo:** todos los modelos subestiman (sesgo positivo). SVR y el ingenuo estacional, mucho más.

### Diagnóstico gráfico de la mejor referencia

Además de las pruebas, los supuestos de un modelo de regresión se revisan con gráficos. Aquí, para la
mejor referencia por MAE; el del modelo original está en el capítulo 5.

```{code-cell} ipython3
:tags: [hide-input]
from common import diagnostico_regresion
diagnostico_regresion(ms.iloc[0].modelo, ms.iloc[0].optimizador)
```

## Interpretabilidad

Con siete variables hay 128 coaliciones posibles, así que los valores de Shapley se calculan de forma
exacta por enumeración (`src/explicacion.py`), sin la aproximación por muestreo.

```{code-cell} ipython3
:tags: [hide-input]
from common import barras
import sys
from model.features import NOMBRES
s = res("shapley_global.csv")
mejor_ref = ms.iloc[0].modelo
g = s[s.modelo == mejor_ref]
show(barras([NOMBRES[v] for v in g.variable], list(g.importancia), f"Importancia global en {mejor_ref}",
            "Media del valor absoluto de Shapley (índice)"))
```

En la mejor referencia, igual que en los demás modelos, la predicción depende sobre todo del mes del
año y de las ventas del mismo mes del año anterior. La categoría, la región y el mes previo aportan
poco. Esa observación es el punto de partida del modelo original.

**Shapley frente a LIME.** Shapley reparte la diferencia entre la predicción y la predicción media
entre las variables, y los aportes suman exactamente esa diferencia. LIME ajusta una regresión lineal
ponderada sobre perturbaciones de la fila: es más rápido, pero sus coeficientes dependen del ancho
del núcleo y del muestreo, y no tienen por qué sumar la predicción. Las dos explicaciones divergen
cuando las variables interactúan o están correlacionadas, como aquí el rezago de un mes y el promedio
de tres meses. `src/explicacion.py` incluye las dos; el tablero usa Shapley.
