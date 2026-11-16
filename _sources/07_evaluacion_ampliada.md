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

# Evaluación ampliada y robustez

*¿Se mantiene el resultado fuera del conjunto principal y en condiciones menos favorables?*

Un buen resultado en un solo conjunto no basta. Este capítulo evalúa el modelo original y las
referencias en tres conjuntos adicionales y somete al modelo a ruido, menos datos y otras semillas.
Todo se fijó en el protocolo antes de experimentar; los estudios de robustez son exploratorios.

## Conjuntos de datos adicionales

| Conjunto | Fuente | Series usadas | Periodo | Naturaleza |
|---|---|---|---|---|
| Tourism Monthly | Monash / Zenodo, doi:10.5281/zenodo.4656096 | 24 de 366 | oct. 2003 – sep. 2007 | Demanda turística; estacionalidad muy fuerte |
| Hospital | Monash / Zenodo, doi:10.5281/zenodo.4656014 | 24 de 767 | ene. 2003 – dic. 2006 | Conteos de pacientes; estacionalidad débil |
| M3 Monthly | Monash / Zenodo, doi:10.5281/zenodo.4656298 | 24 de 1.428 | mar. 1990 – feb. 1994 | Series de seis dominios sin relación entre sí |

Provienen del repositorio de Monash {cite}`godahewa2021`, con licencia CC BY 4.0. Se eligieron por
ser públicos, mensuales, de la misma tarea y de naturaleza distinta, **sin ajustar ningún modelo
sobre ellos**. M3 se incluyó a propósito como caso adverso: sus series no comparten estacionalidad,
que es el supuesto del modelo. De cada conjunto se tomaron 24 series al azar (semilla 42) entre las
que tenían calendario común y los últimos 48 meses completos y positivos; el código está en
`experiments/datos_adicionales.py`.

En cada conjunto se aplicó el mismo procedimiento: 24 meses de origen móvil, validación cruzada
temporal interna y búsqueda bayesiana con 16 evaluaciones para todos los modelos.

```{code-cell} ipython3
:tags: [hide-input]
from common import CONJUNTOS, OPT, ORIGINAL, barras, lineas, md, res, show, tabla, usd
import pandas as pd

mej, n = res("mejores.csv"), res("numeros.json")
# MASE de cada modelo en cada conjunto (error dividido por el del ingenuo estacional antes del primer origen)
piv = mej.pivot_table(index="modelo", columns="conjunto", values="MASE")[list(CONJUNTOS)]
piv["Rango medio"] = piv.rank().mean(axis=1)
t = piv.sort_values("Rango medio").round(3).rename(columns=CONJUNTOS).reset_index().rename(columns={"modelo": "Modelo"})
tabla(t)
```

MASE por conjunto (menor es mejor; por debajo de 1 mejora al ingenuo estacional de referencia). En
Superstore cada modelo aparece con su mejor método de optimización; en los demás, con la búsqueda
bayesiana. No se promedian errores de escalas distintas: el rango medio solo ordena.

```{code-cell} ipython3
:tags: [hide-input]
p = res("pruebas.csv")
filas = []
for c, nombre in CONJUNTOS.items():
    q = p[p.conjunto == c]
    q = q.loc[q.MAE_referencia.idxmin()]
    filas.append({"Conjunto": nombre, "Posición del original": f"{n[c]['posicion_original']} de 11",
                  "Mejor referencia": f"{q.referencia} ({OPT[q.optimizador]})",
                  "Original − mejor referencia [IC 95 %]": f"{q.dif:+,.1f} [{q.ic_lo:+,.1f}, {q.ic_hi:+,.1f}]",
                  "Diebold–Mariano p / p Holm": f"{q.p_DM:.3f} / {q.p_DM_holm:.3f}",
                  "Modelos en el conjunto de confianza": f"{n[c]['mcs_tamano']} de {n[c]['mcs_total']}",
                  "¿El original está?": "Sí" if n[c]["original_en_mcs"] else "No"})
tabla(pd.DataFrame(filas))
```

- **Tourism Monthly.** El original queda tercero, a 6 unidades del primero y sin diferencia
  significativa. El conjunto de confianza retiene cinco modelos, entre ellos el original, y descarta
  a los ensambles de árboles y a los modelos lineales. Es el tipo de panel para el que el modelo fue
  pensado.
- **Hospital.** Queda noveno. Con estacionalidad débil el perfil común aporta poco, y los modelos que
  usan el mes anterior y el promedio de tres meses lo superan. Aun así, ningún modelo se distingue de
  otro: los once quedan en el conjunto de confianza.
- **M3 Monthly.** Queda décimo y es **significativamente peor** que el método Theta. Las series no
  comparten patrón estacional, de modo que el perfil común es un promedio sin sentido. Es el fallo
  previsto, y confirma el límite del modelo.

```{code-cell} ipython3
:tags: [hide-input]
e = n["entre_conjuntos"]
r = pd.Series(e["rango_medio"]).sort_values()
md(f"**Prueba entre conjuntos.** Friedman con la corrección de Iman–Davenport sobre el MASE de los 11 modelos "
   f"(todos con búsqueda bayesiana) en los 4 conjuntos: F = {e['F']:.2f}, p = {e['p']:.2f}. No se rechaza que todos "
   f"los modelos rindan igual. El rango medio del modelo original es {e['rango_medio'][ORIGINAL]:.2f}; el mejor es "
   f"{r.index[0]} ({r.iloc[0]:.2f}). Wilcoxon del original contra cada modelo, con Holm: ningún p ajustado baja de 1. "
   "Con cuatro conjuntos esta prueba casi no tiene potencia; se reporta porque estaba en el protocolo.")
```

**Conclusión de la evaluación ampliada.** El modelo original no tiene el menor error en ninguno de
los tres conjuntos adicionales. La hipótesis H2 se rechaza.

## Aporte de los componentes en cada conjunto

```{code-cell} ipython3
:tags: [hide-input]
a = res("ablacion.csv")
t = a.pivot_table(index="variante", columns="conjunto", values="MASE")[list(CONJUNTOS)].round(3)
tabla(t.rename(columns=CONJUNTOS).reset_index().rename(columns={"variante": "Variante"}))
```

La tabla matiza lo visto en el capítulo 5. En Superstore el peso único es tan bueno como los pesos
por serie. En **turismo los pesos por serie sí importan**: con un peso único el MASE sube de 0,81 a
0,95, y sin rezago propio se duplica. En M3 y hospital las variantes sin perfil común son mejores que
el modelo completo, coherente con que allí no hay estacionalidad compartida que aprovechar.

## Sensibilidad al hiperparámetro

```{code-cell} ipython3
:tags: [hide-input]
s = res("sensibilidad_lam.csv")
ss = s[s.conjunto == "superstore"].sort_values("lam")
show(lineas(ss.lam, {"Modelo original": ss.MAE}, "Sample Superstore: MAE con λ fijo", "λ (escala logarítmica)", "MAE (USD)",
            log_x=True, referencia=("mejor referencia", float(ss.MAE_mejor_referencia.iloc[0]))))
md(f"Con λ fijo en 12 valores entre 0,01 y 1000, el MAE va de {ss.MAE.min():,.0f} a {ss.MAE.max():,.0f} USD: una variación "
   f"de {(ss.MAE.max() / ss.MAE.min() - 1) * 100:.1f} %. El modelo queda por debajo de la mejor referencia "
   f"({ss.MAE_mejor_referencia.iloc[0]:,.0f}) en todo el rango, así que el resultado no depende de afinar λ. "
   "El error baja al crecer λ: los datos prefieren pesos casi iguales para todas las series.")
```

## Estabilidad por semilla

```{code-cell} ipython3
:tags: [hide-input]
z = res("robustez_semillas.csv")
g = z.groupby("modelo").MAE.agg(["mean", "std", "min", "max"]).round(1).reset_index()
g.columns = ["Modelo", "MAE medio (USD)", "Desviación", "Mínimo", "Máximo"]
tabla(g)
```

Random Forest y XGBoost se repitieron con diez semillas y sus hiperparámetros elegidos. La variación
por semilla es de unos pocos dólares, mucho menor que la diferencia entre modelos y que la variación
entre meses (unos 850 USD). El modelo original no usa azar: es determinista y no tiene semilla.

## Ruido y valores atípicos

```{code-cell} ipython3
:tags: [hide-input]
r = res("robustez_ruido.csv")
t = r.pivot(index="nivel", columns="modelo", values="MAE").round(0).astype(int)
t = t.loc[["sd = 0.0", "sd = 0.1", "sd = 0.2", "sd = 0.4", "5 % de atípicos (x3)"]]
tabla(t.reset_index().rename(columns={"nivel": "Contaminación del entrenamiento"}))
```

Se contaminó la variable objetivo de entrenamiento con ruido multiplicativo (desviación 0,1, 0,2 y
0,4) y con un 5 % de valores multiplicados por 3; se pronostica el mes real. Media de cinco
repeticiones. Los tres modelos resisten el ruido: con desviación 0,4 el error sube menos de 3 %. Los
atípicos afectan más al modelo original y a Random Forest, porque la media del perfil es sensible a
valores extremos; SVR, que ignora los errores grandes, casi no cambia.

## Menos datos

```{code-cell} ipython3
:tags: [hide-input]
m = res("robustez_muestra.csv")
m = m.assign(Estudio=m.estudio.map({"meses": "Meses de entrenamiento", "series": "Número de series"}) + ": " + m.nivel.astype(str))
orden = [f"Meses de entrenamiento: {k}" for k in (12, 18, 24)] + [f"Número de series: {k}" for k in (3, 6, 12)]
t = m.pivot(index="Estudio", columns="modelo", values="MAE").round(0).astype(int).loc[orden]
tabla(t.reset_index())
```

Con solo 12 meses de entrenamiento todos los modelos pierden entre 5 % y 8 %. El número de series
importa más para el modelo original: con 3 series su error es mayor que el de SVR, porque el perfil
común se estima con una cuarta parte de los datos; a partir de 6 series ya es el mejor de los tres.

**Escalabilidad.** El tiempo de ajuste al replicar los datos hasta 64 veces está en el capítulo 5:
crece, como mucho, de forma lineal con el número de filas.

## Cambio en el tiempo

```{code-cell} ipython3
:tags: [hide-input]
ms = mej[mej.conjunto == "superstore"].sort_values("MAE")
tabla(pd.DataFrame({"Modelo": ms.modelo, "MAE en 2016 (USD)": ms.MAE_primera_mitad.map("{:,.0f}".format),
                    "MAE en 2017 (USD)": ms.MAE_segunda_mitad.map("{:,.0f}".format),
                    "Cambio": ((ms.MAE_segunda_mitad / ms.MAE_primera_mitad - 1) * 100).map("{:+.0f} %".format)}))
```

```{code-cell} ipython3
:tags: [hide-input]
o = ms[ms.modelo == ORIGINAL].iloc[0]
otros = ms[ms.modelo != ORIGINAL]
m17 = otros.loc[otros.MAE_segunda_mitad.idxmin()]
f = lambda x: f"{x:,.0f}".replace(",", ".")    # miles con punto
md(f"El error se separa por mitades de la evaluación. En 2017 las ventas son mayores y todos los modelos se "
   f"equivocan más en dólares, pero el modelo original es el que más empeora ({f(o.MAE_primera_mitad)} → "
   f"{f(o.MAE_segunda_mitad)}). **Su ventaja viene de 2016: en 2017 {m17.modelo} tiene menor error "
   f"({f(m17.MAE_segunda_mitad)} frente a {f(o.MAE_segunda_mitad)}).** El resultado global depende, por tanto, "
   "del periodo evaluado.")
```

Los tres conjuntos adicionales, de otros dominios y otras épocas, hacen de validación externa:
ninguna decisión de diseño se tomó con ellos.

## Incertidumbre: intervalos de predicción

Los intervalos son conformes {cite}`angelopoulos2023`. Para pronosticar un mes, el radio es el
cuantil de los errores absolutos del índice en los meses evaluados **anteriores**, multiplicado por
el nivel de la serie. La cobertura se mide en los últimos 12 meses.

```{code-cell} ipython3
:tags: [hide-input]
c = res("conformal.csv")
tabla(pd.DataFrame({"Conjunto": c.conjunto.map(CONJUNTOS), "Modelo": c.modelo, "Nivel nominal": c.nivel.map("{:.0%}".format),
                    "Cobertura medida": c.cobertura.map("{:.1%}".format), "Ancho medio": c.ancho_medio.map("{:,.0f}".format),
                    "Pronósticos": c.n}))
```

La cobertura medida queda cerca de la nominal en los cuatro conjuntos (entre 4 puntos por debajo y 5
por encima). En Superstore el intervalo del modelo original es más angosto que el de la mejor
referencia con una cobertura similar. Los intervalos son anchos: al 80 %, unos 7.300 USD para una
venta media de 4.700. Reflejan lo que el capítulo anterior ya mostró: a nivel de una serie y un mes,
la incertidumbre es alta.

## Dónde falla el modelo

```{code-cell} ipython3
:tags: [hide-input]
s = res("por_serie.csv").sort_values("MAE", ascending=False)
tabla(pd.DataFrame({"Categoría": s.Category, "Región": s.Region, "Venta media (USD)": s.venta_media.map("{:,.0f}".format),
                    "MAE del original": s.MAE.map("{:,.0f}".format), "MAE de la mejor referencia": s.MAE_ref.map("{:,.0f}".format),
                    "Error relativo": (s.MAE / s.venta_media).map("{:.0%}".format), "Sesgo (USD)": s.sesgo.map("{:+,.0f}".format)}))
```

```{code-cell} ipython3
:tags: [hide-input]
t = res("por_tercil.csv")
tabla(pd.DataFrame({"Tercil de ventas del mes": t.tercil, "Venta media (USD)": t.venta_media.map("{:,.0f}".format),
                    "MAE del original": t.MAE.map("{:,.0f}".format), "MAE de la mejor referencia": t.MAE_ref.map("{:,.0f}".format),
                    "Sesgo (USD)": t.sesgo.map("{:+,.0f}".format)}))
```

```{code-cell} ipython3
:tags: [hide-input]
pm = res("por_mes.csv")
MES = ["ene", "feb", "mar", "abr", "may", "jun", "jul", "ago", "sep", "oct", "nov", "dic"]
show(lineas([MES[i - 1] for i in pm.mes], {"Modelo original": pm.MAE, "Mejor referencia": pm.MAE_ref},
            "MAE por mes del año en Sample Superstore", "Mes", "MAE (USD)"))
md(f"El modelo original es mejor que la mejor referencia en {n['errores']['series_mejor_que_ref']} de las "
   f"{n['errores']['series']} series.")
```

- **Por serie.** El error en dólares lo dominan las series grandes, sobre todo las de Technology. En
  términos relativos, las series pequeñas son las peores: su error ronda la mitad de su venta media.
- **Por nivel de ventas.** En los meses de ventas altas el modelo subestima con fuerza (sesgo
  positivo): son pedidos excepcionales que ninguna variable anuncia. En los meses bajos sobreestima.
- **Por mes del año.** El error es mayor en los meses de temporada alta, donde también es mayor la
  venta.

**Causa probable.** El modelo pronostica el valor típico del mes; no puede anticipar un pedido
extraordinario. Para eso harían falta variables que el conjunto no tiene, como promociones o
contratos. La misma limitación afecta a todas las referencias.
