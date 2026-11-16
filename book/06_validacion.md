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

# Protocolo y validación estadística

*Qué se decidió antes de experimentar y qué dicen las pruebas*

## Protocolo registrado

Antes de ejecutar los experimentos definitivos se congeló el documento `experiments/protocolo.md`
y se marcó en el repositorio con la etiqueta de Git `protocolo-v1` (15 de noviembre de 2026). Fija:

| Elemento | Decisión |
|---|---|
| Hipótesis confirmatorias | H1: en Sample Superstore el modelo original tiene menor MAE que las referencias. H2: lo mantiene en al menos 2 de 3 conjuntos adicionales sin ser significativamente peor en ninguno |
| Métrica principal | MAE en unidades originales, promediado por mes de evaluación |
| Métricas secundarias | RMSE, MASE, WAPE y desvío del total |
| Particiones | Origen móvil de 24 meses; validación cruzada temporal interna con pliegues de 3 meses |
| Semilla | 42 en todos los componentes |
| Combinación contrastada | El modelo original con búsqueda bayesiana, fijada de antemano |
| Referencias | La mejor combinación de cada uno de los 7 modelos, el ingenuo estacional, Theta y HistGradientBoosting: 10 comparaciones |
| Pruebas | Model Confidence Set (α = 0,10); Giacomini–White, Diebold–Mariano y Wilcoxon con corrección de Holm (α = 0,05) |
| Regla de decisión | Superior solo si se cumplen cuatro condiciones (ver el veredicto) |

**Lo que ya se conocía.** El conjunto principal se había usado para diseñar el modelo, en tres
iteraciones de diseño: un Random Forest global, un perfil estacional con un peso único elegido en
rejilla, y el modelo actual. Los resultados de las dos primeras se conocían; el modelo actual no se
había evaluado. Los tres conjuntos adicionales no se habían tocado: son la evidencia independiente.

**Desviaciones del protocolo.** Dos, ninguna afecta los resultados: (1) se añadió al `Pipeline` un
paso que convierte la matriz a densa, porque con 24 series la codificación one-hot sale dispersa y
HistGradientBoosting no la acepta; los resultados de Superstore, ya calculados, no cambian. (2) En
el título de una sección del protocolo se corrigió la redacción después de registrarlo; la etiqueta
conserva el texto original.

## Equidad de la comparación

- **Mismas particiones.** Todos los modelos pronostican los mismos 24 meses con las mismas filas de
  entrenamiento y los mismos pliegues internos.
- **Mismo presupuesto.** 16 evaluaciones de hiperparámetros para todos en las búsquedas aleatoria,
  bayesiana y genética. Grid Search recorre su rejilla: 12 combinaciones para el modelo original,
  entre 4 y 24 para las referencias.
- **La selección favorece a las referencias.** De cada modelo base se toma su mejor método de
  optimización según el propio bucle externo; del modelo original, un método fijado de antemano.
- **Equipo.** Portátil con procesador AMD Ryzen 5 7520U (4 núcleos, 8 hilos), 8 GB de memoria,
  Windows 11, sin GPU. La corrida de Superstore tardó 115 minutos y cada conjunto adicional unos 6.

## Por qué estas pruebas

Los errores de meses seguidos están correlacionados, y todos los modelos se reestiman en cada mes.
Una prueba *t* corriente supone observaciones independientes y no vale. Las pruebas elegidas trabajan
sobre la serie de 24 errores mensuales de cada modelo:

- **Model Confidence Set** {cite}`hansen2011`: parte de todas las combinaciones y elimina las
  significativamente peores. Controla el efecto de haber probado muchos modelos.
- **Giacomini–White** {cite}`giacomini2006`: capacidad predictiva condicional; es válida cuando los
  modelos se reestiman en cada origen.
- **Diebold–Mariano** {cite}`diebold1995` con la corrección de Harvey, Leybourne y Newbold
  {cite}`harvey1997` para muestras pequeñas.
- **Wilcoxon** de rangos con signo: no supone normalidad de las diferencias.
- Intervalos por **bootstrap estacionario** {cite}`politis1994` (10.000 réplicas, bloque medio de 3
  meses) y tamaño del efecto con la *d* de Cohen y la delta de Cliff.

## Model Confidence Set

```{code-cell} ipython3
:tags: [hide-input]
from common import CONJUNTOS, OPT, ORIGINAL, md, res, show, tabla, usd
import pandas as pd

m = res("mcs.csv")
n = res("numeros.json")
s = m[m.conjunto == "superstore"].sort_values("MAE")
tabla(pd.DataFrame({"Modelo": s.modelo, "Método": s.optimizador.map(OPT), "MAE (USD)": s.MAE.map("{:,.0f}".format),
                    "p-valor MCS": s.p_mcs.map("{:.3f}".format), "¿En el conjunto?": s.en_conjunto.map({True: "Sí", False: "No"})}))
```

```{code-cell} ipython3
:tags: [hide-input]
fuera = s[~s.en_conjunto]
md(f"De las {n['superstore']['mcs_total']} combinaciones, {n['superstore']['mcs_tamano']} quedan en el conjunto de confianza. "
   f"Solo se eliminan {len(fuera)}: " + ", ".join(f"{a} ({OPT[b]})" for a, b in zip(fuera.modelo, fuera.optimizador)) + ". "
   "El modelo original pertenece al conjunto, y también la mejor referencia: **con 24 meses, los datos no "
   "distinguen al modelo original de casi ninguno de los demás.**")
```

## Modelo original frente a cada referencia

```{code-cell} ipython3
:tags: [hide-input]
p = res("pruebas.csv")
ps = p[p.conjunto == "superstore"].sort_values("MAE_referencia")
tabla(pd.DataFrame({
    "Referencia": [f"{a} ({OPT[b]})" for a, b in zip(ps.referencia, ps.optimizador)],
    "Diferencia de MAE [IC 95 %]": [f"{d:+,.0f} [{lo:+,.0f}, {hi:+,.0f}]" for d, lo, hi in zip(ps.dif, ps.ic_lo, ps.ic_hi)],
    "DM: p / p Holm": [f"{a:.3f} / {b:.3f}" for a, b in zip(ps.p_DM, ps.p_DM_holm)],
    "GW: p / p Holm": [f"{a:.3f} / {b:.3f}" for a, b in zip(ps.p_GW, ps.p_GW_holm)],
    "Wilcoxon: p / p Holm": [f"{a:.3f} / {b:.3f}" for a, b in zip(ps.p_wilcoxon, ps.p_wilcoxon_holm)],
    "d de Cohen": ps.d_cohen.map("{:+.2f}".format), "δ de Cliff": ps.delta_cliff.map("{:+.2f}".format),
    "Meses en que gana": [f"{k} de 24" for k in ps.meses_mejor]}))
```

Diferencia = MAE del modelo original − MAE de la referencia, en dólares; negativa significa que gana
el original. H₀ de las tres pruebas: los dos modelos tienen la misma pérdida esperada.

**Lectura.**

- El modelo original tiene menor MAE que las diez referencias y gana en 13 a 19 de los 24 meses.
- **Frente a la mejor referencia, SVR, la diferencia es de −55 USD con un intervalo que va de −218 a
  +134.** Ninguna prueba rechaza la igualdad (Diebold–Mariano p = 0,63) y el efecto es despreciable
  (*d* = −0,10).
- Tras la corrección de Holm, Diebold–Mariano y Wilcoxon solo rechazan frente al **ingenuo
  estacional** y a **Theta**, con efectos medianos (*d* ≈ −0,7). Giacomini–White no rechaza en ningún
  caso tras el ajuste.
- Frente a XGBoost, Ridge, el árbol y HistGradientBoosting los intervalos excluyen el cero, pero los
  p-valores no sobreviven a las diez comparaciones: son indicios, no evidencia.
- **Significancia práctica.** La venta media por serie y mes es de unos 4.700 USD. La ventaja sobre
  SVR equivale al 1 % de esa venta; la ventaja sobre el ingenuo estacional, al 9 %.

**Supuestos.** Las diferencias mensuales no muestran autocorrelación apreciable (Ljung–Box a 3
rezagos, p > 0,05 en nueve de diez comparaciones), lo que respalda la estacionariedad que requiere
Diebold–Mariano. La normalidad se rechaza en algunas comparaciones (Shapiro–Wilk), por lo que se
añadió Wilcoxon. Los modelos comparados no están anidados entre sí, de modo que no hace falta la
corrección de Clark–West.

```{code-cell} ipython3
:tags: [hide-input]
md("Ljung–Box (3 rezagos) y Shapiro–Wilk de las diferencias mensuales, por referencia:")
tabla(pd.DataFrame({"Referencia": ps.referencia, "Ljung–Box (p)": ps.lb_p.map("{:.3f}".format),
                    "Shapiro–Wilk (p)": ps.shapiro_p.map("{:.3f}".format)}))
```

## Veredicto

```{code-cell} ipython3
:tags: [hide-input]
v = res("veredicto.json")
si = lambda b: "Sí" if b else "No"
tabla(pd.DataFrame([
    ("1. Menor MAE que las 10 referencias en Sample Superstore", si(v["c1_menor_MAE_que_todas"])),
    ("2. Al menos tres pruebas rechazan la igualdad frente a la mejor referencia", f"{si(v['c2_tres_pruebas'])} ({sum(v['rechazos'].values())} de 4)"),
    ("3. Tamaño del efecto no despreciable", f"{si(v['c3_efecto'])} (d = {v['d_cohen']:+.2f}; δ = {v['delta_cliff']:+.2f})"),
    ("4. Menor MAE en al menos 2 de los 3 conjuntos adicionales y no significativamente peor en ninguno",
     f"{si(v['c4_consistencia'])} (gana en {sum(v['gana_en'].values())} de 3; peor en {sum(v['significativamente_peor_en'].values())})"),
], columns=["Condición fijada en el protocolo", "¿Se cumple?"]))
```

**El modelo original no es superior.** Cumple la primera condición y ninguna de las otras tres. Tiene
el menor error en el conjunto principal, pero esa ventaja no es estadísticamente significativa, su
tamaño es despreciable frente a la mejor referencia y no se repite en los conjuntos adicionales.

**Causas probables.** (1) *Poca potencia:* con 24 observaciones mensuales y una desviación de unos
850 USD, una diferencia de 55 USD no se puede detectar. (2) *El margen real es pequeño:* todos los
modelos explotan la misma señal, el mes del año y el rezago anual; el modelo original la captura con
menos parámetros, no con más información. (3) *El mecanismo nuevo no aporta aquí:* como muestra la
ablación, el peso por serie no mejora al peso único en este conjunto.

Lo que sí sostiene la evidencia es una afirmación más modesta: en Sample Superstore, un modelo de
doce parámetros y un peso por serie pronostica **tan bien como** los mejores modelos del curso, mejor
que repetir el año anterior, y con un costo mucho menor.
