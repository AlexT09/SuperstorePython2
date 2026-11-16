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

# Revisión de literatura y novedad

*Qué existía antes y en qué se diferencia el modelo propuesto*

La novedad no se declara: se documenta. Este capítulo registra la búsqueda, los trabajos más cercanos
y una declaración que dice solo lo que la búsqueda permite decir.

## Protocolo de búsqueda

- **Bases consultadas:** OpenAlex, Crossref y arXiv, a través de sus interfaces públicas. Son bases
  abiertas; OpenAlex indexa las revistas de pronóstico donde está la literatura relevante
  (*International Journal of Forecasting*, *Journal of Forecasting*, *Journal of the Operational
  Research Society*).
- **Fecha de la consulta:** 15 de noviembre de 2026.
- **Cadenas:** ocho, que combinan la familia del modelo (índices estacionales de grupo, contracción,
  agrupamiento parcial, combinación de pronósticos), el problema (ventas minoristas, Superstore) y
  términos en inglés y en español.
- **Reproducible:** `experiments/literatura/buscar.py` repite la búsqueda y guarda los resultados
  crudos en `experiments/literatura/busqueda.json`.
- **Búsqueda por citación:** se revisaron los artículos que citan a los dos trabajos más cercanos
  (Chen y Boylan, 2008; Zhang y otros, 2013). Hacia atrás, los trabajos cercanos se citan entre sí y
  forman una sola línea de investigación, que es la que recoge la tabla de más abajo.

**Límites de la búsqueda.** Crossref y arXiv no buscan frases exactas: cuentan los documentos que
contienen cualquiera de los términos, por lo que sus totales son muy grandes y poco informativos; de
ellos se revisaron los primeros resultados por relevancia. No se consultaron bases de pago.

## Matriz de novedad

```{code-cell} ipython3
:tags: [hide-input]
import json
import pandas as pd
from common import ROOT, md, tabla

b = json.loads((ROOT / "experiments" / "literatura" / "busqueda.json").read_text(encoding="utf-8"))
# Cadenas con algún resultado directamente relacionado, tras leer título y resumen de los primeros resultados
RELEVANTES = {'"group seasonal indices" forecasting': (6, "Antecedentes directos: índices de grupo y su contracción"),
              '"seasonal indices" shrinkage forecasting': (4, "Los mismos trabajos, más la contracción de Miller y Williams"),
              '"seasonal index" "partial pooling" sales forecasting': (1, "Una tesis sobre demanda estacional por producto; sin esta regla"),
              'hierarchical shrinkage combination weights "seasonal naive" retail sales': (0, "Reconciliación jerárquica y redes; otro problema"),
              '"forecast combination" series-specific weights shrinkage seasonal profile': (0, "Combinación de pronósticos en energía y finanzas"),
              '"Superstore" sales forecasting seasonal': (0, "Otros conjuntos llamados superstore (frutas y verduras); ARIMA y LSTM"),
              '"índices estacionales" pronóstico ventas minoristas': (0, "Trabajos de grado sobre inventarios; sin modelo afín"),
              'cross-learning seasonality pooled seasonal profile short time series retail': (0, "Resultados dispersos; el trabajo afín apareció en arXiv (1908.05339)")}
filas = []
for c in b["consultas"]:
    rel, obs = RELEVANTES[c["cadena"]] if c["base"] == "OpenAlex" else (None, None)
    filas.append({"Base": c["base"], "Cadena de búsqueda": c["cadena"],
                  "Resultados": "sin respuesta" if c["resultados"] is None else f"{c['resultados']:,}".replace(",", "."),
                  "Revisados": len(c["primeros"]), "Relevantes": "" if rel is None else rel,
                  "Observación": obs or "No busca por frase; se revisaron los primeros por relevancia"})
md(f"Fecha de la consulta: **{b['fecha']}**.")
tabla(pd.DataFrame(filas))
```

«Relevantes» cuenta, entre los resultados revisados de OpenAlex, los que tratan de la misma familia de
métodos. Para Crossref y arXiv se leyeron los títulos de los primeros resultados sin encontrar
trabajos distintos de los ya identificados, salvo el de agrupamiento mixto en arXiv.

## Trabajos más cercanos

| Trabajo | Qué propone | Qué se conserva | Qué cambia en este proyecto |
|---|---|---|---|
| Bunn y Vassilopoulos (1993, 1999) {cite}`bunn1993,bunn1999` | Índices estacionales de grupo para pronosticar muchos productos a corto plazo | La idea central: estimar la estacionalidad con varias series juntas | Usan el índice del grupo para todas las series; aquí cada serie puede apartarse de él |
| Miller y Williams (2003) {cite}`miller2003` | Contraer los índices estacionales de una serie hacia 1 (tipo James–Stein) | La contracción como remedio al ruido | Contraen hacia «sin estacionalidad», no hacia un perfil común |
| Chen y Boylan (2007, 2008) {cite}`chen2007,chen2008` | Reglas teóricas y evidencia para elegir entre índice individual, de grupo o contraído | El diagnóstico: con poca historia, el índice individual es ruidoso | Eligen una opción por serie; aquí se mezclan con un peso continuo |
| Zhang, Chen, Boylan y Scarf (2013) {cite}`zhang2013` | Estimador general que combina agrupamiento y contracción | Combinar el índice del grupo con información individual | **Es el antecedente más cercano.** Sus pesos salen de fórmulas de error cuadrático de los índices; aquí se aprenden minimizando el error de pronóstico, y los pesos de las distintas series se agrupan entre sí |
| Boylan, Chen, Mohammadipour y Syntetos (2014) {cite}`boylan2014` | Cómo formar los grupos de series | — | Aquí el grupo es todo el panel; no se forman subgrupos |
| Agrupamiento mixto de la estacionalidad (arXiv:1908.05339) | Modelo mixto que comparte la estacionalidad entre series | Agrupamiento parcial de la estacionalidad | Agrupa el componente estacional dentro de un modelo estadístico completo; aquí se agrupan los pesos de combinación |
| Montero-Manso y Hyndman (2021) {cite}`monteromanso2021` | Principio de los modelos globales: un modelo para todas las series | El marco conceptual | No proponen esta regla |

La búsqueda por citación confirma el cuadro: los 21 trabajos que citan a Chen y Boylan (2008) son
revisiones de pronóstico de demanda, modelos de aprendizaje profundo (DeepAR) y aplicaciones; ninguno
propone aprender por serie el peso entre el perfil del grupo y el rezago propio.

## Diferencia con los modelos del curso

| Familia | Por qué no resuelve la limitación |
|---|---|
| Modelos lineales (Ridge, Lasso) | Un solo coeficiente para el rezago en todas las series, y la estacionalidad entra como seno y coseno: no pueden representar un perfil de doce valores libres |
| Vecinos (KNN) | Promedia meses parecidos sin distinguir entre patrón común y particular |
| Árboles y ensambles (árbol, Random Forest, XGBoost) | Pueden aprender el perfil, pero con cientos de parámetros y mucha varianza para 144 a 420 filas |
| SVR | Minimiza un error insensible a los picos: acierta en los meses corrientes y subestima el total |
| Ensambles estándar (promedio, *stacking* lineal) | Pesos iguales para todas las series |

## Declaración de novedad

Al mejor de nuestro conocimiento, y según la búsqueda descrita, **no se encontraron estudios que
apliquen a Sample Superstore un modelo de perfil estacional común con pesos por serie aprendidos por
agrupamiento parcial**, ni que propongan esa regla de actualización en otra aplicación.

Lo que **no** es nuevo debe quedar igual de claro: estimar índices estacionales con un grupo de
series y contraerlos son ideas conocidas en el pronóstico de demanda desde los años noventa, y el
trabajo de Zhang, Chen, Boylan y Scarf ya combina las dos. La contribución de este proyecto es
acotada: (i) la regla de combinación con un peso por serie, aprendido junto con un peso común por
descenso por bloques sobre un objetivo cuadrático, con su prueba de convergencia; y (ii) su
evaluación frente a siete modelos de aprendizaje automático en un panel minorista. Como se verá, la
evaluación muestra que el mecanismo (i) aporta poco frente a un peso único: el valor está en el
perfil común, que es la parte conocida.

La búsqueda cubre bases abiertas en una fecha concreta. No descarta antecedentes en bases no
consultadas o en literatura no indexada.
