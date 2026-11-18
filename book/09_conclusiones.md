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

# Conclusiones y limitaciones

*Lo que se encontró, lo que no, y qué hacer con ello*

## Hallazgos

1. **Las ventas de Sample Superstore son sobre todo estacionalidad.** Septiembre a diciembre
   concentran cerca de la mitad de las ventas del año, y en todos los modelos la predicción depende
   casi solo del mes del año y de las ventas del mismo mes del año anterior.
2. **Los siete modelos del curso rinden de forma parecida.** Con la mejor combinación de cada uno, el
   MAE va de 2.366 (SVR) a 2.514 USD (Ridge) por serie y mes, y el Model Confidence Set no separa a la
   mayoría. Todos mejoran al ingenuo estacional (2.728).
3. **Los cuatro métodos de optimización llegan a lo mismo.** Con espacios de uno a cuatro
   hiperparámetros y 16 evaluaciones, la búsqueda bayesiana obtiene el menor error medio por un
   margen de pocos dólares; la genética cuesta ocho veces más sin mejorar.
4. **El modelo original es tan preciso como los mejores, con mucho menos.** Obtiene el menor MAE en
   Sample Superstore (2.310 USD), con un modelo de un kilobyte que se ajusta en milisegundos. La
   ventaja frente a SVR, 55 USD, no es estadísticamente significativa, y viene toda de 2016: en 2017
   SVR tiene menor error.
5. **Lo que funciona es el perfil común, no los pesos por serie.** Estimar la estacionalidad con las
   12 series juntas es lo decisivo; aprender un peso distinto por serie no mejora a un peso único en
   este conjunto, aunque sí ayuda en el de turismo.
6. **Fuera de Sample Superstore no gana.** Tercero en turismo, noveno en hospital y décimo en M3,
   donde es significativamente peor que Theta porque las series no comparten estacionalidad.

## Veredicto sobre el modelo original

| Pregunta | Respuesta | Evidencia |
|---|---|---|
| ¿Es nuevo para este problema? | **Parcialmente.** No se encontró la regla de pesos por serie con agrupamiento parcial ni su uso en Sample Superstore; pero los índices estacionales de grupo y su contracción son conocidos | Capítulo 4 |
| ¿Está implementado por el autor? | **Sí.** Clase propia con `numpy`, reglas de actualización derivadas, demostración de convergencia y 11 pruebas unitarias | Capítulo 5 |
| ¿Es superior a las referencias? | **No.** Menor error en el conjunto principal, pero sin significancia estadística, con un efecto despreciable y sin repetirse en los conjuntos adicionales | Capítulos 6 y 7 |

## Proyección 2018

```{code-cell} ipython3
:tags: [hide-input]
from common import hallazgos, load_artifact, load_data, md, usd

r = load_artifact("resumen.json")
rec = r["prueba_recursiva"]
md(f"Con el modelo original reentrenado con 2015–2017 se proyectan **{usd(r['proyeccion_2018'])}** para 2018, "
   f"{(r['proyeccion_2018'] / r['ventas_2017'] - 1) * 100:+.0f} % frente a 2017 ({usd(r['ventas_2017'])}). "
   f"La misma regla, aplicada a 2017 solo con datos hasta 2016, dejó el total anual a "
   f"{rec['total_desvio_total_pct']:+.1f} % del real, con un error típico mensual de ±{rec['total_MAE_pct']:.0f} % "
   "para el total de la empresa.")
```

## Recomendaciones para el negocio

```{code-cell} ipython3
:tags: [hide-input]
h = hallazgos(load_data())
md("\n".join(f"{n}. **{f.Factor}:** {f.Recomendación}" for n, f in enumerate(h.itertuples(), 1)))
```

Salen de los hallazgos del análisis exploratorio, donde está el dato que respalda cada una.

## Limitaciones

- **Un conjunto de demostración.** Sample Superstore no es una empresa identificada; las conclusiones
  sirven para comparar métodos, no para trasladarlas sin validar.
- **Poca potencia estadística.** 24 meses de evaluación no permiten detectar diferencias del orden de
  1 % del nivel de ventas.
- **Diseño informado por los datos.** El modelo se diseñó después de analizar modelos anteriores en
  este mismo conjunto. Por eso los conjuntos adicionales son la evidencia más limpia, y no son
  favorables.
- **Error alto por serie.** A nivel de una categoría en una región y un mes, el error ronda la mitad
  de la venta: el pronóstico es más útil para totales.
- **Sin variables externas.** Ni precios, ni promociones, ni calendario comercial: ningún modelo
  anticipa los pedidos extraordinarios, que explican los peores errores.
- **El análisis exploratorio es asociativo:** no demuestra causalidad.

## Trabajo futuro

- Formar grupos de series con estacionalidad parecida en lugar de usar un perfil para todo el panel,
  para que el modelo no falle en paneles heterogéneos como M3.
- Una versión robusta del perfil (mediana o media recortada) para reducir el efecto de los meses
  atípicos.
- Evaluar con más historia o más series, donde las pruebas tengan potencia suficiente.
- Añadir variables de calendario comercial y promociones, si estuvieran disponibles.

## Declaración de autoría y uso de herramientas

Este trabajo fue realizado por Alex Teran. Se usaron las siguientes herramientas:

- **Python** y las librerías listadas en el capítulo de reproducibilidad, como herramientas de cálculo.
- **Un asistente de inteligencia artificial (Claude, de Anthropic)**, usado para: proponer y programar
  el modelo original y sus pruebas unitarias; programar los experimentos, las pruebas estadísticas y
  los métodos de optimización; construir las páginas nuevas del tablero; hacer la búsqueda
  bibliográfica en bases abiertas; y redactar borradores de este libro, del artículo y de la ficha
  del modelo.
- El autor revisó el código y los resultados, y es responsable de comprender y defender cada
  parte del trabajo entregado.

Firma:

| Nombre | Firma | Fecha |
|---|---|---|
| Alex Teran | | |
