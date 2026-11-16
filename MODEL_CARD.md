# Ficha del modelo: Perfil Estacional con Pesos Jerárquicos

**Versión 1.0.0 · 15 de noviembre de 2026 · Alex Teran y David Estrada · Licencia MIT**

## Qué es

Un modelo de pronóstico para paneles de series mensuales cortas. Predice un índice (ventas del mes ÷
promedio de los 12 meses anteriores de la serie) con dos piezas:

- un **perfil estacional común**: la media del índice en cada mes del año, calculada con todas las series;
- un **peso por serie** que corrige el perfil con lo que hizo esa serie en el mismo mes del año anterior.

Los pesos se aprenden minimizando el error cuadrático, con una penalización `lam` que acerca el peso de
cada serie a un peso común. El pronóstico en dólares es el índice multiplicado por el promedio de los
últimos 12 meses. Código: `src/original_model.py` (solo `numpy`; compatible con scikit-learn).

## Uso previsto

- Pronosticar las ventas del mes siguiente por categoría y región, y proyectar un año de forma
  recursiva, como punto de partida para metas, inventario y presupuesto.
- Servir de referencia simple y barata frente a modelos más complejos en paneles con pocas series,
  historia corta y un calendario comercial compartido.
- Usuarios previstos: analistas de planeación, con el dashboard o con el archivo
  `artifacts/perfil_jerarquico_original.joblib`.

## Usos no recomendados

- **Series que no comparten estacionalidad.** El modelo impone un patrón común. En M3, con series de
  seis dominios distintos, quedó en el puesto 10 de 11.
- Decisiones sobre una sola serie y un solo mes sin mirar el intervalo: el error individual es alto.
- Series con ceros frecuentes o demanda intermitente: el índice divide por el promedio de 12 meses.
- Horizontes largos, productos nuevos (menos de 24 meses de historia) o periodos con cambios de
  precio, promociones o cierres: el modelo no conoce esas variables.
- Decisiones automáticas sin revisión de una persona.

## Datos de entrenamiento

Sample Superstore (conjunto de ejemplo de Tableau): 9.994 líneas de pedido de 2014 a 2017, sumadas en
12 series mensuales (3 categorías × 4 regiones). El primer año solo aporta historia; el modelo final
se entrenó con 432 filas (enero de 2015 a diciembre de 2017). Detalle en `data/DATA_STATEMENT.md`.

## Cómo se evaluó

Origen móvil: cada uno de los 24 meses de 2016 y 2017 se pronosticó entrenando solo con los meses
anteriores, con los hiperparámetros elegidos dentro de cada entrenamiento (validación anidada).
Son 288 pronósticos por modelo. El protocolo se registró antes de experimentar
(`experiments/protocolo.md`, etiqueta `protocolo-v1`).

## Resultados

| Métrica (Sample Superstore) | Modelo original | Mejor referencia (SVR) | Ingenuo estacional |
|---|---|---|---|
| MAE por serie y mes (USD) | 2.310 ± 852 | 2.366 ± 913 | 2.728 ± 928 |
| MASE | 0,996 | 1,020 | 1,176 |
| Desvío del total | −5,1 % | −18,2 % | −19,6 % |
| Cobertura del intervalo al 80 % / 90 % | 80,6 % / 88,2 % | 84,0 % / 92,4 % | — |

- **Frente a la mejor referencia la diferencia no es significativa**: −55 USD, intervalo del 95 %
  [−218, +134]; Diebold–Mariano p = 0,63; d de Cohen = −0,10. El modelo pertenece al conjunto de
  confianza (Model Confidence Set), junto con otras 29 de 35 combinaciones.
- Sí es significativamente mejor que el ingenuo estacional (p ajustado = 0,027) y que Theta (0,034).
- **La ventaja depende del periodo**: en 2016 su MAE es 2.053 frente a 2.288 de SVR; en 2017, 2.567
  frente a 2.444. En el último año evaluado, SVR es mejor.
- **En otros conjuntos no gana**: tercero de 11 en turismo (sin diferencia significativa con el
  primero), noveno en hospital y décimo en M3, donde es significativamente peor que Theta.
- **Veredicto del protocolo: no superior.** Es tan preciso como los mejores modelos en el conjunto
  principal, con una fracción de su costo, y no generaliza a paneles sin estacionalidad común.

## Qué aporta cada pieza

- El perfil común es lo que funciona: estimar un perfil por serie sube el MAE de 2.310 a 2.960.
- Los pesos por serie no mejoran el resultado en Superstore: con un peso único el MAE es 2.280. Sí
  ayudan en turismo (MASE 0,81 frente a 0,95 con peso único), donde las series difieren más entre sí.

## Costo

Un kilobyte en disco. Ajuste en O(n) más O(S) por iteración (S = número de series); pronóstico en O(n).
Es determinista: no usa semilla y siempre da el mismo resultado.

## Limitaciones y riesgos

- **Sesgo hacia abajo en los picos.** Subestima los meses de ventas excepcionales (sesgo medio de
  +239 USD por pronóstico) y el total en 5 %.
- **Error desigual.** Los residuos crecen con el tamaño de la venta (prueba de White, p = 0,001):
  los intervalos son más fiables en series medianas que en las muy grandes o muy pequeñas.
- **Evidencia limitada.** 24 meses de evaluación y un conjunto de datos de demostración. La ventaja
  observada puede no repetirse.
- **Riesgo de uso.** Tomar la proyección anual como una cifra exacta. La proyección de 2018 depende
  del nivel reciente de cada serie y debe leerse con su margen de error.
- **Diseño informado por los datos.** La idea surgió de analizar modelos anteriores sobre este mismo
  conjunto; los tres conjuntos adicionales son la única evidencia independiente, y no es favorable.

## Consideraciones éticas

El modelo no usa datos personales: solo ventas agregadas. Un pronóstico usado para fijar metas afecta
a equipos comerciales; debe acompañarse de su intervalo y revisarse con quien conoce el negocio.

## Mantenimiento

Reentrenar cada mes con el dato nuevo (tarda milisegundos). Revisar el modelo si el error de tres
meses seguidos queda fuera del intervalo del 90 % o si cambia el calendario comercial.

## Cómo citar

Ver `CITATION.cff`. Repositorio: https://github.com/AlexT09/SuperstorePython2
