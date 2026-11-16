# Metodología

*Cómo se pasa de los datos crudos a la proyección*

## Flujo del proyecto

1. **ETL:** CSV crudo → dataset limpio y tabla mensual (categoría × región × mes).
2. **EDA:** distribución de las ventas, factores, estacionalidad y tendencia.
3. **Variables:** mes del año e historial de ventas, sin información del futuro.
4. **Modelos:** 7 regresores dentro de un Pipeline y un modelo propio.
5. **Optimización:** Grid, Random, bayesiana y genética, con el mismo presupuesto.
6. **Validación anidada:** 24 meses pronosticados uno a uno; los hiperparámetros se eligen dentro de
   cada entrenamiento.
7. **Comparación estadística:** Model Confidence Set, Giacomini–White, Diebold–Mariano y Wilcoxon,
   con corrección de Holm, según un protocolo registrado antes de experimentar.
8. **Evaluación ampliada:** tres conjuntos adicionales, ruido, menos datos e intervalos de predicción.
9. **Proyección:** modelo reentrenado con toda la historia → 2018 mes a mes.

## Datos y unidad de análisis

- Se agregan las 9.994 líneas de pedido en ventas mensuales por categoría × región: 12 series × 48
  meses = 576 filas.
- 2014 solo aporta historia (las variables necesitan 12 meses previos).
- Evaluación con **origen móvil**: cada uno de los 24 meses de 2016 y 2017 se pronostica entrenando
  solo con los meses anteriores (288 pronósticos por modelo).
- Nunca se entrena con meses posteriores al que se pronostica: mezclar meses dejaría al modelo ver
  el futuro (fuga de información).

## Variable objetivo

El modelo predice un índice: cuántas veces el promedio de los últimos 12 meses se venderá ese mes.
Luego se multiplica por ese promedio para volver a dólares.

Así una misma regla sirve para series grandes (Technology en West) y pequeñas (Furniture en South), y
la proyección sigue el nivel reciente de cada serie.

## Auditoría de fuga de información

| Posible fuga | Cómo se evita |
|---|---|
| Preprocesamiento ajustado con todos los datos | El codificador one-hot y el escalador están dentro del `Pipeline` y se ajustan en cada pliegue con sus datos de entrenamiento |
| Variables que contienen la respuesta | Todas las variables usan meses anteriores (`shift`); el índice divide por el promedio de los 12 meses *previos* |
| Mezcla temporal | Validación cruzada temporal (`TimeSeriesSplit`) y origen móvil: siempre se entrena con el pasado |
| Duplicados entre particiones | Cada fila es una serie en un mes: no se repite entre entrenamiento y evaluación |
| Selección con los datos de evaluación | Los hiperparámetros se eligen en el bucle interno; el modelo original contrastado y su método de optimización se fijaron en el protocolo |
| Perfil del modelo original | Se calcula solo con los meses de entrenamiento de cada origen |

## Validación y optimización

- **Bucle externo:** los 24 meses de evaluación. Mide el error que se reporta.
- **Bucle interno:** dentro de cada entrenamiento, `TimeSeriesSplit` con pliegues de 3 meses (hasta 4)
  elige los hiperparámetros.
- **Métodos de optimización:** Grid Search, Random Search, bayesiana (Optuna) y genética (DEAP), con
  16 evaluaciones para las tres últimas. Detalle en el capítulo 3.
- **Semilla fija (42)** en todos los componentes con azar.
- **Referencias:** el ingenuo estacional (repetir el mismo mes del año anterior), el método Theta y
  HistGradientBoosting. Un modelo solo aporta si supera al ingenuo estacional.
