# Introducción

*Pronóstico mensual de ventas de Sample Superstore: siete modelos, un modelo propio y una evaluación
honesta*

**Alex Teran y David Estrada**

Tablero interactivo: **https://superstorepython2.onrender.com** ·
Código: **https://github.com/AlexT09/SuperstorePython2**

## Resumen

Una cadena minorista necesita saber cuánto venderá cada categoría de producto en cada región, mes a
mes, para fijar metas, comprar inventario y armar el presupuesto. Con el conjunto Sample Superstore
(9.994 líneas de pedido de 2014 a 2017) se construyeron 12 series mensuales y se compararon siete
modelos de aprendizaje automático, cada uno ajustado con cuatro métodos de optimización, contra un
modelo propio: un perfil estacional común a todas las series combinado con la historia de cada
serie mediante pesos aprendidos.

La comparación siguió un protocolo registrado antes de experimentar: 24 meses pronosticados uno a uno
con validación anidada, cuatro pruebas de hipótesis y tres conjuntos de datos adicionales. El modelo
propio obtuvo el menor error en Sample Superstore, 2.310 USD por serie y mes frente a 2.366 del mejor
modelo del curso (SVR), con un modelo de un kilobyte; pero la diferencia **no es estadísticamente
significativa** y el resultado **no se repite** en los conjuntos adicionales. El veredicto, tal como
lo fija el protocolo, es que el modelo no es superior. Para 2018 se proyectan 886.866 USD.

## Contribuciones

1. Una evaluación de las 28 combinaciones de siete modelos y cuatro métodos de optimización con
   validación anidada sobre un pronóstico de series de tiempo, sin fuga de información.
2. Un modelo propio, el perfil estacional con pesos jerárquicos, con su formulación, reglas de
   actualización derivadas, demostración de convergencia y pruebas unitarias.
3. Una validación estadística registrada de antemano, que reporta el resultado aunque sea desfavorable.
4. Un tablero desplegado que permite explorar los datos, comparar modelos y obtener pronósticos con
   intervalo y explicación.

## Objetivos

Identificar los factores que explican las ventas y construir un modelo que proyecte las ventas
mensuales por categoría y región, comparándolo con rigor con los modelos estudiados en el curso.

## Mapa del libro

| Capítulo | Contenido |
|---|---|
| 1. Planteamiento | Problema y objetivos, marco teórico y metodología |
| 2. Datos | ETL y análisis exploratorio |
| 3. Modelos de referencia | Las 28 combinaciones, optimizadores, complejidad, residuos e interpretabilidad |
| 4. Literatura y novedad | Búsqueda documentada, trabajos cercanos y declaración de novedad |
| 5. Modelo original | Formulación, algoritmo, sustento teórico, implementación, pruebas y aporte de cada componente |
| 6. Validación estadística | Protocolo, pruebas de hipótesis y veredicto |
| 7. Evaluación ampliada | Conjuntos adicionales, robustez, incertidumbre y análisis de errores |
| 8. Dashboard | Arquitectura, contenido y despliegue |
| 9. Conclusiones | Hallazgos, veredicto, limitaciones y declaración de uso de herramientas |
| 10. Reproducibilidad | Entorno, tabla maestra, lista de verificación y bibliografía |

## Impacto para el negocio

- **Metas comerciales:** fijar la meta de cada región según lo que se espera vender.
- **Inventario y compras:** abastecer cada categoría antes de su temporada alta.
- **Alerta temprana:** detectar dónde se proyecta una caída frente al año anterior.
- **Presupuesto:** proyectar los ingresos del año siguiente, con su margen de error.
