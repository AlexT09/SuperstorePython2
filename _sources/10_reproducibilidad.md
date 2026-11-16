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

# Reproducibilidad y bibliografía

*Cómo repetir cada número de este libro*

## Código y datos

- **Repositorio:** https://github.com/AlexT09/SuperstorePython2 (licencia MIT; forma de citarlo en
  `CITATION.cff`).
- **Datos:** `data/raw/Sample - Superstore.csv` (conjunto de ejemplo de Tableau) y los tres conjuntos
  adicionales en `data/adicionales/`, descargados de Zenodo por `experiments/datos_adicionales.py`.
  Detalle en `data/DATA_STATEMENT.md`.
- **Protocolo:** `experiments/protocolo.md`, etiqueta de Git `protocolo-v1`.

## Cómo repetir los resultados

Desde la raíz del repositorio:

```bash
pip install -r requirements-dev.txt
python data/generate_data.py              # CSV crudo -> tabla mensual
python experiments/datos_adicionales.py   # descarga los tres conjuntos adicionales
python experiments/ejecutar.py            # validación anidada de todos los modelos (~2 h)
python experiments/analizar.py            # tablas, pruebas y veredicto (segundos)
python experiments/robustez.py            # ruido, menos datos, semillas, escalabilidad (~15 min)
python experiments/artefactos.py          # modelos finales, metadata.json y proyección 2018 (~10 min)
python -m pytest -q tests                 # pruebas del modelo original
jupyter-book build book/                  # este libro
```

Los resultados ya calculados están versionados en `experiments/resultados/` y `artifacts/`; el libro
y el tablero los leen sin volver a entrenar.

## Entorno

```{code-cell} ipython3
:tags: [hide-input]
from common import ORIGINAL, OPT, md, res, tabla
import pandas as pd

meta = res("metadata.json")
tabla(pd.DataFrame([("Python", meta["python"])] + list(meta["librerias"].items()) +
                   [("statsmodels", "0.15.0"), ("optuna", "5.0.0"), ("deap", "1.4.4"), ("dash", "4.4.1")],
                   columns=["Componente", "Versión"]))
```

Equipo: portátil con AMD Ryzen 5 7520U (4 núcleos, 8 hilos), 8 GB de memoria, Windows 11, sin GPU.

## Tabla maestra de experimentos

`experiments/resultados/tabla_maestra.csv` tiene una fila por conjunto, mes de evaluación, modelo y
método de optimización, con los hiperparámetros elegidos, el error interno, el número de
evaluaciones, los tiempos, el historial de la búsqueda y la semilla.

```{code-cell} ipython3
:tags: [hide-input]
t = res("tabla_maestra.csv")
r = t.groupby(["conjunto", "optimizador"]).agg(Filas=("modelo", "size"), Modelos=("modelo", "nunique"),
                                              Evaluaciones=("evaluaciones", "sum"), Segundos=("busqueda_s", "sum")).reset_index()
r["optimizador"] = r.optimizador.map(OPT)
r["Segundos"] = r.Segundos.round(0).astype(int)
tabla(r.rename(columns={"conjunto": "Conjunto", "optimizador": "Método"}))
md(f"En total, {len(t):,} búsquedas de hiperparámetros y {int(t.evaluaciones.sum()):,} evaluaciones de "
   "validación cruzada.".replace(",", "."))
```

## Lista de verificación

| Elemento | Dónde |
|---|---|
| Versiones exactas de las librerías | `requirements.txt`, `requirements-dev.txt`, tabla de entorno |
| Hardware | Arriba |
| Semilla global | 42 en `model/features.py` y `experiments/config.py`; propagada a numpy, scikit-learn, XGBoost, Optuna y DEAP |
| Rangos de hiperparámetros | `experiments/config.py`; capítulo 3 |
| Número de ejecuciones | 24 meses de evaluación por modelo y conjunto; 10 semillas en el estudio de estabilidad; 5 repeticiones en el de ruido |
| Medidas de dispersión | Desviación estándar entre meses e intervalos por bootstrap estacionario en todas las tablas |
| Presupuesto de ajuste | 16 evaluaciones (Random, bayesiana, genética); rejilla completa en Grid Search |
| Tiempo de cómputo | Tabla maestra; capítulos 3 y 5 |
| Datos | `data/`, `data/DATA_STATEMENT.md` |
| Código | Repositorio público |
| Pruebas automáticas | `tests/`, GitHub Actions (`.github/workflows/pruebas.yml`) |
| Prueba de ida y vuelta de los modelos guardados | `artifacts/metadata.json` (campo `ida_y_vuelta`) |

## Bibliografía

```{bibliography}
```
