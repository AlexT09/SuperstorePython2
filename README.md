# Pronóstico de ventas Sample Superstore

Pronóstico mensual de ventas por categoría y región con el conjunto **Sample Superstore** (2014–2017):
siete modelos de aprendizaje automático, cuatro métodos de optimización de hiperparámetros y un
modelo propio, comparados con validación anidada y pruebas de hipótesis registradas de antemano.

**Autores:** Alex Teran y David Estrada · **Licencia:** MIT · **Cómo citarlo:** `CITATION.cff`

- **Tablero (Dash):** https://superstorepython.onrender.com
- **Informe (Jupyter Book):** https://alext09.github.io/SuperstorePython/

## Resultados en una tabla

24 meses (2016–2017) pronosticados uno a uno, reentrenando cada mes; 288 pronósticos por modelo.

| Modelo (mejor combinación) | MAE por serie y mes (USD) | Desvío del total |
|---|---|---|
| **Perfil estacional jerárquico (modelo propio)** | **2.310** | −5,1 % |
| SVR (Grid Search) | 2.366 | −18,2 % |
| Random Forest (Random Search) | 2.390 | −4,9 % |
| KNN (Random Search) | 2.436 | −5,5 % |
| HistGradientBoosting (bayesiana) | 2.439 | −4,1 % |
| Árbol de decisión (bayesiana) | 2.480 | −6,8 % |
| XGBoost (bayesiana) | 2.481 | −4,8 % |
| Lasso (genética) | 2.501 | −6,1 % |
| Ridge (Random Search) | 2.514 | −6,4 % |
| Ingenuo estacional (repetir el año anterior) | 2.728 | −19,6 % |
| Theta | 2.756 | −0,2 % |

**Veredicto: el modelo propio no es superior.** Tiene el menor error, pero la diferencia frente a SVR
(−55 USD, intervalo del 95 % [−218, +134]) no es estadísticamente significativa, su ventaja viene
toda de 2016 y no se repite en los tres conjuntos adicionales (tercero en turismo, noveno en
hospital, décimo en M3). Es significativamente mejor que repetir el año anterior. Detalle en el
informe y en `MODEL_CARD.md`.

**Proyección 2018:** 886.866 USD (+21 % frente a 2017).

## El modelo propio

`src/original_model.py` — un perfil estacional común, estimado con todas las series, que cada serie
corrige con sus ventas del mismo mes del año anterior mediante un peso propio. Los pesos se aprenden
minimizando el error con una penalización que los acerca a un peso común; las reglas de
actualización se derivan del objetivo y convergen al mínimo (demostración en el capítulo 5 del
informe). Solo usa `numpy` y es compatible con scikit-learn. Pruebas: `tests/test_original_model.py`.

## Tablero

Tres secciones, cada una con sus pestañas:

| Sección | Contenido |
|---|---|
| **Contexto** (`/`) | Resumen, problema y objetivos, datos y diccionario, metodología, marco teórico, limitaciones y ética, conclusiones |
| **EDA** (`/eda`) | Explorador con filtros globales (categoría, región, fechas, descuento): distribución, correlación, explorador de ejes, faltantes y atípicos, descomposición, ACF/PACF y Dickey–Fuller. Análisis exploratorio completo y ETL |
| **Modelos** (`/modelos`) | Tabla de líderes, las 28 combinaciones y los optimizadores; predicción en vivo con intervalo y explicación; diagnóstico; Model Confidence Set y pruebas; el modelo propio; ficha del modelo; proyección 2018 y pronóstico animado |

El tablero no entrena nada: carga los modelos de `artifacts/` y las tablas de `experiments/resultados/`.

Hay además un **Asistente IA** flotante (botón 💬) que traduce la proyección en metas, inventario,
alertas y presupuesto. Usa la librería `openai` con Google Gemini por defecto. Para activarlo, copia
`.env.example` como `.env` y pon tu clave (gratis en [Google AI Studio](https://aistudio.google.com)).
El archivo `.env` no se sube al repositorio. Sin clave, el tablero funciona igual.

## Cómo correr el tablero

Requiere Python 3.10.

```powershell
python -m venv .venv
.venv\Scripts\activate       # en macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
copy .env.example .env       # opcional, para el Asistente IA
python app.py
```

Después abre **http://127.0.0.1:8060**. En producción (Render): `gunicorn app:server`, con la
configuración de `render.yaml`.

## Cómo repetir los experimentos

Los resultados ya vienen en el repositorio. Para recalcularlos, desde la raíz:

```powershell
pip install -r requirements-dev.txt
python data/generate_data.py              # CSV crudo -> tabla mensual
python experiments/datos_adicionales.py   # descarga los tres conjuntos adicionales (Zenodo)
python experiments/ejecutar.py            # validación anidada de todos los modelos (~2 h)
python experiments/analizar.py            # tablas, pruebas y veredicto
python experiments/robustez.py            # ruido, menos datos, semillas, escalabilidad
python experiments/artefactos.py          # modelos finales, metadata.json y proyección 2018
python -m pytest -q tests                 # pruebas del modelo propio (también en GitHub Actions)
```

El protocolo se registró antes de ejecutar los experimentos: `experiments/protocolo.md`, etiqueta de
Git `protocolo-v1`. Semilla global: 42.

## Informe (Jupyter Book)

En la carpeta `book/`. Once capítulos: introducción; planteamiento; datos (ETL y EDA); modelos de
referencia; literatura y novedad; modelo original; validación estadística; evaluación ampliada;
dashboard; conclusiones; reproducibilidad y bibliografía.

```powershell
jupyter-book build book/        # -> book/_build/html/index.html
```

**Publicarlo.** El informe se sirve desde la rama `gh-pages`, que solo contiene el HTML más un
archivo `.nojekyll`:

```powershell
git worktree add ..\gh-pages gh-pages
cd ..\gh-pages
git rm -r -q .
Copy-Item -Recurse ..\SuperstorePython\book\_build\html\* .
New-Item .nojekyll -ItemType File -Force
git add -A; git commit -m "Publica el libro"; git push origin gh-pages
cd ..\SuperstorePython; git worktree remove ..\gh-pages
```

## Estructura

```
├── app.py                    # Tablero Dash (use_pages=True); expone server para gunicorn
├── pages/                    # contexto.py, eda.py, modelos.py: una página por sección
├── tabs/                     # contenidos reutilizados por las páginas y el Asistente IA
├── src/
│   ├── original_model.py     # modelo propio
│   ├── estadistica.py        # Diebold–Mariano, Giacomini–White, MCS, bootstrap estacionario, Holm…
│   ├── optimizacion.py       # Grid, Random, bayesiana (Optuna) y genética (DEAP)
│   ├── explicacion.py        # valores de Shapley exactos y sustituto local tipo LIME
│   ├── data.py · plots.py    # carga de resultados y figuras del tablero
├── experiments/
│   ├── protocolo.md          # protocolo registrado
│   ├── config.py · ejecutar.py · analizar.py · robustez.py · artefactos.py
│   ├── datos_adicionales.py  # turismo, hospital y M3 (Monash, Zenodo)
│   ├── literatura/           # búsqueda bibliográfica documentada
│   └── resultados/           # predicciones, tabla maestra, pruebas, veredicto
├── artifacts/                # 8 modelos (.joblib) y metadata.json
├── model/                    # variables (features.py), métricas y resumen para el tablero
├── data/                     # datos crudos, procesados, adicionales y DATA_STATEMENT.md
├── tests/                    # pruebas unitarias del modelo propio
├── book/                     # informe (Jupyter Book)
├── paper/                    # artículo (plantilla Springer)
├── assets/                   # estilos y efectos de la interfaz
├── MODEL_CARD.md · CITATION.cff · LICENSE · render.yaml
├── requirements.txt          # dependencias del tablero (versiones exactas)
└── requirements-dev.txt      # + experimentos, pruebas y libro
```

## Datos

Sample Superstore es un conjunto de ejemplo que Tableau distribuye con su software; se usa con fines
académicos. Los conjuntos adicionales provienen del Monash Time Series Forecasting Repository
(licencia CC BY 4.0). Procedencia, licencias y sesgos en `data/DATA_STATEMENT.md`.
