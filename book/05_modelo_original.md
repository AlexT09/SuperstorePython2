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

# Modelo original

*Perfil estacional con pesos jerárquicos*

## La limitación de la que parte

Cada serie tiene 48 meses. Descontado el primer año, que solo aporta historia, quedan tres
observaciones de cada mes del año. Con tres datos, la estacionalidad de una serie individual se
estima con mucho ruido, y los modelos generales tienen que aprenderla junto con todo lo demás.

Dos resultados del capítulo anterior lo muestran:

- El ingenuo estacional, que usa solo la historia propia de cada serie, es el peor modelo (MAE de
  2.728 USD).
- En todos los modelos de referencia, la importancia de las variables se concentra en el mes del año
  y en las ventas del mismo mes del año anterior. Random Forest dedica 300 árboles a aprender algo
  que se puede calcular directamente.

La pregunta de diseño es entonces cuánta información tomar del grupo y cuánta de la propia serie.
El modelo responde con un peso por serie, y aprende esos pesos de los datos.

## Formulación

Sea $z_{s,t}$ el índice de la serie $s$ en el mes $t$ (ventas del mes divididas por el promedio de
los 12 meses anteriores), $m(t)$ el mes del año y $r_{s,t}=z_{s,t-12}^{*}$ el índice del mismo mes
del año anterior, medido con el nivel actual.

**Perfil común.** Es la media del índice en cada mes del año, sobre todas las series y años de
entrenamiento:

$$
P_m=\frac{1}{|\{(s,t):m(t)=m\}|}\sum_{(s,t):\,m(t)=m} z_{s,t},\qquad m=1,\dots,12.
$$ (eq-perfil)

**Pronóstico.** Cada serie corrige el perfil con su propio rezago, con un peso $w_s\in[0,1]$:

$$
\hat z_{s,t}=P_{m(t)}+w_s\,\bigl(r_{s,t}-P_{m(t)}\bigr),\qquad \hat y_{s,t}=\max(0,\ \hat z_{s,t}\,\ell_{s,t}),
$$ (eq-pronostico)

donde $\ell_{s,t}$ es el promedio de los 12 meses anteriores. Con $w_s=0$ la serie sigue el perfil
común; con $w_s=1$ el modelo es el ingenuo estacional.

**Aprendizaje de los pesos.** Con $a_{s,t}=z_{s,t}-P_{m(t)}$ y $b_{s,t}=r_{s,t}-P_{m(t)}$, los pesos
y un peso común $w_0$ minimizan

$$
J(w_1,\dots,w_S,w_0)=\sum_{s=1}^{S}\sum_{t}\bigl(a_{s,t}-w_s\,b_{s,t}\bigr)^2+\lambda\sum_{s=1}^{S}(w_s-w_0)^2,
\qquad 0\le w_s\le 1.
$$ (eq-objetivo)

El primer término es el error de pronóstico. El segundo acerca cada peso al peso común: es un
agrupamiento parcial de los pesos. Con $\lambda$ grande todas las series comparten un peso; con
$\lambda=0$ cada serie usa el suyo.

**Reglas de actualización.** Sean $S_{ab,s}=\sum_t a_{s,t}b_{s,t}$ y $S_{bb,s}=\sum_t b_{s,t}^2$.
Derivando {eq}`eq-objetivo` respecto de $w_s$ con $w_0$ fijo e igualando a cero:

$$
-2S_{ab,s}+2w_sS_{bb,s}+2\lambda(w_s-w_0)=0
\ \Longrightarrow\
w_s\leftarrow\operatorname{recortar}_{[0,1]}\!\left(\frac{S_{ab,s}+\lambda w_0}{S_{bb,s}+\lambda}\right).
$$ (eq-ws)

Derivando respecto de $w_0$ con los $w_s$ fijos:

$$
-2\lambda\sum_s(w_s-w_0)=0\ \Longrightarrow\ w_0\leftarrow\frac{1}{S}\sum_{s=1}^{S}w_s.
$$ (eq-w0)

La regla {eq}`eq-ws` se lee así: el peso de una serie es un promedio entre su propio peso de mínimos
cuadrados, $S_{ab,s}/S_{bb,s}$, y el peso común, y la evidencia de la serie, $S_{bb,s}$, decide cuál
domina.

## Algoritmo

```text
Entrada: índices z(s,t), rezagos r(s,t), mes m(t), fuerza de agrupamiento lam, tolerancia tol
Salida:  perfil P[1..12], pesos w_s de cada serie, peso común w0

 1. P[m]  <- media de z(s,t) en los meses con m(t) = m, sobre todas las series      (ecuación del perfil)
 2. a_st  <- z(s,t) - P[m(t)]          b_st <- r(s,t) - P[m(t)]
 3. S_ab[s] <- suma_t a_st * b_st      S_bb[s] <- suma_t b_st^2
 4. w0 <- recortar( suma_s S_ab[s] / suma_s S_bb[s] )      w_s <- w0
 5. repetir
 6.     w_s <- recortar( (S_ab[s] + lam * w0) / (S_bb[s] + lam) )   para cada serie s
 7.     w0  <- promedio de los w_s
 8. hasta que J baje menos de tol * (1 + J)            (criterio de parada)
 9. devolver P, w, w0
```

## Sustento teórico

**Proposición 1 (descenso monótono y convergencia).** Si $\lambda>0$ y alguna serie tiene
$S_{bb,s}>0$, el algoritmo produce una sucesión de valores de $J$ que no aumenta y converge al único
mínimo de {eq}`eq-objetivo`.

*Demostración.* (a) *Cada paso es un mínimo exacto.* Con $w_0$ fijo, $J$ es una suma de funciones
cuadráticas de una variable, una por serie, con coeficiente principal $S_{bb,s}+\lambda>0$. El mínimo
de una cuadrática convexa en un intervalo es su mínimo libre recortado al intervalo, que es
{eq}`eq-ws`. Con los $w_s$ fijos, $J$ es cuadrática convexa en $w_0$ y su mínimo es {eq}`eq-w0`, que
cae en $[0,1]$ por ser un promedio. Como cada paso minimiza $J$ en un bloque dejando el otro fijo,
$J$ no puede aumentar. (b) *El mínimo es único.* $J$ es una forma cuadrática. Su parte de segundo
orden es $\sum_s S_{bb,s}w_s^2+\lambda\sum_s(w_s-w_0)^2$, que solo se anula si $w_s=w_0$ para todo
$s$ y $S_{bb,s}w_s^2=0$ para todo $s$; con algún $S_{bb,s}>0$ eso obliga a $w_0=0$ y a todos los
$w_s=0$. Luego $J$ es estrictamente convexa, y sobre el conjunto convexo y cerrado
$[0,1]^S\times\mathbb{R}$ tiene un único mínimo. (c) *Convergencia.* $J\ge0$ y no aumenta, así que
converge. Para una función estrictamente convexa y diferenciable con restricciones separables por
bloque, el descenso por bloques con minimización exacta converge al mínimo global (Tseng, 2001
{cite}`tseng2001`). $\blacksquare$

**Proposición 2 (casos límite).** Cuando $\lambda\to\infty$, todos los pesos tienden a un mismo
valor, el peso de mínimos cuadrados del panel, $\sum_sS_{ab,s}/\sum_sS_{bb,s}$ recortado a $[0,1]$.
Cuando $\lambda=0$, $w_s=\operatorname{recortar}(S_{ab,s}/S_{bb,s})$: mínimos cuadrados por serie.

*Demostración.* En {eq}`eq-ws`, al crecer $\lambda$ el cociente tiende a $w_0$, luego $w_s=w_0$ para
todo $s$; sustituyendo en {eq}`eq-objetivo` queda $\sum_s\sum_t(a_{s,t}-w_0b_{s,t})^2$, cuyo mínimo es
el indicado. Con $\lambda=0$ la penalización desaparece y cada serie se resuelve por separado.
$\blacksquare$

**Supuestos y su plausibilidad.** El modelo supone que las series comparten el patrón estacional.
En Sample Superstore es razonable: todas son líneas de una misma cadena con el mismo calendario
comercial. En un panel de series sin relación entre sí el supuesto falla, y el capítulo 7 muestra
que entonces el modelo pierde.

**Complejidad.** El ajuste recorre los datos una vez para calcular el perfil y las sumas, $O(n)$, y
cada iteración cuesta $O(S)$; en total $O(n+S\cdot I)$ con $I$ iteraciones. La memoria es
$O(S+12)$. El pronóstico es una consulta y una combinación lineal: $O(1)$ por fila.

## Verificación numérica

```{code-cell} ipython3
:tags: [hide-input]
from common import ORIGINAL, barras, lineas, md, res, show, tabla, usd
import pandas as pd

conv = res("convergencia.csv")
meta = res("metadata.json")["modelos"][ORIGINAL]
show(lineas(conv.iteracion, {"J": conv.J}, "Valor del objetivo en cada iteración", "Iteración", "J"))
md(f"Con toda la historia (432 filas), el objetivo baja de {conv.J.iloc[0]:.4f} a {conv.J.iloc[-1]:.4f} en "
   f"{len(conv) - 1} iteraciones y nunca aumenta, como establece la proposición 1. "
   f"El valor de λ elegido por la búsqueda bayesiana es {meta['hiperparametros']['lam']:.2f}.")
```

```{code-cell} ipython3
:tags: [hide-input]
e = res("escalabilidad.csv")
t = e.pivot(index="filas", columns="modelo", values="ajuste_ms").round(2).reset_index().rename(columns={"filas": "Filas"})
tabla(t)
```

```{code-cell} ipython3
:tags: [hide-input]
exp = e.groupby("modelo").exponente_ajuste.first()
md("Tiempo de ajuste en milisegundos al replicar los datos 1, 4, 16 y 64 veces. El exponente empírico "
   "(pendiente en escala log-log) es " + ", ".join(f"{v:.2f} para {k}" for k, v in exp.items()) +
   ". Un exponente igual o menor que 1 es consistente con un costo que crece, como mucho, de forma lineal "
   "con el número de filas; queda por debajo de 1 porque con estos tamaños todavía pesa el costo fijo de "
   "cada ajuste.")
```

## Implementación

La clase está en `src/original_model.py`. Hereda de `BaseEstimator` y `RegressorMixin`, por lo que
funciona con `Pipeline`, `clone`, `GridSearchCV` y `cross_val_score`. Solo usa `numpy`. El núcleo del
ajuste:

```python
# sumas por serie: lo único que necesita el algoritmo
a, b = y - p, r - p
Saa = np.bincount(idx, a * a, S)
Sab = np.bincount(idx, a * b, S)
Sbb = np.bincount(idx, b * b, S)

w0 = float(np.clip(Sab.sum() / Sbb.sum(), 0, 1))      # peso global de mínimos cuadrados
w = np.full(S, w0)
for _ in range(self.max_iter):
    # regla de los pesos: mínimo exacto de J en cada w_s con w0 fijo, recortado a [0, 1]
    w = np.clip((Sab + self.lam * w0) / (Sbb + self.lam), 0, 1)
    # regla del peso común: mínimo exacto de J en w0 con los w_s fijos
    w0 = float(w.mean())
    self.historial_.append(self.objetivo(w, w0, Saa, Sab, Sbb, self.lam))
    if self.historial_[-2] - self.historial_[-1] <= self.tol * (1 + self.historial_[-1]):
        break
```

El método `explicar` devuelve, para cada pronóstico, tres sumandos exactos: la base (media del
perfil), lo que aporta el mes del año y lo que aporta el rezago propio. Es la explicación local que
muestra el tablero.

## Pruebas unitarias

`tests/test_original_model.py` contiene 11 pruebas, que se ejecutan en cada *push* con GitHub Actions:

| Prueba | Qué verifica |
|---|---|
| Recupera la solución conocida | En datos sintéticos generados con el propio modelo, recupera el perfil y el peso de cada serie |
| La pérdida no aumenta | $J$ es no creciente para cuatro valores de $\lambda$ (proposición 1) |
| Converge al mínimo | El resultado coincide con la solución exacta del sistema lineal de las condiciones de primer orden |
| Casos límite | $\lambda$ grande da un solo peso; $\lambda=0$ coincide con mínimos cuadrados por serie (proposición 2) |
| Pesos entre 0 y 1 | El recorte se respeta aunque los datos empujen fuera |
| Reproducible | Dos ajustes dan exactamente el mismo resultado (el algoritmo es determinista) |
| `clone` | Conserva los hiperparámetros |
| Entrada inválida | Rechaza valores faltantes, columnas ausentes, $\lambda$ negativo y tamaños distintos |
| Serie no vista | Usa el peso común |
| Explicación | Los tres sumandos suman la predicción |
| scikit-learn | Funciona con `cross_val_score` y `GridSearchCV` |

## Pesos aprendidos

```{code-cell} ipython3
:tags: [hide-input]
w = res("pesos_original.csv")
show(barras([s.replace("|", " · ") for s in w.serie], list(w.peso), "Peso del rezago propio en cada serie",
            "Peso (0 = solo perfil común; 1 = ingenuo estacional)", alto=380))
md(f"Los pesos van de {w.peso.min():.2f} a {w.peso.max():.2f}, con media {w.peso.mean():.2f}. Son bajos: "
   "las series se pronostican casi por completo con el perfil común.")
```

## Aporte de cada componente

Cada variante cambia una pieza y mantiene el resto. Se evalúa con el mismo protocolo que el modelo
completo (24 meses de origen móvil) y las mismas pruebas, con corrección de Holm sobre las seis
variantes.

```{code-cell} ipython3
:tags: [hide-input]
a = res("ablacion.csv")
a = a[a.conjunto == "superstore"]
tabla(pd.DataFrame({
    "Variante": a.variante, "MAE (USD)": a.MAE.map("{:,.0f}".format),
    "Variante − completo [IC 95 %]": ["—" if pd.isna(d) else f"{-d:+,.0f} [{-hi:+,.0f}, {-lo:+,.0f}]" for d, lo, hi in zip(a.dif, a.ic_lo, a.ic_hi)],
    "Diebold–Mariano p / p Holm": ["—" if pd.isna(x) else f"{x:.3f} / {h:.3f}" for x, h in zip(a.p_DM, a.p_DM_holm)],
    "Desvío del total": a.desvio_total_pct.map("{:+.1f} %".format)}))
```

- **El perfil común es la pieza decisiva.** Con un perfil por serie el MAE sube 650 USD
  (p ajustado = 0,001): es el único efecto significativo.
- **Los pesos por serie no ayudan en este conjunto.** Con un peso único, o sin rezago propio, el MAE
  es 31 USD *menor* que el del modelo completo. La diferencia no es significativa, pero va en contra
  del mecanismo propuesto: en Superstore, aprender un peso por serie añade ruido.
- **El promedio simple** de los dos pronósticos (peso 0,5) y el modelo **sin perfil** son peores y
  subestiman el total en 12 % y 13 %.

La conclusión es incómoda y se reporta tal cual: lo que funciona es la parte conocida del modelo, el
perfil estacional común, y no la regla nueva de los pesos. El capítulo 7 muestra que en otros
conjuntos la regla sí aporta.

## Diagnóstico de regresión

Las mismas métricas y gráficos de regresión que en el capítulo 3, ahora para el modelo original con su mejor
método de optimización.

```{code-cell} ipython3
:tags: [hide-input]
from common import diagnostico_regresion
mo = res("mejores.csv")
mo = mo[(mo.conjunto == "superstore") & (mo.modelo == ORIGINAL)].iloc[0]
diagnostico_regresion(ORIGINAL, mo.optimizador)
```
