# Análisis estadístico

Código: `scripts/04_analyze.py`, `scripts/04b_pairwise_LME.py`, `scripts/05_tables.py`; cuaderno P-2 a P-6. Referencias: Demšar (2006), García y Herrera (2008), Benavoli et al. (2017).

## Unidades de análisis (bloques)
| Bloques | N | Independencia | Uso |
|---|---|---|---|
| **Conjuntos de evaluación** (media de las 10 particiones) | 3 (001-2017 ME, 001-2017 MI, 001-2019) | Conjuntos distintos, aunque ME y MI comparten participantes | **Análisis principal** (Demšar). N = 3 es muy inferior al recomendado (≥ 10), así que la potencia es baja y la DC es grande |
| Partición × conjunto de evaluación | 30 | **No independientes**: las particiones comparten participantes de entrenamiento, y en 001-2019 dos pares de semillas repiten el conjunto de prueba | Sensibilidad; valores p descriptivos |
| Participante × conjunto de evaluación | Participantes que estuvieron en prueba al menos una vez (media de sus particiones) | No independientes (mismo participante en ME y MI) | Sensibilidad |
| Solo sanos (001-2017), partición × conjunto de evaluación | 20 | Igual que arriba | Subgrupo |
| **Solo lesión medular (001-2019)**, particiones | 10 | Igual que arriba | **Análisis separado del conjunto de evaluación clínico** |

Solo entran bloques en los que los seis modelos tienen resultado. 001-2020 no entra (ver [LIMITACIONES.md](LIMITACIONES.md)).

## Prueba global de Friedman
- Rangos dentro de cada bloque (1 = mayor métrica; empates promediados): χ²_F = 12N / (k(k+1)) · Σⱼ (R̄ⱼ − (k+1)/2)².
- p asintótica con χ²(k − 1) y corrección de **Iman-Davenport**: F_F = (N − 1)χ²_F / (N(k − 1) − χ²_F), con F(k − 1, (k − 1)(N − 1)).
- **p por permutación** (Monte Carlo): se permutan los rangos dentro de cada bloque. Se usan 20 000 permutaciones con bloques = conjuntos de evaluación y 5 000 en los demás casos, con generador `default_rng(20261113)`, y p = (n_≥ + 1)/(n_perm + 1). Con N pequeño, esta p es la referencia.

## Nemenyi y diagrama de diferencia crítica (DC)
- DC = q_α √(k(k + 1)/(6N)), con q₀.₀₅ = 2,850 para k = 6. Resulta DC = 4,35 con N = 3, 1,38 con N = 30 y 2,38 con N = 10.
- En el diagrama, una barra roja une los modelos cuya diferencia de rango promedio es menor que la DC (sin diferencia significativa). Las figuras están solo en español, en PNG a 300 ppp, SVG y PDF (`figures/DC_*`).

## Comparaciones por pares
- **Wilcoxon de rangos con signo** (`zero_method='wilcox'`) para los 15 pares, con **corrección de Holm**. Bloques = partición × conjunto de evaluación, y por separado 001-2019 con bloques = particiones.
- **Prueba bayesiana de rangos con signo** (`baycomp.two_on_multiple`). ROPE = 0,01 en exactitud balanceada y 0,02 en MCC. Se informan P(A mejor), P(equivalencia práctica) y P(B mejor).

## Comparación frente al azar
Wilcoxon unilateral (`alternative='greater'`) de (exactitud balanceada − 1/K) sobre las 10 particiones de cada conjunto de evaluación × modelo, con Holm sobre las 18 combinaciones. Azar: 1/7 en 001-2017 y 1/5 en 001-2019.

## Control por permutación (fuga de información)
Con k = 0 y k = 1, las etiquetas de **entrenamiento** se permutan dentro de cada participante. Las de prueba no se tocan, y las particiones, la normalización y los hiperparámetros son idénticos. Semillas de la permutación: clásicos `20261113 + k`; redes `20261113 + k + Σ ord(modelo + conjunto de evaluación)`. Si el desempeño permutado está en el nivel de azar, se descarta que el resultado real provenga de fuga de información.
