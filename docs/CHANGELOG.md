# Historial de cambios

## v2.0-v5 (21-11-2026)
- Nueva carpeta [`v5/`](../v5/README.md) con el código de la **corrida v5** (`extract_epochs.py`, `models.py`, `evaluate.py`), los scripts de descarga y de control con etiquetas de entrenamiento permutadas, y la **ablación de banda** exploratoria: A1 y A2 (EEGNet en 0,5–3 Hz y 8–30 Hz), B1 (MDM en 0,5–3 Hz) y C1 (Tensor-CSPNet en banda baja; `c1_runner_v2.py`, `score_c1.py`).
- Tablas agregadas de la v5 y de la ablación (Holm sobre 36 pruebas), sin datos por participante.
- `v5/PROCEDENCIA.md`: SHA-256 de cada archivo ejecutado y descripción de las adaptaciones de la versión pública (solo rutas e identificadores internos).
- La evaluación v2 (raíz del repositorio, `v1.0-articulo`) queda como piloto no citable.

## v1.0-articulo (17-11-2026)
Versión que acompaña al artículo. Incluye todo lo anterior y además:
- **Corrección de la configuración de las redes en 001-2019.** EEGNet, ShallowConvNet y DeepConvNet se reejecutaron con **pérdida ponderada por clase** (wᶜ = n / (K·nᶜ)) y **`CosineAnnealingLR(T_max = 20)`**, como indica la configuración establecida, en las 10 particiones y en el control por permutación (k = 0, 1). Fueron 36 corridas en Colab T4, con 0,30 unidades de cómputo.

  | Red (001-2019) | Exactitud balanceada, sin ponderación ni programador | Exactitud balanceada final | MCC, sin ponderación ni programador | MCC final |
  |---|---|---|---|---|
  | EEGNet | 0,196 ± 0,010 | 0,199 ± 0,013 | −0,005 ± 0,013 | −0,001 ± 0,018 |
  | ShallowConvNet | 0,202 ± 0,009 | 0,204 ± 0,011 | 0,002 ± 0,013 | 0,005 ± 0,015 |
  | DeepConvNet | 0,199 ± 0,013 | 0,201 ± 0,015 | 0,000 ± 0,019 | 0,001 ± 0,019 |

  **No cambió ninguna conclusión.** Los rangos y la prueba de Friedman con bloques = conjuntos de evaluación no cambiaron (χ² = 10,48; p = 0,063; p por permutación = 0,026; DC = 4,35). Con bloques alternativos, p pasó de 0,463 a 0,565 (N = 30) y de 0,605 a 0,518 (N = 35); en 001-2019 por separado, de 0,150 a 0,227.
- Semilla fija en la prueba bayesiana (`random_state = 20261113`): la estadística regenerada desde `preds/` es idéntica byte a byte.
- Terminología del artículo: «conjunto de evaluación», «configuración establecida», «MDM» y etapas Análisis → Diseño → Implementación → Pruebas.
- Cuaderno reproducible documentado (`TESIS_pipeline_reproducible.ipynb`, sin ejecutar), script de verificación y documentación completa.
- Correspondencia con la versión final del artículo (noviembre 2026).
- **Figura 1** del artículo (flujo de una BCI motora) en `figures/Figura_pipeline_BCI.png` y script `scripts/make_fig_pipeline_bci.py`. El alcance de la evaluación offline cubre los pasos 3 a 5; el paso 6 y el lazo cerrado en línea quedan fuera del alcance.
- Numeración de figuras alineada con la versión final del artículo: Figura 2 (etapas del protocolo), Figura 3 (PRISMA), Figura 4 (diagrama DC de la partición única) y Figura 5 (cajas de exactitud balanceada).
- Punto decimal (no coma) en el eje vertical de las figuras de cajas, como en el artículo.

## v1 — particiones repetidas (16-11-2026, 19:20–20:50, hora de Bogotá)
- Se pasó de una partición única a **10 particiones 80/20 disjuntas por participante** (semillas 20261113 + k, k = 0…9; k = 0 es la partición establecida).
- Se añadieron métricas por partición y por participante, matrices de confusión, F1 macro, kappa y MCC.
- Se añadieron: Friedman con p por permutación e Iman-Davenport; Nemenyi y diagramas DC con bloques = conjuntos de evaluación y bloques alternativos; Wilcoxon-Holm; prueba bayesiana con ROPE; comparación frente al azar; control por permutación con k = 0, 1.
- Datos re-extraídos desde Zenodo con los manifiestos del estudio. Con k = 0, los clásicos reproducen exactamente la v3.
- 001-2020 no se pudo descargar (servidor no disponible).
- Las redes de 001-2019 usaron la configuración del código v3 (sin ponderación ni programador). Esos valores se conservan en `tables/anteriores_sin_ponderacion_ni_scheduler/`.

## v3 — partición única (estudio de David Rodriguez Chamorro, 13–14-11-2026)
- Una partición 80/20 disjunta por participante (semilla 20261113) en 6 conjuntos de evaluación (001-2017 ME y MI; 001-2019; 001-2020 gel, agua y seco), con 6 modelos.
- Control por permutación con la misma semilla.
- Friedman con 6 bloques y diagrama DC (Figura 4 del artículo).
