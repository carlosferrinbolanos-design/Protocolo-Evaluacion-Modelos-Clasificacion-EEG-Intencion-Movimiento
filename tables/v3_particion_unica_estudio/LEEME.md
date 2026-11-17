# Partición única v3 (estudio de David Rodriguez Chamorro, 13–14-11-2026)

Resultados de la ejecución v3 del estudio con la partición establecida (semilla 20261113) en los **6 conjuntos de evaluación**, incluidos los tres de 001-2020. Sirven para verificar las **Tablas 1 y 3** y la **Figura 3** del artículo. Se copian sin modificaciones desde la carpeta del estudio (`Correccion_pipeline_EEG_v2_2026-11-11/cierre_metodologico_ferrin_v2/` y `control_permutacion_v2/`).

| Archivo | Contenido |
|---|---|
| `RESULTADOS_ETIQUETAS_REALES_CONSOLIDADOS_V2.csv` | Exactitud balanceada y F1 macro con etiquetas reales: 36 filas (6 conjuntos de evaluación × 6 modelos), con participantes de prueba y configuración |
| `RESUMEN_METRICAS_REALES_V2.csv` | Resumen de las mismas métricas |
| `CSP_FBCSP_PERMUTACION_RESUMEN.csv`, `RIEMANNIAN_MDM_PERMUTACION_RESUMEN.csv` | Control por permutación de los modelos clásicos |
| `CONTROL_PERMUTACION_PROFUNDO_CONSOLIDADO_V2.csv` | Control por permutación de las redes |

En esta ejecución, las redes de 001-2019 y 001-2020 se entrenaron **sin ponderación ni programador**, porque el código v3 no los aplicaba (ver `docs/CHANGELOG.md`). La columna `stratum` contiene el nombre del conjunto de evaluación.
