# Reproducibilidad

Este documento explica cómo verificar los resultados del artículo de forma independiente. Hay tres niveles de verificación, de menor a mayor costo.

| Nivel | Qué se verifica | Requiere | Tiempo | Unidades de cómputo Colab |
|---|---|---|---|---|
| **1. Estadística desde predicciones** | Que todas las tablas, pruebas y figuras salen exactamente de las 216 predicciones por época publicadas (`preds/`) | CPU; sin datos EEG | < 5 min | 0 (local) o < 0,2 |
| **2. RAPIDO (k = 0)** | Que la partición establecida reproduce los resultados v3 del estudio (clásicos idénticos a 4 decimales) | Colab con T4; descarga de unos 28 GB | 1,5–2 h | 3–4 |
| **3. COMPLETO (k = 0…9 + permutación)** | Que el reentrenamiento completo reproduce las medias publicadas (clásicos exactos; redes dentro de ± 0,02) | Colab con T4 | 8–10 h (1 VM) | 12–18 |

Los tiempos se estimaron con las ejecuciones del 16-11-2026, en las que todo el estudio, repartido en 4 VMs más el computador de trabajo, usó 3,21 unidades de cómputo. Una sola VM es más lenta porque FBCSP+SVM en 001-2017 tarda unos 100–130 s por partición.

## Semillas
| Elemento | Semilla |
|---|---|
| Partición k | `numpy.random.default_rng(20261113 + k + Σ ord(nombre_del_conjunto))`; k = 0 es la partición establecida (Tabla 1) |
| Permutación (clásicos) | `default_rng(20261113 + k)` dentro de cada participante de entrenamiento |
| Permutación (redes) | `default_rng(20261113 + k + Σ ord(modelo + nombre_del_conjunto))` |
| Redes | `torch.manual_seed`, `np.random.seed` y `random.seed` con `20261113 + k + Σ ord(modelo + nombre_del_conjunto + control)`; `torch.Generator` del DataLoader con la misma semilla |
| p por permutación de Friedman | un único `default_rng(20261113)` compartido, en el orden de llamadas de `04_analyze.py` |
| Prueba bayesiana (baycomp) | `random_state = 20261113` en cada comparación |

## Límites del determinismo
- **Modelos clásicos:** deterministas. Con k = 0, los 9 valores (3 modelos × 3 conjuntos de evaluación) coinciden exactamente con la ejecución v3 del estudio, en dos entornos distintos (scikit-learn 1.6.1 y 1.9.1).
- **Redes en GPU:** se usa `cudnn.benchmark = True`, como en el estudio, y algunas operaciones de cuDNN no son deterministas. Al reentrenar con la misma semilla, **la exactitud balanceada puede variar hasta 0,0125** (variación observada con k = 0 entre la ejecución v3 y la reejecución con la configuración v3; ver `tables/reproduccion_semilla0_vs_david.csv`). Por eso la verificación de las redes usa una tolerancia de ± 0,02 en las medias.
- **Estadística:** determinista. Regenerar las tablas desde `preds/` produce archivos **idénticos byte a byte** (comprobado el 16-11-2026).

## Entorno fijado
`requirements.txt` y `environment.yml` (Python 3.13). Los resultados se generaron con:
- **VMs de Colab:** numpy 2.4.6, scipy 1.16.3, scikit-learn 1.6.1, pandas 2.2.3, matplotlib 3.10.0, mne 1.13.2, pyriemann 0.12 y torch 2.14.1+cu130.
- **Computador de trabajo** (clásicos de 001-2019 y estadística): numpy 2.5.5, scipy 1.18.1, scikit-learn 1.9.1, pandas 3.1.1, matplotlib 3.11.2, mne 1.13.2, pyriemann 0.12 y baycomp 1.0.3.

## Lista de verificación para un revisor externo
1. [ ] Clonar el repositorio y comprobar que **no** contiene datos EEG (`git ls-files | grep -E "\.(gdf|mat|npz|npy|edf|fif)$"` no devuelve nada).
2. [ ] Comprobar las sumas SHA-256 de las tablas publicadas: `sha256sum -c docs/SHA256SUMS.txt`.
3. [ ] **Nivel 1:** `pip install -r requirements.txt`, luego `RESULTADOS=$(mktemp -d)`, copiar `preds/` a esa carpeta, crear `tables/` y `figures/` dentro y ejecutar `scripts/04_analyze.py`, `scripts/04b_pairwise_LME.py` y `scripts/05_tables.py` con `RESULTADOS` exportada. Después ejecutar `python scripts/verificar_resultados.py --regenerado $RESULTADOS/tables --estricto` y confirmar el mensaje `RESULTADO: VERIFICADO`. Alternativa: abrir el cuaderno con `USAR_PREDICCIONES_GUARDADAS = True`.
4. [ ] Comprobar que las tablas regeneradas son idénticas: `cmp $RESULTADOS/tables/TABLAS_ES_estadistica.md tables/TABLAS_ES_estadistica.md`.
5. [ ] **Nivel 2:** abrir `TESIS_pipeline_reproducible.ipynb` en Colab con T4, dejar `MODO = "RAPIDO"`, ejecutar todo y confirmar que la celda P-7 imprime `OK: k = 0 reproduce los resultados clásicos v3`.
6. [ ] Revisar en la celda B-1 que la partición k = 0 coincide con la Tabla 1 del artículo: S08, S13, S15 (ME); S08, S10, S14 (MI); P06, P10 (001-2019).
7. [ ] Confirmar que ningún participante de prueba aparece en entrenamiento (la función `particion` lo verifica con `assert`).
8. [ ] Revisar el control por permutación (`tables/permutation_control.csv`): la exactitud balanceada permutada debe quedar en el nivel de azar (1/7 y 1/5).
9. [ ] **Nivel 3 (opcional):** `MODO = "COMPLETO"`; la celda P-7 compara las 18 medias con las publicadas (clásicos ± 5·10⁻⁴; redes ± 0,02).
10. [ ] Contrastar cada número del artículo con su archivo según la tabla de correspondencia del `README.md`.

## Qué no se puede reproducir exactamente
- Los valores individuales de las redes por partición: ver los límites del determinismo en GPU.
- Los conjuntos de evaluación de 001-2020: no se pudieron descargar el 16-11-2026 (ver [LIMITACIONES.md](LIMITACIONES.md)).
