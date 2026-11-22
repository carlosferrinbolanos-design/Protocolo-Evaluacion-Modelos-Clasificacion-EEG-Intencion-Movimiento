# Corrida v5 y ablación de banda (versión citada en el artículo)

Esta carpeta contiene el código de la **corrida v5** del pipeline y de la **ablación de banda exploratoria** (brazos A1, A2, B1 y C1). En ellas se basan todas las cifras del artículo desde la versión 4.x (noviembre de 2026). **No contiene datos EEG, épocas, predicciones por época ni identificadores de participantes.**

> **v5 frente a v2.** La raíz del repositorio (`v1.0-articulo`: cuaderno, `scripts/`, `preds/`, `tables/`) corresponde a la evaluación **v2**, que el 14-11-2026 se declaró **piloto no citable**. La v5 la reemplaza con la misma configuración para todos los conjuntos, filtrado pasa-altos sobre la señal continua, selección de canales por nombre y 001-2020 completo. Para verificar el artículo, use esta carpeta.

## Contenido
| Ruta | Qué hace |
|---|---|
| `code/extract_epochs.py` | Regenera las épocas desde los archivos públicos: canales por nombre, pasa-altos FIR de fase cero de 0.5 Hz sobre la señal continua, remuestreo a 256 Hz y épocas de −0.5 a 1.5 s (513 muestras), sin línea base ni rechazo por amplitud. |
| `code/models.py` | Los seis modelos, fijados antes de evaluar: CSP+LDA, FBCSP+SVM, MDM, EEGNet, ShallowConvNet y DeepConvNet (braindecode; 20 épocas, AdamW lr 1e-3, wd 1e-4, lote 64, pérdida ponderada por clase, programador coseno, sin parada temprana). |
| `code/evaluate.py` | Evaluación entre participantes: 10 particiones disjuntas por participante (round(20 %) de prueba, semillas 20261114 + k), exactitud balanceada por participante, 1000 permutaciones de etiquetas de prueba dentro de participante, IC 95 % por bootstrap de participantes (2000) y 200 reentrenamientos con etiquetas de entrenamiento permutadas (`permtrain`). |
| `scripts/` | Descarga (`download.sh`), descompresión segura (`unzip_all.sh`) y control `permtrain` (`run_permtrain*.sh`). |
| `ablacion_banda/code/ablation_v1_en_ejecucion.py` | Script que produjo A1, A2 y B1 (corrida 1, SHA-256 en `PROCEDENCIA.md`): `smoke`, `cross`, `permtrain`, `score`. |
| `ablacion_banda/code/ablation.py` | Versión ampliada del mismo protocolo, con subcomandos de verificación: `check_splits`, `merge`, `a0check`, `c1sanity` y `score`. |
| `ablacion_banda/code/c1/c1_runner_v2.py` | Brazo C1, Tensor-CSPNet (`spd_learn`) en banda baja: 4 subbandas (0.5–1.5, 1–2, 1.5–2.5 y 2–3 Hz; Butterworth de orden 4, `sosfiltfilt`), `n_patches = 4`, `tcn_channels = 16`, 20 épocas y AdamW. Mismas particiones y semillas que la v5; reanudable por partición. |
| `ablacion_banda/code/c1/smoke_c1_cpu.py` | Prueba de humo de C1 en CPU (forma y finitud con 11 y 61 canales; una partición de 001-2020 gel). |
| `ablacion_banda/code/c1/score_c1.py` | Script original que puntuó C1 con la misma función que A1, A2 y B1 (`evaluate.score_predictions`), más kappa y una validación contra A1 y B1 en 001-2017 ME. |
| `ablacion_banda/v5-ablacion.json`, `v5-ablacion-corrida1.json` | Configuración de la ablación. La de la corrida 1 se escribió a posteriori. |
| `tablas/` | Resúmenes agregados, sin datos por participante: `v5_cross_subject_summary_holm.csv` (6 modelos × 6 conjuntos), `ablation_summary_36.csv` (brazos y contrastes, Holm sobre 36 pruebas), `ablation_permtrain.csv` y `v5_timing.csv` (segundos por partición de las redes). |
| `requirements_v5_pip_freeze.txt` | Entorno exacto de la corrida (Python 3.12, `torch 2.11.0+cu128`, `braindecode 2.0.0`, `mne 1.13.2`, `pyriemann 0.12`, `scikit-learn 1.9.1`). |
| `PROCEDENCIA.md` | SHA-256 de los archivos ejecutados, adaptaciones de la versión pública y lista de lo excluido. |

## Brazos de la ablación de banda (exploratoria y post hoc)
| Brazo | Modelo y banda | Nota |
|---|---|---|
| A0 | EEGNet en banda ancha (v5) | Referencia; reutiliza las predicciones de la v5, sin reentrenar. |
| A1 | EEGNet en 0.5–3 Hz | Pasa-bajos Butterworth de orden 4 a 3 Hz sobre las épocas v5. |
| A2 | EEGNet en 8–30 Hz | Pasa-banda Butterworth de orden 4. |
| B1 | MDM (OAS, distancia riemanniana) en 0.5–3 Hz | Mismo pasa-bajos que A1. |
| C1 | Tensor-CSPNet en 0.5–3 Hz (4 subbandas) | Se ejecutó aparte, sin control con etiquetas de entrenamiento permutadas. Tardó unos 10 a 20 min por partición en una RTX 4070 SUPER. |

Estadística: exactitud balanceada y kappa por participante, promediadas por partición y sobre las 10 particiones; IC 95 % por bootstrap de participantes; *p* frente al azar por permutación de etiquetas de prueba. Los contrastes A1 − A2 y A1 − A0 se hicieron con *sign-flip* pareado por participante (10 000 réplicas). Holm se aplicó a una sola familia de 36 pruebas (4 brazos × 6 conjuntos frente al azar + 2 contrastes × 6 conjuntos). Las desviaciones declaradas son el relleno automático de scipy (unas 33 muestras) en vez de 256 en A1, A2 y B1, la configuración congelada a posteriori y C1 sin control `permtrain`. Todas aparecen en el artículo y en `PROCEDENCIA.md`.

## Cómo reproducir
1. **Datos públicos.** Descargue de BNCI Horizon 2020 (<https://bnci-horizon-2020.eu/database/data-sets>) los conjuntos **001-2017** (Ofner et al., 2017), **001-2019** (Ofner et al., 2019; el participante P01 desde el archivo de Zenodo) y **001-2020** (Schwarz et al., 2020). El conjunto 001-2017 también está disponible en MOABB (`Ofner2017`). `scripts/download.sh` espera un `download_list.txt` con pares «URL nombre» por línea, que debe armar desde esas páginas. Los datos no se redistribuyen aquí.
2. **Entorno.** Python 3.12 y `pip install -r v5/requirements_v5_pip_freeze.txt`. Para C1, además, `pip install spd_learn==0.2.1` (la versión instalada en el entorno de la corrida; no figura en el congelado porque se instaló después). Las redes se entrenaron con CUDA, una sola corrida de GPU a la vez.
3. **Épocas.** `python v5/code/extract_epochs.py 001-2017 <raíz_extraída_001-2017> v5/epochs` (y lo mismo con 001-2019 y 001-2020).
4. **Corrida v5.** Desde `v5/code`, ejecute `python evaluate.py cross ../epochs ../results CSP_LDA,FBCSP_SVM,MDM,EEGNet,ShallowConvNet,DeepConvNet`, luego `python evaluate.py score ../results` y después `bash ../scripts/run_permtrain.sh` para el control con etiquetas de entrenamiento permutadas.
5. **Ablación.** Desde `v5/ablacion_banda/code`, ejecute `python ablation_v1_en_ejecucion.py all` (A1, A2, B1, `permtrain` y `score`). Para C1, ejecute `python c1/c1_runner_v2.py` y después `python c1/score_c1.py`.
6. Compare con `v5/tablas/`. El entrenamiento con cuDNN no es reproducible bit a bit, así que se esperan diferencias pequeñas en las redes. Los métodos clásicos y la estadística sobre unas mismas predicciones son deterministas.

`TESIS_V5_ROOT` permite apuntar a otra carpeta raíz (por defecto, `v5/`).

## Resultados principales (exactitud balanceada media de 10 particiones)
Ver `tablas/v5_cross_subject_summary_holm.csv` y `tablas/ablation_summary_36.csv`. Por ejemplo, Tensor-CSPNet (C1) obtuvo 0.2017 en 001-2017 ME, 0.1570 en 001-2017 MI y 0.2262 en 001-2019 (*p* de Holm = 0.034 en los tres). En 001-2020 no superó el azar.

## Licencia
MIT, igual que el resto del repositorio (ver `LICENSE`).
