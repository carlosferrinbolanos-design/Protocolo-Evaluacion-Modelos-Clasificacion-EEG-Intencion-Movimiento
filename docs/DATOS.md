# Datos

> **Este repositorio no contiene datos EEG.** Los registros pertenecen a sus autores y se distribuyen con licencia CC BY 4.0 desde sus repositorios oficiales. Para reproducir los resultados hay que descargarlos (el cuaderno lo hace automáticamente). Solo se incluyen los **manifiestos de épocas** (`manifests/`). Son tablas de metadatos (archivo fuente, participante, clase, `onset_sample`) y no contienen ninguna muestra de señal.

## Fuentes
| Conjunto de datos | Contenido | Participantes | Canales / fs | Clases usadas | Fuente oficial | DOI | Licencia |
|---|---|---|---|---|---|---|---|
| BNCI 001-2017 (Ofner et al., 2017) | Movimientos del miembro superior, ejecución (ME) e imaginación (MI), formato GDF | 15 personas sanas (S01_ME excluido) | 61 EEG · 512 Hz | 7 (6 movimientos + reposo) | Zenodo, registro 834976 | [10.5281/zenodo.834976](https://doi.org/10.5281/zenodo.834976) | CC BY 4.0 |
| BNCI 001-2019 (Ofner et al., 2019) | Movimientos intentados en personas con lesión medular cervical, GDF | 10 (P01–P10) | 61 EEG · 256 Hz | 5 | Zenodo, registro 2222268 | [10.5281/zenodo.2222268](https://doi.org/10.5281/zenodo.2222268) | CC BY 4.0 |
| BNCI 001-2020 (Schwarz et al., 2020) | Alcance y agarre natural con electrodos de gel, agua y secos, MAT | 15 por sistema (45) | 11 canales sensoriomotores comunes · 256 Hz | 2 | BNCI Horizon 2020 (`https://lampx.tugraz.at/~bci/database/001-2020/`) | — (ver página del conjunto en bnci-horizon-2020.eu) | CC BY 4.0 |

## Descarga
- **001-2017 y 001-2019:** desde la API de Zenodo, en paralelo, verificando el tamaño de cada archivo (cuaderno A-1; en el estudio se usaron `scripts/colab/dl.sh` y `scripts/colab/dl19.sh`). Solo se descargan los archivos que figuran en los manifiestos (unos 24 GB y 4 GB).
- **001-2020:** **el 16-11-2026 el servidor `lampx.tugraz.at` no respondía** (fallo de TLS, HTTP 000), y las copias del Internet Archive estaban truncadas. Por eso las particiones repetidas no se ejecutaron en 001-2020, y en el artículo esos conjuntos de evaluación solo tienen la partición única v3 del estudio. El cuaderno comprueba la disponibilidad del servidor y, si responde, incluye los conjuntos de evaluación 2020 (`INCLUIR_2020_SI_DISPONIBLE = True`).

## Reconstrucción de épocas desde los manifiestos
| Conjunto de evaluación | Manifiesto | Filas | Épocas usadas |
|---|---|---|---|
| 001-2017_ME | `manifests/001-2017_cue_aligned_raw/dataset1_me_cue_epoch_manifest_reextracted_v2.csv` | 5 880 | 5 880 |
| 001-2017_MI | `manifests/001-2017_cue_aligned_raw/dataset1_mi_cue_epoch_manifest_reextracted_v2.csv` | 6 300 | 6 300 |
| 001-2019_offline | `manifests/dataset2_offline_epoch_manifest_model_input_v2.csv` | 3 600 | 3 598 (`finite_for_model`) |
| 001-2020_gel / water / dry | `manifests/dataset3_{gel,water,dry}_epoch_manifest_v2.csv` | 2 325 / 2 436 / 2 433 | — (no ejecutado) |

Reglas:
1. Las épocas se ordenan por (`epoch_file`, `epoch_index`), igual que en el estudio, para conservar el orden.
2. Cada época es `señal[:, onset − pre : onset − pre + n]`, con pre = 0,5 s y n = 2 s + 1 muestra.
3. 001-2017 se remuestrea con `resample_poly(1, 2)`.
4. No hay filtrado, corrección de línea base ni rechazo.
5. Cada `onset_sample` se contrasta con los eventos del archivo crudo y se informa la fracción de coincidencia.

En los manifiestos se eliminó solo el prefijo de la ruta de Google Drive personal (`/content/drive/MyDrive/`) de las columnas `table_file` y `epoch_file`. Al ser un prefijo común, el orden lexicográfico no cambia (comprobado).

## Comprobación de la extracción (estudio, 16-11-2026)
- 001-2019: 3 598 × 61 × 513, todo finito, 10 participantes.
- 001-2017: ME 5 880 × 61 × 513 y MI 6 300 × 61 × 513.
- Con k = 0, los modelos clásicos reproducen exactamente los resultados v3 del estudio, lo que confirma que las épocas son idénticas.
