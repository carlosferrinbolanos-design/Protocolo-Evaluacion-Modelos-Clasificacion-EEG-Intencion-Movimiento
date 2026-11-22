# Procedencia de los archivos de `v5/`

Los archivos se copiaron el 21-11-2026 desde el equipo de cómputo de la corrida (Windows, RTX 4070 SUPER) en modo de solo lectura. Para cada archivo se comprobó que la copia era byte a byte idéntica al original (SHA-256 calculado en el equipo con `Get-FileHash`, finales de línea LF).

| Archivo en este repositorio | SHA-256 del archivo ejecutado | Estado |
|---|---|---|
| `code/evaluate.py` | `5e7cbc8da7523d8e28d40b0babea252f3ee81695a6fa72bd51ed8f87374fb1ef` | idéntico |
| `code/extract_epochs.py` | `f8adfe5819e5727094b6aef2ddbf503093aaea15caaa00f93a10bc5529a47d77` | idéntico |
| `code/models.py` | `8768c4481f17d24a3d11fc9ac1844800da090a4e7ef56ecece921f5b4236f98c` | idéntico |
| `ablacion_banda/code/ablation_v1_en_ejecucion.py` | `780da030da3047fe4b3de28551e73182cf8c2bf53fd32250223c8e5fb9b21136` | adaptado (ver abajo) |
| `ablacion_banda/code/ablation.py` | `5baddfec31377dd7768adfb5c2fb493421dfe68e908c267562e2985fe83102ed` | adaptado (ver abajo) |
| `ablacion_banda/code/c1/c1_runner_v2.py` | `d6c8f9a9b3ef1234c14ebab3ac95cf7f0b8ca1df0c0463aeb5bc3bfacbaf0711` | adaptado (ver abajo) |
| `ablacion_banda/code/c1/smoke_c1_cpu.py` | `bd56d54b3799e6a16b96cc6647ca5a4bdd5d97472f0578db747af8e9b9a37ad9` | adaptado (ver abajo) |
| `ablacion_banda/v5-ablacion.json` | `7d12bc7718e9b3dabc52c03293b95da0953ac0ee3c4ed13136b5ce49fd171a5f` | adaptado (ver abajo) |
| `requirements_v5_pip_freeze.txt` | `67ee9dee9ff3ae98439cd82d24f9c606cb1dad2d9806f53eccd47d3be3e84b97` | idéntico |
| `scripts/download.sh` | `7152a2d035e4874956f24ed4b5c81c43a9841fdfb027603cc20c90e639628f61` | adaptado (ver abajo) |
| `scripts/unzip_all.sh` | `d23af8a1ee01fc23aad3d09aa47f26cd8479ba00c65b4a652c77f7e45bc0acf6` | adaptado (ver abajo) |
| `scripts/run_permtrain.sh` | `257e6a38f9f4e65fc45999f677d730698b3a967333613586bc1cdd9109749929` | adaptado (ver abajo) |
| `scripts/run_permtrain_2017.sh` | `8e9769caf3f4a043b43c64b59d3878b3b80d2f9f29ea1503e78033818b48c72c` | adaptado (ver abajo) |
## Adaptaciones para la versión pública
Las adaptaciones se limitan a rutas, intérpretes e identificadores internos. No cambian la lógica, los hiperparámetros, las semillas ni la estadística.

- **Rutas absolutas → variable de entorno.** `V5 = Path(r"D:\ICTQ\TESIS_pipeline_v5")` pasó a `V5 = Path(os.environ.get("TESIS_V5_ROOT", <carpeta v5/ de este repositorio>))` (con `import os` donde faltaba). Las salidas de C1 se derivan de `V5` (`V5/ablacion_banda/results/c1`).
- **Scripts de shell.** Las rutas de datos pasan a `$TESIS_DATA_ROOT` y el intérprete a `$PYTHON`. `unzip_all.sh` llamaba a un verificador temporal de rutas de zip (`zipcheck.py`), que no se conservó: ahora se indica con `$ZIPCHECK`.
- **Identificadores internos.** Se quitaron de los docstrings y de `v5-ablacion.json` los identificadores de las órdenes internas de ejecución y su verificación.
- **`v5-ablacion-corrida1.json`.** Es la configuración de la corrida 1 de la ablación, escrita a posteriori (el proceso arrancó a las 14:43:43 del 20-11-2026, antes de la configuración formal). Se quitaron el PID, la tarea programada y el identificador de la orden. Se conserva el SHA-256 del script ejecutado.
- **`score_c1.py`.** Es nuevo y reproduce el cálculo en memoria que se hizo el 21-11-2026: `evaluate.score_predictions` con 1000 permutaciones, 2000 réplicas bootstrap y semilla 0, igual que para A1, A2 y B1.

## No incluido
- Datos EEG crudos, épocas (`.npz`), manifiestos por participante, predicciones por época y checkpoints. Todos contienen o derivan de códigos o nombres de archivo de participantes. Los datos son públicos y se descargan de sus fuentes (ver `README.md`).
- `download_list.txt` (lista de URL y nombres de archivo de origen), registros de ejecución, informes internos y scripts operativos de espera o reanudación (`run_c1_after.ps1`).
- `c1_runner.py` (v1), reemplazado por `c1_runner_v2.py`. La v2 usa los mismos checkpoints y solo añade registro de progreso, `faulthandler` y otro orden de conjuntos.
- `ablation_antigravity_20261120.py`, de igual tamaño que `ablation.py`: es una copia de trabajo.
- **Configuración congelada de la v5.** No existe ningún archivo de configuración congelada de la corrida v5 principal (el `v5_configuracion.md` que menciona `v5-ablacion.json` no se encontró). El protocolo de la v5 está fijado en los docstrings y las constantes de `code/*.py`, cuyos SHA-256 aparecen arriba.
