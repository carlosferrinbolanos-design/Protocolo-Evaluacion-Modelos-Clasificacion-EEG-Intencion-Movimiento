"""Genera TESIS_pipeline_reproducible.ipynb (sin ejecutar) a partir de celdas definidas aquí.
Los valores de referencia de la verificación se leen de tables/summary_mean_sd.csv y tables/stats.json."""
import json, nbformat as nbf, pandas as pd
from pathlib import Path
R = Path(__file__).resolve().parents[1]
A = pd.read_csv(R/'tables/summary_mean_sd.csv'); A = A[A.control == 'true_labels']
REF = {f'{r.stratum}|{r.model}': {'ba_mean': round(r.balanced_accuracy_mean, 4), 'ba_sd': round(r.balanced_accuracy_std, 4),
                                  'mcc_mean': round(r.mcc_mean, 4), 'mcc_sd': round(r.mcc_std, 4)} for r in A.itertuples()}
ST = json.load(open(R/'tables/stats.json'))['stats']
REF_FRIED = {m: {b: {'chi2': round(v['chi2'], 3), 'p_asymptotic': round(v['p_asymptotic'], 4), 'N': v['N']}
                 for b, v in ST[m].items() if b != 'pairwise'} for m in ST}
DAVID_V3 = {'001-2017_ME|CSP_LDA': .1468, '001-2017_ME|FBCSP_SVM': .1524, '001-2017_ME|Riemannian_MDM': .1349,
            '001-2017_MI|CSP_LDA': .1413, '001-2017_MI|FBCSP_SVM': .1429, '001-2017_MI|Riemannian_MDM': .1421,
            '001-2019_offline|CSP_LDA': .2167, '001-2019_offline|FBCSP_SVM': .2083, '001-2019_offline|Riemannian_MDM': .2236}
cells = []
from terminologia import texto, codigo
def md(s): cells.append(nbf.v4.new_markdown_cell(texto(s.strip('\n'))))
def code(s): cells.append(nbf.v4.new_code_cell(codigo(s.strip('\n'))))

md(r'''
# Protocolo de evaluación comparativa con diagrama de diferencia crítica para decodificar la intención de movimiento de mano desde EEG orientado a órtesis en lesión medular

**Cuaderno de reproducibilidad del artículo**

| | |
|---|---|
| **Autor** | David Aogusto Rodriguez Chamorro |
| **Director** | Carlos Diego Ferrin Bolaños |
| **Institución** | Universidad Santiago de Cali — Maestría en Informática |
| **Versión** | v1.0-articulo (16 de noviembre de 2026) |
| **Licencia del código** | MIT |

## Propósito
Este cuaderno reproduce de extremo a extremo los resultados del artículo, siguiendo las cuatro etapas de su metodología (Figura 2): **Análisis → Diseño → Implementación → Pruebas**. Hace lo siguiente:
1. Descarga los datos públicos (BNCI Horizon 2020 / Zenodo).
2. Reconstruye las épocas a partir de los manifiestos congelados del estudio.
3. Genera las **10 particiones 80/20 disjuntas por participante** (semillas **20261113 + k**, k = 0…9).
4. Entrena los seis modelos (**CSP+LDA, FBCSP+SVM, MDM, EEGNet, ShallowConvNet, DeepConvNet**).
5. Ejecuta el **control por permutación de etiquetas** para descartar **fuga de información**.
6. Calcula **exactitud balanceada, F1 macro, kappa y MCC**.
7. Aplica la **prueba global** de Friedman (con Iman-Davenport y p por permutación), el **diagrama de diferencia crítica (DC)** de Nemenyi, Wilcoxon-Holm por pares, la prueba bayesiana con ROPE y las comparaciones frente al azar.
8. Produce todas las tablas y figuras (solo en español).

> **Este cuaderno se entrega sin ejecutar.** Los resultados de referencia se generaron con los scripts equivalentes del repositorio (`scripts/`). La última celda verifica que los números regenerados coinciden con los publicados.

## Cómo ejecutarlo
1. Abra el cuaderno en Google Colab y elija **Entorno de ejecución → Cambiar tipo → GPU T4**.
2. En la celda **0-2 (Configuración)**, elija `MODO = "RAPIDO"` (1 partición, k = 0) o `MODO = "COMPLETO"` (10 particiones y control por permutación en k = 0, 1).
3. La celda 0-1 clona el repositorio; `RUTA_MANIFIESTOS` y `RUTA_PREDICCIONES_GUARDADAS` apuntan por defecto a sus carpetas `manifests/` y `preds/`.
4. Ejecute todo (**Entorno de ejecución → Ejecutar todo**).

| Modo | Qué ejecuta | Tiempo estimado (T4) | Unidades de cómputo Colab (aprox.) |
|---|---|---|---|
| `RAPIDO` | 3 estratos × 6 modelos × 1 partición (k = 0) | 1,5–2 h (dominado por la descarga de unos 28 GB y FBCSP+SVM en 001-2017) | 3–4 |
| `COMPLETO` | 3 estratos × 6 modelos × 10 particiones + permutación (k = 0, 1) | 8–10 h en una sola VM (en el estudio se repartió en 4 VMs: unas 1,5 h y 2,9 UC) | 12–18 |
| `USAR_PREDICCIONES_GUARDADAS = True` | Omite descarga y entrenamiento; recalcula métricas, estadística, tablas y figuras desde `preds/` | < 5 min | < 0,2 |

## Versiones fijadas (entorno con el que se generaron los resultados)
| Paquete | Versión | Paquete | Versión |
|---|---|---|---|
| Python | 3.13 | torch | 2.14.1 (CUDA 13.0, Colab) |
| numpy | 2.4.6 | mne | 1.13.2 |
| scipy | 1.16.3 | pyriemann | 0.12 |
| scikit-learn | 1.6.1 | baycomp | 1.0.3 |
| pandas | 2.2.3 | matplotlib | 3.10.0 |

Los modelos clásicos de 001-2019 y el análisis estadístico se ejecutaron además con scikit-learn 1.9.1, numpy 2.5.5 y pandas 3.1.1. Los resultados clásicos con k = 0 coinciden exactamente con la versión v3 del estudio en ambos entornos.

## Semillas y determinismo
- **Particiones:** `numpy.random.default_rng(20261113 + k + Σ ord(estrato))`, con ⌈20 %⌉ de participantes en prueba. **k = 0 reproduce la partición congelada del estudio (Tabla 1).**
- **Permutación:** clásicos `default_rng(20261113 + k)`; profundos `default_rng(20261113 + k + Σ ord(modelo + estrato))`. La permutación se hace dentro de cada participante de entrenamiento.
- **Redes:** `torch.manual_seed(semilla + Σ ord(modelo + estrato + control))`, con generador propio del `DataLoader`. `cudnn.benchmark = True`, como en el estudio, por lo que **en GPU no hay determinismo bit a bit**. Entre ejecuciones se observaron diferencias de hasta **0,0125** en exactitud balanceada para los modelos profundos (k = 0 frente a v3). Los modelos clásicos son deterministas.

## Fuentes de datos (no se redistribuyen)
| Conjunto | Descripción | Fuente | DOI / URL | Licencia |
|---|---|---|---|---|
| BNCI 001-2017 | Movimientos del miembro superior, ejecución (ME) e imaginación (MI), 15 participantes, 61 EEG, 512 Hz | Ofner et al. (2017) | Zenodo 834976 — https://doi.org/10.5281/zenodo.834976 | CC BY 4.0 |
| BNCI 001-2019 | Movimientos intentados en personas con lesión medular, 10 participantes, 61 EEG, 256 Hz | Ofner et al. (2019) | Zenodo 2222268 — https://doi.org/10.5281/zenodo.2222268 | CC BY 4.0 |
| BNCI 001-2020 | Alcance y agarre con electrodos de gel, agua y secos, 45 participantes | Schwarz et al. (2020) | https://bnci-horizon-2020.eu/database/data-sets (servidor lampx.tugraz.at) | CC BY 4.0 |

## Estructura de carpetas (dentro de `RUTA_TRABAJO`)
```
datos_crudos/2017, datos_crudos/2019, datos_crudos/2020   # descargas (no se versionan)
epocas/<conjunto>.npz, epocas/<conjunto>__b<lo>_<hi>.npy      # épocas y bandas filtradas
predicciones/<conjunto>__<modelo>__<control>__s<k>.csv.gz      # predicción por época
tablas/, figuras/                                            # salidas finales
```

## Correspondencia artículo ↔ cuaderno
| Elemento del artículo | Celda del cuaderno | Script equivalente | Archivo de salida |
|---|---|---|---|
| Figura 1 (flujo de una BCI motora) | — (alcance: evaluación offline, pasos 3 a 5; paso 6 y lazo cerrado fuera de alcance) | `make_fig_pipeline_bci.py` | `figuras/Figura_pipeline_BCI.png` (en el repositorio: `figures/`) |
| Figura 2 (etapas del protocolo) | estructura A → B → C → D | — | — |
| Figura 3 (flujo PRISMA) | no aplica (revisión de literatura) | — | — |
| Tabla 1 (estratos y partición disjunta por participante, k = 0) | A-2, B-1 | `01_extract_epochs.py`, `core.py` | `tablas/particiones.csv` |
| Tabla 2 (requisitos del diagrama DC) | P-3 (N, k, bloques) | `04_analyze.py` | `tablas/stats.json` |
| Tabla 3 (exactitud balanceada real y permutada, partición única) | P-1, P-2 (filtrar k = 0), P-6 | `02_run_classical.py`, `03_run_deep.py`, `05_tables.py` | `tablas/metrics_per_split.csv`, `tablas/permutation_control.csv` |
| Figura 4 (DC con partición única v2) | P-3 y P-5 restringidos a k = 0 | `04_analyze.py` | — (figura de la versión v2 del estudio) |
| Tabla 4 (BA y MCC en 10 particiones, rango promedio N = 3, p de Holm frente al azar) | P-2, P-3, P-4, P-6 | `04_analyze.py`, `05_tables.py` | `tablas/TABLAS_ES_listas_para_pegar.md`, `tablas/TABLAS_ES_estadistica.md`, `tablas/vs_chance.csv` |
| Figura 5 (exactitud balanceada en las 10 particiones) | P-5 | `06_fig_cajas.py` | `figuras/Figura_cajas_exactitud_balanceada.png` |
| Material suplementario: DC con 10 particiones, Wilcoxon-Holm, ROPE, MCC | P-3, P-4, P-5 | `04_analyze.py`, `04b_pairwise_LME.py` | `figuras/DC_*`, `tablas/pairwise_*.csv` |
| Verificación de números | P-7 | `verificar_resultados.py` | — |

## Limitaciones y desviaciones (resumen; detalle en la sección final)
- **001-2020 no se pudo repetir:** el servidor oficial no respondía el 16-11-2026. Sus estratos solo tienen la partición única v3.
- **001-2019:** las semillas 0/6 y 2/8 producen el mismo conjunto de prueba (8 particiones distintas de 10).
- **Los bloques partición × estrato y participante × estrato no son independientes.** Los bloques = estratos son solo 3.
''')

md(r'''
---
# 0-1. Instalación del entorno
**Qué:** clona el repositorio de verificación (manifiestos, predicciones y tablas publicadas; sin datos EEG) e instala las versiones fijadas. **Por qué:** los resultados dependen de MNE (CSP), pyRiemann (MDM) y PyTorch. En Colab, torch ya viene instalado (2.14.x); solo se fijan los demás paquetes.
''')
code(r'''
# Repositorio de verificación (manifiestos de épocas, predicciones publicadas, tablas de referencia). No contiene datos EEG.
![ -d /content/Protocolo-Evaluacion-Modelos-Clasificacion-EEG-Intencion-Movimiento ] || git clone --depth 1 https://github.com/carlosferrinbolanos-design/Protocolo-Evaluacion-Modelos-Clasificacion-EEG-Intencion-Movimiento.git /content/Protocolo-Evaluacion-Modelos-Clasificacion-EEG-Intencion-Movimiento
# Versiones fijadas (ver tabla de la portada). En Colab, ejecutar una vez y reiniciar si cambia numpy.
%pip -q install "mne==1.13.2" "pyriemann==0.12" "scikit-learn==1.6.1" "scipy==1.16.3" "pandas==2.2.3" "matplotlib==3.10.0" "baycomp==1.0.3"
''')

md(r'''
# 0-2. Configuración
**Qué:** todos los parámetros del estudio en un solo objeto. **Por qué:** ninguna ruta ni hiperparámetro queda escrito dentro de las funciones.

| Parámetro | Valor | Origen |
|---|---|---|
| `SEMILLA_BASE` | 20261113 | configuración congelada del estudio |
| `N_PARTICIONES` | 10 en COMPLETO, 1 en RAPIDO | particiones repetidas |
| `SEMILLAS_PERMUTACION` | k = 0, 1 | control de fuga de información |
| `PROPORCION_PRUEBA` | 0,20 de participantes (⌈·⌉) | partición 80/20 disjunta por participante |
| `VENTANA_S` | −0,5 a 1,5 s | Diseño (artículo) |
''')
code(r'''
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class Configuracion:
    """Parámetros congelados del protocolo y rutas de trabajo."""
    MODO: str = "RAPIDO"                      # "RAPIDO" | "COMPLETO"
    USAR_PREDICCIONES_GUARDADAS: bool = False # True: recalcula todo desde RUTA_PREDICCIONES_GUARDADAS sin entrenar
    RUTA_TRABAJO: Path = Path("/content/tesis_eeg")
    RUTA_MANIFIESTOS: Path = Path("/content/Protocolo-Evaluacion-Modelos-Clasificacion-EEG-Intencion-Movimiento/manifests")
    RUTA_PREDICCIONES_GUARDADAS: Path = Path("/content/Protocolo-Evaluacion-Modelos-Clasificacion-EEG-Intencion-Movimiento/preds")
    SEMILLA_BASE: int = 20261113
    PROPORCION_PRUEBA: float = 0.20
    SEMILLAS_PERMUTACION: tuple = (0, 1)
    ESTRATOS: tuple = ("001-2017_ME", "001-2017_MI", "001-2019_offline")   # 001-2020: ver A-1 (servidor)
    ESTRATOS_2020: tuple = ("001-2020_gel", "001-2020_water", "001-2020_dry")
    INCLUIR_2020_SI_DISPONIBLE: bool = False
    MODELOS: tuple = ("CSP_LDA", "FBCSP_SVM", "Riemannian_MDM", "EEGNet", "ShallowConvNet", "DeepConvNet")
    FS_HZ: int = 256
    BANDA_MOTORA: tuple = (8, 30)
    BANDAS_FBCSP: tuple = ((8, 12), (12, 16), (16, 20), (20, 24), (24, 28), (28, 32))
    CANALES_2020: tuple = ("FC3", "FCZ", "FC4", "C3", "C1", "CZ", "C2", "C4", "CP3", "CPZ", "CP4")
    EPOCAS_RED: int = 20
    LR: float = 1e-3
    WEIGHT_DECAY: float = 1e-4
    LOTE_ENTRENAMIENTO: int = 64
    LOTE_EVALUACION: int = 128
    ROPE_BA: float = 0.01
    ROPE_MCC: float = 0.02
    N_PERM_FRIEDMAN: int = 5000
    N_PERM_FRIEDMAN_ESTRATOS: int = 20000

    @property
    def particiones(self) -> range:
        """Índices k de las particiones a ejecutar según el modo."""
        return range(10) if self.MODO == "COMPLETO" else range(1)

    @property
    def semillas_permutacion(self) -> tuple:
        return self.SEMILLAS_PERMUTACION if self.MODO == "COMPLETO" else (0,)

    def ruta(self, nombre: str) -> Path:
        p = self.RUTA_TRABAJO / nombre
        p.mkdir(parents=True, exist_ok=True)
        return p

CFG = Configuracion()
NOMBRE_MODELO = {"CSP_LDA": "CSP+LDA", "FBCSP_SVM": "FBCSP+SVM", "Riemannian_MDM": "MDM", "EEGNet": "EEGNet",
                 "ShallowConvNet": "ShallowConvNet", "DeepConvNet": "DeepConvNet"}
AZAR = {"001-2017_ME": 1/7, "001-2017_MI": 1/7, "001-2019_offline": 1/5, "001-2020_gel": .5, "001-2020_water": .5, "001-2020_dry": .5}
CONTROL_REAL, CONTROL_PERM = "true_labels", "labels_permuted_within_train_participant"
print(CFG.MODO, list(CFG.particiones), CFG.semillas_permutacion)
''')

md(r'''
# 0-3. Importaciones y semillas globales
''')
code(r'''
import gc, itertools, json, os, random, re, subprocess, time, urllib.request, zipfile, warnings
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy import stats
from scipy.io import loadmat
from scipy.signal import butter, sosfiltfilt, resample_poly
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.metrics import balanced_accuracy_score, f1_score, cohen_kappa_score, matthews_corrcoef, confusion_matrix
import mne
from mne.decoding import CSP
from pyriemann.estimation import Covariances
from pyriemann.classification import MDM
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import baycomp
mne.set_log_level("ERROR"); warnings.filterwarnings("ignore")
plt.rcParams["font.family"] = "DejaVu Sans"
DISPOSITIVO = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True   # igual que en el estudio (ver nota de determinismo)
print("Dispositivo:", DISPOSITIVO, "| torch", torch.__version__, "| mne", mne.__version__)
''')

md(r'''
---
# A. Etapa de Análisis
*Artículo: Metodología → Análisis; Desarrollo → Cómo se realizó el Análisis.*

## A-1. Descarga de datos públicos
**Qué:** descarga los registros crudos desde sus repositorios oficiales y verifica su tamaño con la API de Zenodo.
**Por qué:** los datos no se redistribuyen (licencia CC BY 4.0; se cita la fuente).
**Parámetros:** solo se descargan los archivos que aparecen en los manifiestos del estudio. Por eso S01_ME, excluido en el Análisis, no se descarga.

Para **001-2020**, se comprueba primero que el servidor `lampx.tugraz.at` responde. El 16-11-2026 no respondía; en ese caso los estratos de 2020 se omiten con un aviso explícito.
''')
code(r'''
MANIFIESTO = {
    "001-2017_ME": ("001-2017_cue_aligned_raw/dataset1_me_cue_epoch_manifest_reextracted_v2.csv", "subject"),
    "001-2017_MI": ("001-2017_cue_aligned_raw/dataset1_mi_cue_epoch_manifest_reextracted_v2.csv", "subject"),
    "001-2019_offline": ("dataset2_offline_epoch_manifest_model_input_v2.csv", "subject"),
    "001-2020_gel": ("dataset3_gel_epoch_manifest_v2.csv", "participant"),
    "001-2020_water": ("dataset3_water_epoch_manifest_v2.csv", "participant"),
    "001-2020_dry": ("dataset3_dry_epoch_manifest_v2.csv", "participant"),
}
URL_2020 = "https://lampx.tugraz.at/~bci/database/001-2020/"

def leer_manifiesto(estrato: str) -> pd.DataFrame:
    """Lee el manifiesto congelado de un estrato con el mismo filtro y orden del estudio (finite_for_model; orden epoch_file, epoch_index)."""
    archivo, _ = MANIFIESTO[estrato]
    d = pd.read_csv(CFG.RUTA_MANIFIESTOS / archivo)
    if "finite_for_model" in d.columns:
        d = d[d.finite_for_model.astype(bool)].copy()
    return d.sort_values(["epoch_file", "epoch_index"]).reset_index(drop=True)

def descargar(url: str, destino: Path, tamano: int | None = None, intentos: int = 8) -> bool:
    """Descarga con reintentos y verificación de tamaño (curl)."""
    for _ in range(intentos):
        if destino.exists() and (tamano is None or destino.stat().st_size == tamano):
            return True
        parcial = destino.with_suffix(destino.suffix + ".part")
        subprocess.run(["curl", "-sL", "--retry", "5", "-o", str(parcial), url])
        if parcial.exists() and (tamano is None or parcial.stat().st_size == tamano):
            parcial.replace(destino); return True
    return False

def descargar_estratos(estratos: tuple) -> dict:
    """Descarga en paralelo (32 hilos) los archivos de Zenodo que requieren los estratos indicados."""
    import concurrent.futures as cf
    tareas = []
    if any(e.startswith("001-2017") for e in estratos):
        meta = {f["key"]: f["size"] for f in json.load(urllib.request.urlopen("https://zenodo.org/api/records/834976"))["files"]}
        nec = set()
        for e in [e for e in estratos if e.startswith("001-2017")]:
            nec |= {os.path.basename(x) for x in leer_manifiesto(e).source_gdf}
        tareas += [(f"https://zenodo.org/records/834976/files/{k}?download=1", CFG.ruta("datos_crudos/2017") / k, meta[k]) for k in sorted(nec)]
    if "001-2019_offline" in estratos:
        meta = {f["key"]: f["size"] for f in json.load(urllib.request.urlopen("https://zenodo.org/api/records/2222268"))["files"]}
        tareas += [(f"https://zenodo.org/records/2222268/files/P{i:02d}.zip?download=1", CFG.ruta("datos_crudos/2019") / f"P{i:02d}.zip", meta[f"P{i:02d}.zip"]) for i in range(1, 11)]
    with cf.ThreadPoolExecutor(32) as ex:
        ok = list(ex.map(lambda t: descargar(*t), tareas))
    for z in CFG.ruta("datos_crudos/2019").glob("*.zip"):
        zipfile.ZipFile(z).extractall(z.with_suffix(""))
    return {"archivos": len(tareas), "fallidos": int(len(ok) - sum(ok))}

def servidor_2020_disponible() -> bool:
    """True si el servidor oficial de BNCI 001-2020 responde a una petición parcial."""
    r = subprocess.run(["curl", "-s", "-o", "/dev/null", "-w", "%{http_code}", "--max-time", "25", "-r", "0-100", URL_2020 + "G01.mat"], capture_output=True, text=True)
    return r.stdout in ("200", "206")

ESTRATOS = list(CFG.ESTRATOS)
if CFG.INCLUIR_2020_SI_DISPONIBLE:
    if servidor_2020_disponible():
        ESTRATOS += list(CFG.ESTRATOS_2020)
    else:
        print("AVISO: lampx.tugraz.at no responde; los estratos 001-2020 se omiten (ver Limitaciones).")
if not CFG.USAR_PREDICCIONES_GUARDADAS:
    print(descargar_estratos(tuple(ESTRATOS)))
    for e in [e for e in ESTRATOS if e.startswith("001-2020")]:
        for f in sorted(set(leer_manifiesto(e).source_file)):
            assert descargar(URL_2020 + f, CFG.ruta("datos_crudos/2020") / f), f
''')

md(r'''
## A-2. Reglas de inclusión y extracción de épocas desde los manifiestos
**Qué:** reconstruye exactamente las épocas del estudio (mismo archivo fuente, `onset_sample` y clase), con ventana de **−0,5 s a 1,5 s**.

**Reglas del Análisis (artículo):**
- S01_ME excluido; ME y MI se tratan por separado.
- 001-2019: solo corridas offline de movimientos intentados (3 598 entradas finitas).
- 001-2020: gel, agua y seco como estratos separados, con 4 eventos a menos de 4 s excluidos.
- **Sin rechazo por amplitud ni corrección de línea base, y sin filtrado previo.**

**Detalles de canales y muestreo:**
- 001-2017: primeros 61 canales, 512 Hz → 1 025 muestras → `resample_poly(1, 2)` → 513 muestras a 256 Hz.
- 001-2019: primeros 61 canales, 256 Hz.
- 001-2020: 11 canales sensoriomotores comunes.

Cada onset del manifiesto se contrasta con los eventos del archivo crudo. Se informan dos valores: la fracción de coincidencia exacta tras redondear el tiempo de la anotación a muestras (como en `01_extract_epochs.py`) y la distancia máxima a la anotación más cercana. En el estudio, todos los onsets quedaron **a 1 muestra o menos** de una anotación cruda (comprobado el 16-11-2026). La coincidencia exacta fue menor (media de 0,58 en 001-2017 y 0,99 en 001-2019), probablemente porque los onsets del manifiesto truncan, en lugar de redondear, el tiempo de la anotación. Las épocas se toman siempre en el `onset_sample` del manifiesto, igual que en el estudio.
''')
code(r'''
def canon(x: str) -> str:
    """Nombre de canal canónico (mayúsculas, solo alfanuméricos)."""
    return re.sub("[^A-Z0-9]+", "", str(x).upper())

def _fuente_2019(src: str) -> Path:
    base = Path(src).name
    c = [p for p in CFG.ruta("datos_crudos/2019").rglob(base) if "__MACOSX" not in str(p)]
    assert c, src
    return c[0]

def extraer_epocas(estrato: str) -> dict:
    """Extrae X (n, canales, 513), y, s del estrato según su manifiesto; guarda epocas/<estrato>.npz y devuelve un resumen de control."""
    d = leer_manifiesto(estrato); _, col_sujeto = MANIFIESTO[estrato]
    col_fuente = "source_gdf" if "source_gdf" in d.columns else "source_file"
    X = [None] * len(d); coincidencias = []; dist_max = 0
    for src, b in d.groupby(col_fuente, sort=False):
        if estrato.startswith("001-2017"):
            raw = mne.io.read_raw_gdf(CFG.ruta("datos_crudos/2017") / Path(src).name, preload=True)
            fs = raw.info["sfreq"]; assert fs == 512
            sig = raw.get_data()[:61].astype(np.float32); ev = np.round(raw.annotations.onset * fs).astype(int); pre, n = 256, 1025
        elif estrato.startswith("001-2019"):
            raw = mne.io.read_raw_gdf(_fuente_2019(src), preload=True)
            fs = raw.info["sfreq"]; assert fs == 256
            sig = raw.get_data()[:61].astype(np.float32); ev = np.round(raw.annotations.onset * fs).astype(int); pre, n = 128, 513
        else:
            m = loadmat(CFG.ruta("datos_crudos/2020") / src, struct_as_record=False, squeeze_me=True)
            etiquetas = [canon(x) for x in np.ravel(m["header"].channels_labels)]
            sig = np.asarray(m["signal"])[[etiquetas.index(c) for c in CFG.CANALES_2020]].astype(np.float32)
            ev = np.ravel(m["events"].positions).astype(int) - 1; pre, n = 128, 513
        on = b.onset_sample.to_numpy(int)
        coincidencias.append(float(np.isin(on, ev).mean()))   # fracción de coincidencia exacta (igual que el script)
        dist_max = max(dist_max, int(np.max([np.min(np.abs(ev - o)) for o in on])))
        for i, o in zip(b.index, on):
            e = sig[:, o - pre:o - pre + n]; assert e.shape[1] == n, (src, o)
            X[i] = resample_poly(e, 1, 2, axis=-1).astype(np.float32) if estrato.startswith("001-2017") else e
    X = np.stack(X).astype(np.float32); assert np.isfinite(X).all(), estrato
    y = np.asarray(d.movement_class.astype(str).tolist(), dtype=str); s = np.asarray(d[col_sujeto].astype(str).tolist(), dtype=str)
    np.savez(CFG.ruta("epocas") / f"{estrato}.npz", X=X, y=y, s=s)
    return {"estrato": estrato, "forma": X.shape, "participantes": len(set(s)),
            "coincidencia_onset_min": min(coincidencias), "coincidencia_onset_media": float(np.mean(coincidencias)),
            "distancia_max_muestras": dist_max}

if not CFG.USAR_PREDICCIONES_GUARDADAS:
    for e in ESTRATOS:
        r = extraer_epocas(e); print(r); assert r["distancia_max_muestras"] <= 1, "onset del manifiesto sin evento crudo cercano"
''')

md(r'''
---
# B. Etapa de Diseño
*Artículo: Metodología → Diseño; Tabla 1.*

## B-1. Partición 80/20 disjunta por participante
**Qué:** para cada estrato y cada k se elige ⌈20 %⌉ de los participantes para prueba con `default_rng(20261113 + k + Σ ord(estrato))`. **Ningún participante de prueba se usa en entrenamiento, normalización ni ajuste.**
**Por qué:** así se evalúa la generalización a participantes nuevos y se evita la **fuga de información** entre sujetos. Las mismas particiones se usan para los seis modelos (bloques pareados para Friedman).
''')
code(r'''
def semilla(k: int) -> int:
    """Semilla de la partición k (k = 0 reproduce la partición congelada del estudio)."""
    return CFG.SEMILLA_BASE + k

def particion(estrato: str, s: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray, list]:
    """Máscaras de entrenamiento/prueba disjuntas por participante y lista de participantes de prueba."""
    u = np.array(sorted(set(s)))
    rng = np.random.default_rng(semilla(k) + sum(map(ord, estrato)))
    prueba = set(rng.choice(u, size=max(1, int(np.ceil(CFG.PROPORCION_PRUEBA * len(u)))), replace=False))
    te = np.array([q in prueba for q in s]); assert not set(s[~te]) & set(s[te])
    return ~te, te, sorted(prueba)

def permutar_entrenamiento(y: np.ndarray, s: np.ndarray, tr: np.ndarray, sem: int) -> np.ndarray:
    """Permuta las etiquetas de entrenamiento dentro de cada participante; las de prueba quedan intactas (control de fuga)."""
    z = y.copy(); rng = np.random.default_rng(sem)
    for q in sorted(set(s[tr])):
        ii = np.where((s == q) & tr)[0]; z[ii] = rng.permutation(z[ii])
    return z

def cargar_epocas(estrato: str):
    z = np.load(CFG.ruta("epocas") / f"{estrato}.npz", allow_pickle=True)
    return z["X"], z["y"], z["s"]

if not CFG.USAR_PREDICCIONES_GUARDADAS:
    filas = []
    for e in ESTRATOS:
        _, _, s = cargar_epocas(e)
        for k in range(10):
            filas.append({"estrato": e, "k": k, "semilla": semilla(k), "participantes_prueba": " ".join(particion(e, s, k)[2])})
    PARTICIONES = pd.DataFrame(filas); PARTICIONES.to_csv(CFG.ruta("tablas") / "particiones.csv", index=False)
    display(PARTICIONES[PARTICIONES.k == 0])   # Tabla 1: S08 S13 S15 | S08 S10 S14 | P06 P10
''')

md(r'''
---
# C. Etapa de Implementación
*Artículo: Metodología → Implementación; Desarrollo → Cómo se realizó la Implementación.*

## C-1. Pipelines clásicos: CSP+LDA, FBCSP+SVM y MDM
- **CSP+LDA:** pasa-banda Butterworth de orden 4 (8–30 Hz, fase cero), CSP con 12 componentes (Ledoit-Wolf, log-varianza) y LDA lsqr con contracción automática.
- **FBCSP+SVM:** seis bandas de 4 Hz entre 8 y 32 Hz, CSP con 4 componentes por banda, estandarización y SVM RBF (C = 1, gamma = scale, `class_weight='balanced'`).
- **MDM:** 8–30 Hz, covarianza OAS, media y distancia riemannianas.

El filtrado es por época y no depende de las etiquetas, así que se calcula una vez por estrato y se guarda en disco (`.npy`, lectura por *memory map*). Esto equivale al código del estudio, que filtraba todas las épocas antes de dividir.
''')
code(r'''
def preparar_bandas(estrato: str) -> None:
    """Filtra las épocas en la banda motora y en las seis bandas FBCSP; guarda epocas/<estrato>__b<lo>_<hi>.npy."""
    X, _, _ = cargar_epocas(estrato)
    for lo, hi in (CFG.BANDA_MOTORA,) + CFG.BANDAS_FBCSP:
        f = CFG.ruta("epocas") / f"{estrato}__b{lo}_{hi}.npy"
        if not f.exists():
            sos = butter(4, [lo, hi], btype="bandpass", fs=CFG.FS_HZ, output="sos")
            np.save(f, sosfiltfilt(sos, X, axis=-1).astype(np.float32))

def banda(estrato: str, lo: int, hi: int) -> np.ndarray:
    return np.load(CFG.ruta("epocas") / f"{estrato}__b{lo}_{hi}.npy", mmap_mode="r")

def ajustar_clasico(modelo: str, estrato: str, tr: np.ndarray, te: np.ndarray, y_ent: np.ndarray) -> np.ndarray:
    """Entrena un pipeline clásico con las épocas de entrenamiento y predice las de prueba (códigos de clase)."""
    if modelo == "CSP_LDA":
        XB = banda(estrato, *CFG.BANDA_MOTORA)
        csp = CSP(n_components=12, reg="ledoit_wolf", log=True, norm_trace=False)
        F1 = csp.fit_transform(np.asarray(XB[tr]), y_ent[tr]); F2 = csp.transform(np.asarray(XB[te]))
        return LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(F1, y_ent[tr]).predict(F2)
    if modelo == "Riemannian_MDM":
        XB = banda(estrato, *CFG.BANDA_MOTORA); cov = Covariances(estimator="oas")
        return MDM(metric="riemann").fit(cov.transform(np.asarray(XB[tr])), y_ent[tr]).predict(cov.transform(np.asarray(XB[te])))
    if modelo == "FBCSP_SVM":
        f1, f2 = [], []
        for lo, hi in CFG.BANDAS_FBCSP:
            B = banda(estrato, lo, hi); csp = CSP(n_components=4, reg="ledoit_wolf", log=True, norm_trace=False)
            f1.append(csp.fit_transform(np.asarray(B[tr]), y_ent[tr])); f2.append(csp.transform(np.asarray(B[te])))
        F1, F2 = np.hstack(f1), np.hstack(f2); esc = StandardScaler().fit(F1)
        return SVC(kernel="rbf", C=1.0, gamma="scale", class_weight="balanced").fit(esc.transform(F1), y_ent[tr]).predict(esc.transform(F2))
    raise ValueError(modelo)
''')

md(r'''
## C-2. Redes profundas: EEGNet, ShallowConvNet y DeepConvNet
Las arquitecturas son idénticas a las del estudio (código v3 ejecutado). Configuración de entrenamiento:
- 20 épocas, AdamW (lr = 10⁻³, decaimiento = 10⁻⁴), lote de 64 (128 en evaluación).
- Normalización por canal ajustada **solo con entrenamiento**.

**Configuración final por estrato (configuración congelada del estudio):**
- **001-2017:** pérdida **no ponderada**, sin programador de tasa.
- **001-2019 y 001-2020:** pérdida **ponderada por clase** con pesos n / (K · nᶜ) calculados sobre las etiquetas de entrenamiento, y **programación coseno** `CosineAnnealingLR(T_max = 20)` aplicada por época. Es el mismo `train_net` del cuaderno fuente del estudio.

> La ejecución v3 original de 001-2019 se hizo sin ponderación ni programador. Los resultados finales del artículo para 001-2019 corresponden a la reejecución con la configuración congelada (ver CHANGELOG).
''')
code(r'''
class EEGNet(nn.Module):
    """EEGNet compacta (8 filtros temporales de 64 muestras, D = 2, separable 16)."""
    def __init__(self, c: int, k: int):
        super().__init__()
        self.t = nn.Conv2d(1, 8, (1, 64), padding=(0, 32), bias=False); self.b1 = nn.BatchNorm2d(8)
        self.s = nn.Conv2d(8, 16, (c, 1), groups=8, bias=False); self.b2 = nn.BatchNorm2d(16)
        self.d = nn.Conv2d(16, 16, (1, 16), groups=16, padding=(0, 8), bias=False); self.p = nn.Conv2d(16, 16, 1, bias=False)
        self.b3 = nn.BatchNorm2d(16); self.o = nn.Dropout(.25); self.f = nn.Linear(16, k)
    def forward(self, x):
        x = F.elu(self.b1(self.t(x))); x = F.elu(self.b2(self.s(x))); x = self.o(F.avg_pool2d(x, (1, 4)))
        x = F.elu(self.b3(self.p(self.d(x)))); x = self.o(F.avg_pool2d(x, (1, 8)))
        return self.f(F.adaptive_avg_pool2d(x, 1).flatten(1))

class ShallowConvNet(nn.Module):
    """ShallowConvNet: convolución temporal (25) + espacial, cuadrado, promedio (75/15), logaritmo."""
    def __init__(self, c: int, k: int):
        super().__init__()
        self.t = nn.Conv2d(1, 40, (1, 25), padding=(0, 12), bias=False); self.s = nn.Conv2d(40, 40, (c, 1), bias=False)
        self.b = nn.BatchNorm2d(40); self.o = nn.Dropout(.5); self.f = nn.Linear(40, k)
    def forward(self, x):
        x = self.b(self.s(self.t(x))); x = x * x; x = F.avg_pool2d(x, (1, 75), stride=(1, 15))
        x = torch.log(torch.clamp(x, min=1e-6)); x = self.o(x)
        return self.f(F.adaptive_avg_pool2d(x, 1).flatten(1))

class DeepConvNet(nn.Module):
    """DeepConvNet: cuatro bloques conv-BN-ELU-maxpool (25, 50, 100, 200 filtros)."""
    def __init__(self, c: int, k: int):
        super().__init__()
        self.a = nn.Sequential(nn.Conv2d(1, 25, (1, 10), padding=(0, 5), bias=False), nn.Conv2d(25, 25, (c, 1), bias=False),
                               nn.BatchNorm2d(25), nn.ELU(), nn.MaxPool2d((1, 3), stride=(1, 3)), nn.Dropout(.5))
        def bloque(i, o):
            return nn.Sequential(nn.Conv2d(i, o, (1, 10), padding=(0, 5), bias=False), nn.BatchNorm2d(o), nn.ELU(),
                                 nn.MaxPool2d((1, 3), stride=(1, 3)), nn.Dropout(.5))
        self.b = bloque(25, 50); self.c = bloque(50, 100); self.d = bloque(100, 200); self.f = nn.Linear(200, k)
    def forward(self, x):
        return self.f(F.adaptive_avg_pool2d(self.d(self.c(self.b(self.a(x)))), 1).flatten(1))

REDES = {"EEGNet": EEGNet, "ShallowConvNet": ShallowConvNet, "DeepConvNet": DeepConvNet}

def ponderacion_y_coseno(estrato: str) -> bool:
    """Configuración congelada: pérdida ponderada + coseno en 001-2019 y 001-2020; no en 001-2017."""
    return not estrato.startswith("001-2017")

def ajustar_red(modelo: str, estrato: str, control: str, X: np.ndarray, yy: np.ndarray, y_ent: np.ndarray,
                tr: np.ndarray, te: np.ndarray, sem: int) -> np.ndarray:
    """Entrena una red con la configuración del estrato y devuelve las predicciones de prueba."""
    base = sem + sum(map(ord, modelo + estrato + control))
    mu = X[tr].mean(axis=(0, 2), dtype=np.float64).astype(np.float32)
    sd = np.maximum(X[tr].std(axis=(0, 2), dtype=np.float64).astype(np.float32), 1e-6)
    xn = ((X - mu[None, :, None]) / sd[None, :, None]).astype(np.float32)
    g = torch.Generator(); g.manual_seed(base)
    dl = DataLoader(TensorDataset(torch.from_numpy(xn[tr]).unsqueeze(1), torch.from_numpy(y_ent[tr]).long()),
                    batch_size=CFG.LOTE_ENTRENAMIENTO, shuffle=True, generator=g, num_workers=0)
    tl = DataLoader(TensorDataset(torch.from_numpy(xn[te]).unsqueeze(1), torch.from_numpy(yy[te]).long()),
                    batch_size=CFG.LOTE_EVALUACION, shuffle=False, num_workers=0)
    torch.manual_seed(base); np.random.seed(base % 2**32); random.seed(base)
    K = int(yy.max()) + 1
    red = REDES[modelo](xn.shape[1], K).to(DISPOSITIVO)
    opt = torch.optim.AdamW(red.parameters(), lr=CFG.LR, weight_decay=CFG.WEIGHT_DECAY)
    if ponderacion_y_coseno(estrato):
        conteo = np.bincount(y_ent[tr], minlength=K)
        criterio = nn.CrossEntropyLoss(weight=torch.tensor((tr.sum() / (K * conteo)).astype(np.float32), device=DISPOSITIVO))
        prog = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=CFG.EPOCAS_RED)
    else:
        criterio, prog = nn.CrossEntropyLoss(), None
    for _ in range(CFG.EPOCAS_RED):
        red.train()
        for xb, yb in dl:
            xb, yb = xb.to(DISPOSITIVO, non_blocking=True), yb.to(DISPOSITIVO, non_blocking=True)
            opt.zero_grad(set_to_none=True); criterio(red(xb), yb).backward(); opt.step()
        if prog is not None:
            prog.step()
    red.eval(); pred = []
    with torch.no_grad():
        for xb, _ in tl:
            pred.extend(red(xb.to(DISPOSITIVO, non_blocking=True)).argmax(1).cpu().numpy())
    del red, opt, dl, tl, xn; torch.cuda.empty_cache(); gc.collect()
    return np.asarray(pred)
''')

md(r'''
---
# D. Etapa de Pruebas
*Artículo: Metodología → Pruebas; Desarrollo → Cómo se realizaron las Pruebas; Resultados en la etapa de Pruebas.*

## P-1. Ejecución con etiquetas reales y control por permutación
**Qué:** para cada estrato, partición k y modelo, entrena, predice y guarda la predicción por época en `predicciones/` (punto de control: lo ya hecho se omite).
**Control de fuga de información:** con k = 0 y k = 1, se reentrena cada modelo con las etiquetas de entrenamiento permutadas dentro de cada participante. El desempeño debe quedar en el nivel de azar.
''')
code(r'''
def guardar_prediccion(estrato, modelo, control, k, s, te, y_true, y_pred, clases, ts) -> None:
    """Guarda la predicción por época (CSV comprimido) y metadatos (JSON)."""
    out = CFG.ruta("predicciones"); nombre = f"{estrato}__{modelo}__{control}__s{k}"
    pd.DataFrame({"stratum": estrato, "model": modelo, "control": control, "seed_idx": k, "seed": semilla(k), "subject": s[te],
                  "y_true": clases[y_true], "y_pred": clases[y_pred]}).to_csv(out / f"{nombre}.csv.gz", index=False, compression="gzip")
    (out / f"{nombre}.json").write_text(json.dumps({"stratum": estrato, "model": modelo, "control": control, "seed_idx": k,
                                                     "seed": semilla(k), "test_subjects": ts, "n_test": int(te.sum())}))

def ejecutar_estrato(estrato: str) -> None:
    """Ejecuta los seis modelos en las particiones del modo elegido, con etiquetas reales y permutadas."""
    X, y, s = cargar_epocas(estrato); preparar_bandas(estrato)
    le = LabelEncoder(); yy = le.fit_transform(y); clases = le.classes_
    trabajos = [(CONTROL_REAL, k) for k in CFG.particiones] + [(CONTROL_PERM, k) for k in CFG.semillas_permutacion]
    for control, k in trabajos:
        tr, te, ts = particion(estrato, s, k)
        for modelo in CFG.MODELOS:
            if (CFG.ruta("predicciones") / f"{estrato}__{modelo}__{control}__s{k}.csv.gz").exists():
                continue
            t0 = time.time()
            if modelo in REDES:
                y_ent = yy if control == CONTROL_REAL else le.transform(permutar_entrenamiento(y, s, tr, semilla(k) + sum(map(ord, modelo + estrato))))
                pred = ajustar_red(modelo, estrato, control, X, yy, y_ent, tr, te, semilla(k))
            else:
                y_ent = yy if control == CONTROL_REAL else le.transform(permutar_entrenamiento(y, s, tr, semilla(k)))
                pred = ajustar_clasico(modelo, estrato, tr, te, y_ent)
            guardar_prediccion(estrato, modelo, control, k, s, te, yy[te], pred, clases, ts)
            print(f"{estrato} {modelo} {control} k={k}: {balanced_accuracy_score(yy[te], pred):.4f} ({time.time()-t0:.0f} s)")

if CFG.USAR_PREDICCIONES_GUARDADAS:
    RUTA_PRED = CFG.RUTA_PREDICCIONES_GUARDADAS
else:
    for e in ESTRATOS:
        ejecutar_estrato(e)
    RUTA_PRED = CFG.ruta("predicciones")
''')

md(r'''
## P-2. Métricas: exactitud balanceada, F1 macro, kappa y MCC
**Qué:** a partir de las predicciones por época se calculan las métricas por partición, por participante y las matrices de confusión.
**Por qué:** el MCC y el kappa corrigen por azar y por desbalance. El azar de la exactitud balanceada es 1/K y el del MCC y el kappa es 0.
''')
code(r'''
def metricas(d: pd.DataFrame) -> dict:
    """Exactitud balanceada, F1 macro, kappa de Cohen y MCC de un conjunto de predicciones."""
    yt, yp = d.y_true, d.y_pred
    return dict(balanced_accuracy=balanced_accuracy_score(yt, yp), macro_f1=f1_score(yt, yp, average="macro", labels=sorted(set(yt)), zero_division=0),
                cohen_kappa=cohen_kappa_score(yt, yp), mcc=matthews_corrcoef(yt, yp), n=len(d))

filas, filas_p, cms = [], [], []
for f in sorted(Path(RUTA_PRED).glob("*.csv.gz")):
    d = pd.read_csv(f); r0 = d.iloc[0]
    clave = dict(stratum=r0.stratum, model=r0.model, control=r0.control, seed_idx=int(r0.seed_idx), seed=int(r0.seed))
    filas.append({**clave, **metricas(d)})
    filas_p += [{**clave, "subject": q, **metricas(g)} for q, g in d.groupby("subject")]
    labs = sorted(set(d.y_true)); cm = confusion_matrix(d.y_true, d.y_pred, labels=labs)
    cms += [{**clave, "true": a, "pred": b, "count": int(cm[i, j])} for i, a in enumerate(labs) for j, b in enumerate(labs)]
S = pd.DataFrame(filas); S["chance"] = S.stratum.map(AZAR); PP = pd.DataFrame(filas_p)
T = CFG.ruta("tablas")
S.to_csv(T / "metrics_per_split.csv", index=False); PP.to_csv(T / "metrics_per_participant_split.csv", index=False)
pd.DataFrame(cms).to_csv(T / "confusion_matrices_per_split.csv", index=False)
METRICAS = ["balanced_accuracy", "macro_f1", "cohen_kappa", "mcc"]
RES = S.groupby(["control", "stratum", "model"])[METRICAS].agg(["mean", "std", "count"]); RES.columns = ["_".join(c) for c in RES.columns]
RES = RES.reset_index(); RES.to_csv(T / "summary_mean_sd.csv", index=False)
REAL = S[S.control == CONTROL_REAL]
display(RES[RES.control == CONTROL_REAL][["stratum", "model", "balanced_accuracy_mean", "balanced_accuracy_std", "mcc_mean", "mcc_std"]].round(4))
''')

md(r'''
## P-3. Prueba global de Friedman, Iman-Davenport y diferencia crítica de Nemenyi
**Qué:**
- Rangos por bloque (1 = mejor; empates promediados).
- Estadístico χ²_F de Friedman con su p asintótica, la corrección de Iman-Davenport y una **p por permutación** (Monte Carlo, permutando rangos dentro de cada bloque), adecuada para N pequeño.
- DC de Nemenyi = q_α √(k(k+1)/(6N)), con q₀.₀₅ = 2,850 para k = 6 (Demšar, 2006).

**Bloques analizados:**
1. **Estratos** (media de las particiones; N = 3). Es la elección principal.
2. **Partición × estrato** (N = 30). Bloques no independientes.
3. **Participante × estrato** (participantes que estuvieron en prueba al menos una vez). Bloques no independientes.
4. Solo sanos (001-2017).
5. **Solo lesión medular (001-2019)**, analizado por separado.

Solo se usan bloques en los que los seis modelos tienen resultado.
''')
code(r'''
Q_ALFA = {6: 2.850}

def rangos(m: np.ndarray) -> np.ndarray:
    """Rangos por fila (bloque); 1 = mayor métrica."""
    return np.vstack([stats.rankdata(-r) for r in m])

def friedman(m: np.ndarray, n_perm: int, rng: np.random.Generator) -> dict:
    """Friedman (p asintótica), Iman-Davenport y p por permutación de rangos dentro de bloques.
    Se usa un único generador compartido en el mismo orden de llamadas que `04_analyze.py`, para reproducir la p por permutación."""
    N, k = m.shape; Rk = rangos(m)
    chi = lambda R: 12 * N / (k * (k + 1)) * np.sum((R.mean(0) - (k + 1) / 2) ** 2)
    c = chi(Rk); ff = (N - 1) * c / (N * (k - 1) - c) if N * (k - 1) - c > 0 else np.inf
    ge = sum(chi(np.vstack([rng.permutation(r) for r in Rk])) >= c - 1e-12 for _ in range(n_perm))
    return dict(N=N, k=k, chi2=c, p_asymptotic=1 - stats.chi2.cdf(c, k - 1), iman_davenport_F=ff,
                p_iman_davenport=1 - stats.f.cdf(ff, k - 1, (k - 1) * (N - 1)), p_permutation=(ge + 1) / (n_perm + 1),
                avg_ranks=dict(zip(CFG.MODELOS, Rk.mean(0))), CD=Q_ALFA[k] * np.sqrt(k * (k + 1) / (6 * N)))

def matriz(df: pd.DataFrame, metrica: str, bloques: list) -> pd.DataFrame:
    return df.pivot_table(index=bloques, columns="model", values=metrica)[list(CFG.MODELOS)].dropna()

cnt = REAL.groupby(["stratum", "seed_idx"]).model.nunique(); completos = cnt[cnt == len(CFG.MODELOS)].index
RB = REAL.set_index(["stratum", "seed_idx"]).loc[completos].reset_index()
PPR = PP[PP.control == CONTROL_REAL].set_index(["stratum", "seed_idx"]).loc[completos].reset_index()
ESTAD, MATRICES = {}, {}
RNG_FRIEDMAN = np.random.default_rng(CFG.SEMILLA_BASE)   # generador compartido (idéntico al script)
for met in ["balanced_accuracy", "mcc"]:
    m_est = RB.groupby(["stratum", "model"])[met].mean().unstack()[list(CFG.MODELOS)]
    m_par = matriz(RB, met, ["stratum", "seed_idx"])
    m_suj = PPR.groupby(["stratum", "subject", "model"])[met].mean().unstack()[list(CFG.MODELOS)].dropna()
    ESTAD[met] = {"blocks_strata": friedman(m_est.values, CFG.N_PERM_FRIEDMAN_ESTRATOS, RNG_FRIEDMAN),
                  "blocks_split_x_stratum": friedman(m_par.values, CFG.N_PERM_FRIEDMAN, RNG_FRIEDMAN),
                  "blocks_participant_x_stratum": friedman(m_suj.values, CFG.N_PERM_FRIEDMAN, RNG_FRIEDMAN)}
    sanos = RB[~RB.stratum.str.startswith("001-2019")]; lme = RB[RB.stratum.str.startswith("001-2019")]
    if len(sanos): ESTAD[met]["healthy_only_blocks_split"] = friedman(matriz(sanos, met, ["stratum", "seed_idx"]).values, CFG.N_PERM_FRIEDMAN, RNG_FRIEDMAN)
    if len(lme): ESTAD[met]["SCI_2019_only_blocks_split"] = friedman(matriz(lme, met, ["stratum", "seed_idx"]).values, CFG.N_PERM_FRIEDMAN, RNG_FRIEDMAN)
    MATRICES[met] = m_par
    for b, v in ESTAD[met].items():
        print(met, b, f"N={v['N']} chi2={v['chi2']:.3f} p={v['p_asymptotic']:.4f} p_perm={v['p_permutation']:.4f} DC={v['CD']:.2f}")
''')

md(r'''
## P-4. Comparaciones por pares (Wilcoxon-Holm y bayesiana con ROPE) y comparación frente al azar
- **Wilcoxon de rangos con signo** para los 15 pares (bloques partición × estrato), con **corrección de Holm**.
- **Prueba bayesiana de rangos con signo** (baycomp), con ROPE de 0,01 en exactitud balanceada y 0,02 en MCC. Devuelve P(A mejor), P(ROPE) y P(B mejor).
- **Análisis separado de 001-2019:** mismas pruebas con bloques = 10 particiones.
- **Frente al azar:** Wilcoxon unilateral de (exactitud balanceada − 1/K) sobre las particiones, con Holm en las 18 combinaciones estrato × modelo.
''')
code(r'''
def holm(p: np.ndarray) -> np.ndarray:
    """Valores p ajustados por Holm-Bonferroni."""
    p = np.asarray(p, float); o = np.argsort(p); m = len(p); adj = np.empty(m); acum = 0.0
    for i, j in enumerate(o):
        acum = max(acum, (m - i) * p[j]); adj[j] = min(1.0, acum)
    return adj

def pares(m: pd.DataFrame, metrica: str) -> pd.DataFrame:
    """Wilcoxon + Holm y bayesiana (ROPE) para todos los pares de modelos sobre los bloques de m."""
    rope = CFG.ROPE_BA if metrica == "balanced_accuracy" else CFG.ROPE_MCC; filas = []
    for a, b in itertools.combinations(CFG.MODELOS, 2):
        x, y = m[a].values, m[b].values
        try: p = stats.wilcoxon(x, y, zero_method="wilcox").pvalue
        except ValueError: p = 1.0
        pl, pe, pr = baycomp.two_on_multiple(x, y, rope=rope, random_state=CFG.SEMILLA_BASE)   # semilla fija del muestreo bayesiano
        filas.append(dict(metric=metrica, model_a=a, model_b=b, mean_diff=float(np.mean(x - y)), median_diff=float(np.median(x - y)), wilcoxon_p=p,
                          p_left_a_better=pl, p_rope=pe, p_right_b_better=pr))
    r = pd.DataFrame(filas); r["wilcoxon_p_holm"] = holm(r.wilcoxon_p.values); return r

PARES = {met: pares(MATRICES[met], met) for met in ["balanced_accuracy", "mcc"]}
for met, r in PARES.items(): r.to_csv(T / f"pairwise_wilcoxon_holm_bayes_{met}.csv", index=False)
lme = REAL[REAL.stratum == "001-2019_offline"]
if lme.seed_idx.nunique() > 1:
    PARES_LME = pd.concat([pares(lme.pivot(index="seed_idx", columns="model", values=met)[list(CFG.MODELOS)], met) for met in ["balanced_accuracy", "mcc"]])
    PARES_LME.to_csv(T / "pairwise_LME_2019_wilcoxon_holm_bayes.csv", index=False)
filas = []
for (e, mo), g in REAL.groupby(["stratum", "model"]):
    dif = g.balanced_accuracy.values - AZAR[e]
    p = stats.wilcoxon(dif, alternative="greater").pvalue if len(dif) >= 5 and np.any(dif != 0) else np.nan
    filas.append(dict(stratum=e, model=mo, n_splits=len(g), mean_ba=g.balanced_accuracy.mean(), chance=AZAR[e], mean_minus_chance=dif.mean(), wilcoxon_greater_p=p))
VS_AZAR = pd.DataFrame(filas); VS_AZAR["p_holm"] = holm(VS_AZAR.wilcoxon_greater_p.fillna(1).values); VS_AZAR.to_csv(T / "vs_chance.csv", index=False)
PERM = S[S.control == CONTROL_PERM].groupby(["stratum", "model", "seed_idx"])[METRICAS].mean().reset_index(); PERM.to_csv(T / "permutation_control.csv", index=False)
json.dump({"stats": ESTAD}, open(T / "stats.json", "w"), indent=1, default=float)
display(PARES["balanced_accuracy"].round(3)); display(VS_AZAR.round(4))
''')

md(r'''
## P-5. Figuras (solo en español): diagramas DC y distribución por partición
**Figura 4 del artículo:** diagrama de diferencia crítica de la exactitud balanceada con partición única (v2); aquí se generan además los DC de las particiones repetidas. Rango 1 = mejor; la barra roja une modelos cuya diferencia de rango promedio es menor que la DC. Se generan en PNG (300 ppp), SVG y PDF.
**Figura 5 del artículo:** exactitud balanceada en prueba en las 10 particiones disjuntas por participante (`Figura_cajas_exactitud_balanceada`); la línea discontinua indica el nivel de azar.
''')
code(r'''
def fc(x: float, n: int = 2) -> str:
    """Número con coma decimal."""
    return f"{x:.{n}f}".replace(".", ",")

def diagrama_dc(rangos_prom: dict, dc: float, titulo: str, archivo: Path, nota: str | None = None) -> None:
    """Diagrama de diferencia crítica (Demšar, 2006) con el estilo de las figuras del artículo."""
    nombres = sorted(rangos_prom, key=rangos_prom.get); r = np.array([rangos_prom[n] for n in nombres]); nombres = [NOMBRE_MODELO[n] for n in nombres]
    k = len(nombres); lo, hi = 1, k
    fig, ax = plt.subplots(figsize=(7.2, 2.1), dpi=300); ax.set_xlim(0, 10); ax.set_ylim(2.3, 6.4); ax.axis("off")
    X = lambda v: 2.2 + (v - lo) / (hi - lo) * 5.6; y0 = 4.8
    ax.plot([X(lo), X(hi)], [y0, y0], color="k", lw=1)
    for t in np.arange(lo, hi + 0.01, 0.5):
        ax.plot([X(t)] * 2, [y0, y0 + (0.18 if t % 1 == 0 else 0.1)], color="k", lw=0.8)
        if t % 1 == 0: ax.text(X(t), y0 + 0.28, f"{int(t)}", ha="center", fontsize=8)
    ax.text(X(hi) + 0.25, y0 + 0.28, "rango promedio (1 = mejor)", ha="left", fontsize=6.5, style="italic")
    ax.plot([X(lo), X(lo + dc)], [5.65, 5.65], color="k", lw=1.6)
    for xx in (X(lo), X(lo + dc)): ax.plot([xx, xx], [5.55, 5.75], color="k", lw=1)
    ax.text((X(lo) + X(lo + dc)) / 2, 5.82, f"DC = {fc(dc)}", ha="center", fontsize=7.5)
    mitad = (k + 1) // 2
    for i, (n, v) in enumerate(zip(nombres, r)):
        j = i if i < mitad else k - 1 - i; yy = y0 - 0.55 - j * 0.55
        ax.plot([X(v), X(v)], [y0, yy], color="k", lw=0.8)
        if i < mitad: ax.plot([X(v), 2.0], [yy, yy], color="k", lw=0.8); ax.text(1.95, yy, f"{n} ({fc(v)})", ha="right", va="center", fontsize=7.5)
        else: ax.plot([X(v), 8.0], [yy, yy], color="k", lw=0.8); ax.text(8.05, yy, f"({fc(v)}) {n}", ha="left", va="center", fontsize=7.5)
    grupos = []
    for i in range(k):
        j = i
        while j + 1 < k and r[j + 1] - r[i] < dc: j += 1
        if j > i and not any(a <= i and j <= b for a, b in grupos): grupos.append((i, j))
    for m_, (a, b) in enumerate(grupos):
        ax.plot([X(r[a]) - 0.05, X(r[b]) + 0.05], [y0 - 0.25 - m_ * 0.14] * 2, color="#b00000", lw=2.6, solid_capstyle="round")
    ax.text(5, 6.25, titulo, ha="center", fontsize=6.8, fontweight="bold")
    if nota: ax.text(5, 2.4, nota, ha="center", fontsize=6.6, style="italic")
    plt.tight_layout(pad=0.2)
    for ext in ("png", "svg", "pdf"): fig.savefig(f"{archivo}.{ext}", dpi=300)
    plt.close(fig)

FIG = CFG.ruta("figuras"); NOMBRE_MET = {"balanced_accuracy": "exactitud balanceada", "mcc": "MCC"}; ETQ = {"balanced_accuracy": "exactitud_balanceada", "mcc": "MCC"}
BLOQ = {"blocks_strata": ("bloques = estratos (media de las particiones)", "bloques_conjuntos", "estratos"),
        "blocks_split_x_stratum": ("bloques = partición × estrato (no independientes)", "bloques_particion_conjunto", "bloques"),
        "blocks_participant_x_stratum": ("bloques = participante × estrato", "bloques_participante_conjunto", "bloques"),
        "SCI_2019_only_blocks_split": ("solo 001-2019 (lesión medular), bloques = partición", "solo_LME_2019", "particiones")}
for met, res in ESTAD.items():
    for b, (txt, suf, unidad) in BLOQ.items():
        if b not in res: continue
        v = res[b]
        nota = f"Friedman χ²({v['k']-1}) = {fc(v['chi2'])}; p asintótica = {fc(v['p_asymptotic'],3)}; p por permutación = {fc(v['p_permutation'],3)}; N = {v['N']} {unidad}, k = 6 modelos, α = 0,05"
        diagrama_dc(v["avg_ranks"], v["CD"], f"Diferencia crítica (Nemenyi) — {NOMBRE_MET[met]}; {txt}", FIG / f"DC_{ETQ[met]}_{suf}", nota)

def figura_cajas(metrica: str, etiqueta_y: str, archivo: Path) -> None:
    """Distribución por partición de una métrica, por estrato y modelo, con línea de azar."""
    etq = ["CSP+\nLDA", "FBCSP+\nSVM", "MDM", "EEGNet", "Shallow-\nConvNet", "Deep-\nConvNet"]
    titulos = {"001-2017_ME": "001-2017 ejecución (7 clases)", "001-2017_MI": "001-2017 imaginación (7 clases)", "001-2019_offline": "001-2019 lesión medular (5 clases)"}
    est = [e for e in titulos if e in set(REAL.stratum)]
    fig, axs = plt.subplots(1, len(est), figsize=(7.2, 2.6), dpi=300, squeeze=False)
    for ax, e in zip(axs[0], est):
        datos = [REAL[(REAL.stratum == e) & (REAL.model == m)][metrica].values for m in CFG.MODELOS]
        ax.boxplot(datos, widths=.6, medianprops=dict(color="#b00000"), flierprops=dict(markersize=2))
        for i, v in enumerate(datos): ax.scatter(np.full(len(v), i + 1) + np.random.default_rng(i).uniform(-.15, .15, len(v)), v, s=3, color="k", alpha=.5, zorder=3)
        ax.axhline(AZAR[e] if metrica == "balanced_accuracy" else 0, ls="--", color="gray", lw=.8)
        ax.set_title(titulos[e], fontsize=7); ax.set_xticks(range(1, 7)); ax.set_xticklabels(etq, fontsize=5.2); ax.tick_params(axis="y", labelsize=6)
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: f'{x:.3f}'))  # punto decimal, como en el artículo
    axs[0][0].set_ylabel(etiqueta_y, fontsize=7)
    fig.text(.5, .005, "Cada punto es una partición 80/20 disjunta por participante (10 semillas); línea discontinua = " + ("nivel de azar" if metrica == "balanced_accuracy" else "MCC de 0 (azar)"), ha="center", fontsize=6, style="italic")
    plt.tight_layout(rect=(0, .04, 1, 1))
    for ext in ("png", "svg", "pdf"): fig.savefig(f"{archivo}.{ext}", dpi=300)
    plt.close(fig)

figura_cajas("balanced_accuracy", "Exactitud balanceada", FIG / "Figura_cajas_exactitud_balanceada")
figura_cajas("mcc", "MCC", FIG / "Figura_cajas_MCC")
print(sorted(p.name for p in FIG.glob("*.png")))
''')

md(r'''
## P-6. Tablas en español listas para el artículo
Las tablas usan coma decimal y el formato media ± DE sobre las particiones. Se escriben en `tablas/TABLAS_ES_listas_para_pegar.md` y `tablas/TABLAS_ES_estadistica.md`.
''')
code(r'''
NOMBRE_ESTRATO = {"001-2017_ME": "001-2017 ejecución (ME)", "001-2017_MI": "001-2017 imaginación (MI)", "001-2019_offline": "001-2019 LME (offline)",
                  "001-2020_gel": "001-2020 gel", "001-2020_water": "001-2020 agua", "001-2020_dry": "001-2020 seco"}
NOMBRE_METRICA = {"balanced_accuracy": "Exactitud balanceada", "macro_f1": "F1 macro", "cohen_kappa": "Kappa de Cohen", "mcc": "MCC"}
f3 = lambda x: fc(0.0 if abs(x) < 0.0005 else x, 3)

def tabla_media_de(metrica: str) -> list[str]:
    """Tabla estrato × modelo con media ± DE de la métrica."""
    R = RES[RES.control == CONTROL_REAL]
    lineas = [f"**Tabla. {NOMBRE_METRICA[metrica]} en prueba (media ± DE sobre particiones 80/20 disjuntas por participante; semillas 20261113 + k, k = 0…9).**", "",
              "| Estrato | Azar | " + " | ".join(NOMBRE_MODELO[m] for m in CFG.MODELOS) + " |", "|" + "---|" * (len(CFG.MODELOS) + 2)]
    for e in [e for e in NOMBRE_ESTRATO if e in set(R.stratum)]:
        az = "—" if metrica == "macro_f1" else f3(AZAR[e] if metrica == "balanced_accuracy" else 0.0)
        celdas = []
        for m in CFG.MODELOS:
            r = R[(R.stratum == e) & (R.model == m)].iloc[0]
            celdas.append(f"{f3(r[metrica+'_mean'])} ± {f3(r[metrica+'_std'])}" + (f" (n={int(r[metrica+'_count'])})" if r[metrica + "_count"] < 10 else ""))
        lineas.append(f"| {NOMBRE_ESTRATO[e]} | {az} | " + " | ".join(celdas) + " |")
    return lineas + [""]

(T / "TABLAS_ES_listas_para_pegar.md").write_text("\n".join(sum([tabla_media_de(m) for m in METRICAS], [])), encoding="utf-8")
BLOQ_ES = {"blocks_strata": "Estratos (media de particiones)", "blocks_split_x_stratum": "Partición × estrato", "blocks_participant_x_stratum": "Participante × estrato",
           "healthy_only_blocks_split": "Solo sanos (001-2017 ME/MI), partición × estrato", "SCI_2019_only_blocks_split": "Solo LME (001-2019), particiones"}
o = ["**Tabla. Prueba de Friedman y diferencia crítica de Nemenyi (k = 6 modelos, α = 0,05).**", "",
     "| Métrica | Bloques | N | χ²(5) | p asintótica | p Iman-Davenport | p por permutación | DC | Rangos promedio (1 = mejor) |", "|---|---|---|---|---|---|---|---|---|"]
for met, res in ESTAD.items():
    for b, v in res.items():
        rk = "; ".join(f"{NOMBRE_MODELO[x]} {fc(v['avg_ranks'][x])}" for x in sorted(v["avg_ranks"], key=v["avg_ranks"].get))
        pid = "—" if not np.isfinite(v["p_iman_davenport"]) else f3(v["p_iman_davenport"])
        o.append(f"| {NOMBRE_METRICA[met]} | {BLOQ_ES[b]} | {v['N']} | {fc(v['chi2'])} | {f3(v['p_asymptotic'])} | {pid} | {f3(v['p_permutation'])} | {fc(v['CD'])} | {rk} |")
def tabla_pares(df: pd.DataFrame, titulo: str) -> list[str]:
    l = ["", titulo, "", "| Métrica | Modelo A | Modelo B | Diferencia media (A − B) | p Wilcoxon | p Holm | P(A mejor) | P(ROPE) | P(B mejor) |", "|---|---|---|---|---|---|---|---|---|"]
    return l + [f"| {NOMBRE_METRICA[r.metric]} | {NOMBRE_MODELO[r.model_a]} | {NOMBRE_MODELO[r.model_b]} | {f3(r.mean_diff)} | {f3(r.wilcoxon_p)} | {f3(r.wilcoxon_p_holm)} | {f3(r.p_left_a_better)} | {f3(r.p_rope)} | {f3(r.p_right_b_better)} |" for r in df.itertuples()]
o += tabla_pares(pd.concat(PARES.values()), "**Tabla. Comparaciones por pares: Wilcoxon de rangos con signo (Holm) y prueba bayesiana de rangos con signo (ROPE = 0,01 en exactitud balanceada; 0,02 en MCC). Bloques = partición × estrato.**")
o += ["", "**Tabla. Exactitud balanceada frente al azar por estrato y modelo (Wilcoxon unilateral sobre particiones, corrección de Holm).**", "",
      "| Estrato | Modelo | Particiones | Media | Azar | Media − azar | p | p Holm |", "|---|---|---|---|---|---|---|---|"]
o += [f"| {NOMBRE_ESTRATO[r.stratum]} | {NOMBRE_MODELO[r.model]} | {r.n_splits} | {f3(r.mean_ba)} | {f3(r.chance)} | {f3(r.mean_minus_chance)} | {'—' if pd.isna(r.wilcoxon_greater_p) else f3(r.wilcoxon_greater_p)} | {f3(r.p_holm)} |" for r in VS_AZAR.itertuples()]
if len(PERM):
    o += ["", "**Tabla. Control con etiquetas de entrenamiento permutadas dentro de cada participante.**", "", "| Estrato | Modelo | Partición (semilla) | Exactitud balanceada | MCC |", "|---|---|---|---|---|"]
    o += [f"| {NOMBRE_ESTRATO[r.stratum]} | {NOMBRE_MODELO[r.model]} | {semilla(int(r.seed_idx))} | {f3(r.balanced_accuracy)} | {f3(r.mcc)} |" for r in PERM.sort_values(["stratum", "model", "seed_idx"]).itertuples()]
if "PARES_LME" in globals():
    o += tabla_pares(PARES_LME, "**Tabla. Análisis separado de 001-2019 (lesión medular): Wilcoxon (Holm) y bayesiana (ROPE). Bloques = 10 particiones.**")
(T / "TABLAS_ES_estadistica.md").write_text("\n".join(o) + "\n", encoding="utf-8")
print((T / "TABLAS_ES_listas_para_pegar.md").read_text()[:1500])
''')

md(r'''
## P-7. Verificación frente a los resultados del artículo
**Qué:** compara los números regenerados con los publicados.
- **Siempre:** con k = 0, los nueve resultados clásicos deben coincidir con la ejecución v3 del estudio (tolerancia 5·10⁻⁵; los modelos clásicos son deterministas).
- **En modo COMPLETO o con predicciones guardadas:** las medias de exactitud balanceada y MCC por estrato × modelo deben coincidir con las tablas del artículo. Las tolerancias son 5·10⁻⁴ para los clásicos y 0,02 para las redes con reentrenamiento en GPU (exactas con predicciones guardadas). También se comparan χ² y p de Friedman.
''')
code(r'''
REFERENCIA = ''' + json.dumps(REF, indent=0) + r'''
REFERENCIA_FRIEDMAN = ''' + json.dumps(REF_FRIED, indent=0) + r'''
DAVID_V3_K0 = ''' + json.dumps(DAVID_V3, indent=0) + r'''

def verificar() -> None:
    """Afirma la coincidencia de los números regenerados con los del artículo; imprime un resumen."""
    r0 = REAL[REAL.seed_idx == 0].set_index(["stratum", "model"]).balanced_accuracy
    for clave, v in DAVID_V3_K0.items():
        e, m = clave.split("|")
        if (e, m) in r0.index:
            assert abs(r0[(e, m)] - v) < 5e-5, (clave, r0[(e, m)], v)
    print("OK: k = 0 reproduce los resultados clásicos v3.")
    completo = CFG.USAR_PREDICCIONES_GUARDADAS or CFG.MODO == "COMPLETO"
    if not completo:
        print("Modo RAPIDO: la verificación de medias sobre 10 particiones no aplica."); return
    R = RES[RES.control == CONTROL_REAL].set_index(["stratum", "model"])
    for clave, v in REFERENCIA.items():
        e, m = clave.split("|")
        tol = 5e-4 if (m not in REDES or CFG.USAR_PREDICCIONES_GUARDADAS) else 0.02
        assert abs(R.loc[(e, m), "balanced_accuracy_mean"] - v["ba_mean"]) <= tol, (clave, "BA")
        assert abs(R.loc[(e, m), "mcc_mean"] - v["mcc_mean"]) <= tol, (clave, "MCC")
    if CFG.USAR_PREDICCIONES_GUARDADAS:
        for met, bl in REFERENCIA_FRIEDMAN.items():
            for b, v in bl.items():
                assert abs(ESTAD[met][b]["chi2"] - v["chi2"]) < 1e-2 and abs(ESTAD[met][b]["p_asymptotic"] - v["p_asymptotic"]) < 1e-3, (met, b)
    print("OK: medias por estrato × modelo (y Friedman con predicciones guardadas) coinciden con el artículo.")

verificar()
''')

md(r'''
---
# Limitaciones y desviaciones
1. **001-2020 (gel, agua, seco):** el servidor oficial `lampx.tugraz.at` no respondió el 16-11-2026, y las copias archivadas están truncadas. Las particiones repetidas no se ejecutaron. Para esos estratos el artículo solo informa la partición única v3. El cuaderno los incluye automáticamente (`INCLUIR_2020_SI_DISPONIBLE = True`) cuando el servidor está disponible.
2. **Particiones repetidas en 001-2019:** solo existen 45 particiones posibles (2 de 10 participantes en prueba). Las semillas 0 y 6 (P06, P10) y 2 y 8 (P02, P04) coinciden, así que hay 8 conjuntos de prueba distintos. P07, S05 (ME) y S03/S04 (MI) nunca quedan en prueba.
3. **Independencia:** los bloques partición × estrato y participante × estrato comparten participantes de entrenamiento; sus valores p son descriptivos. Los bloques = estratos son solo 3 (Demšar recomienda ≥ 10 conjuntos independientes).
4. **Determinismo en GPU:** variación de hasta 0,0125 en exactitud balanceada para las redes entre ejecuciones.
5. **Configuración de las redes en 001-2019:** la ejecución v3 se hizo sin ponderación ni programador, contrario a la configuración congelada. Los resultados finales usan la configuración congelada; los anteriores se documentan en el CHANGELOG.
6. **p por permutación de Friedman:** Monte Carlo (20 000 o 5 000 permutaciones), no enumeración exacta.
7. **Prueba bayesiana:** la de rangos con signo no jerárquica (baycomp).
8. **Control por permutación:** 2 semillas (k = 0, 1).
''')

for i, c in enumerate(cells): c['id'] = f'celda-{i:02d}'
nb = nbf.v4.new_notebook(); nb['cells'] = cells
nb['metadata'] = {"kernelspec": {"name": "python3", "display_name": "Python 3"}, "language_info": {"name": "python"},
                  "accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"}}
out = R/'TESIS_pipeline_reproducible.ipynb'; nbf.write(nb, out); print('escrito', out, len(cells), 'celdas')
