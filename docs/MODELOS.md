# Modelos

Los seis modelos usan hiperparámetros **fijos**, tomados de la configuración establecida del estudio (`configs/CONFIGURACION_CONGELADA_V2_ANTES_CONTROL_PERMUTACION.json`) y del código v3 que se ejecutó. **No se ajusta ningún hiperparámetro con los datos de prueba** y no hay validación interna. La normalización, los filtros espaciales y los pesos de clase se estiman **solo con los participantes de entrenamiento**.

## Clásicos (CPU; deterministas)
| Modelo | Preprocesamiento | Extracción | Clasificador | Código |
|---|---|---|---|---|
| **CSP+LDA** | Butterworth pasa-banda de orden 4, 8–30 Hz, fase cero (`sosfiltfilt`) | CSP multiclase, 12 componentes, regularización Ledoit-Wolf, log-varianza | LDA `solver='lsqr'`, `shrinkage='auto'` | `scripts/02_run_classical.py` |
| **FBCSP+SVM** | 6 bandas: 8–12, 12–16, 16–20, 20–24, 24–28 y 28–32 Hz (orden 4, fase cero) | CSP con 4 componentes por banda (Ledoit-Wolf), 24 rasgos y `StandardScaler` | SVM RBF, C = 1, `gamma='scale'`, `class_weight='balanced'` | ídem |
| **MDM** | 8–30 Hz | Covarianza OAS (`pyriemann.Covariances`) | Mínima distancia a la media riemanniana (`metric='riemann'`) | ídem |

El filtrado es por época y no usa etiquetas, así que se calcula una vez por conjunto de evaluación antes de dividir. Es equivalente al código del estudio.

## Profundos (GPU T4)
Entrada: (1, canales, 513 muestras), normalizada por canal con la media y la desviación estándar del entrenamiento.

| Modelo | Arquitectura (PyTorch) |
|---|---|
| **EEGNet** | Conv temporal 8 × (1, 64) → BN → Conv espacial en profundidad (C, 1) con D = 2 (16) → BN → ELU → AvgPool (1, 4) → Dropout 0,25 → Conv separable 16 × (1, 16) + punto a punto → BN → ELU → AvgPool (1, 8) → Dropout 0,25 → GAP → Lineal |
| **ShallowConvNet** | Conv temporal 40 × (1, 25) → Conv espacial 40 × (C, 1) → BN → cuadrado → AvgPool (1, 75), paso 15 → log(máx(x, 10⁻⁶)) → Dropout 0,5 → GAP → Lineal |
| **DeepConvNet** | [Conv 25 × (1, 10) → Conv 25 × (C, 1) → BN → ELU → MaxPool (1, 3) → Dropout 0,5] → 3 bloques [Conv (1, 10) con 50, 100 y 200 filtros → BN → ELU → MaxPool (1, 3) → Dropout 0,5] → GAP → Lineal |

### Entrenamiento (común)
| Parámetro | Valor |
|---|---|
| Épocas | 20 (sin parada temprana) |
| Optimizador | AdamW, lr = 10⁻³, weight decay = 10⁻⁴ |
| Lote | 64 en entrenamiento (barajado con generador propio); 128 en evaluación |
| Semilla | `semilla_partición + Σ ord(modelo + conjunto de evaluación + control)` para torch, numpy, random y el generador del DataLoader |
| cuDNN | `benchmark = True` (ver [REPRODUCIBILIDAD.md](REPRODUCIBILIDAD.md)) |

### Configuración final por conjunto de evaluación
| Conjunto de evaluación | Pérdida | Programador de tasa de aprendizaje |
|---|---|---|
| 001-2017 ME / MI | Entropía cruzada **no ponderada** | Ninguno |
| **001-2019 offline** (y 001-2020) | Entropía cruzada **ponderada por clase**, pesos wᶜ = n / (K · nᶜ) con las etiquetas de entrenamiento (las permutadas en el control) | **`CosineAnnealingLR(T_max = 20)`**, un paso por época |

> **Corrección incorporada (ver [CHANGELOG.md](CHANGELOG.md)).** La configuración establecida indica ponderación por clase y programador coseno para 001-2019 y 001-2020, pero la ejecución v3 del estudio entrenó las redes de 001-2019 sin ellos. Los resultados finales de este repositorio reejecutan las tres redes en 001-2019, en las 10 particiones y en el control por permutación (k = 0, 1), con la configuración establecida. Los valores anteriores se conservan en `tables/anteriores_sin_ponderacion_ni_scheduler/`.
