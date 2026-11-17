# Resultados — particiones repetidas disjuntas por participante (TESIS EEG)

**Ejecutado por:** DevOPS (Grok Bot), 16-11-2026, 19:20–21:20 (hora de Bogotá). Reproduce el protocolo establecido v3 de David (correcciones Ferrin v2/v3).
**Versión de este informe:** v2. Incorpora la reejecución de las redes profundas en 001-2019 con ponderación por clase y programador coseno (§1.1). Los resultados de la v1, sin ponderación ni programador, están en `tables/anteriores_sin_ponderacion_ni_scheduler/`.
**Estado:** son corridas nuevas que complementan la tabla v3 de partición única; no la reemplazan.

**Conjunto de evaluación:** cada combinación de conjunto de datos y condición que se evalúa por separado (001-2017 ejecución, 001-2017 imaginación y 001-2019 lesión medular). En las pruebas estadísticas, cada conjunto de evaluación actúa como un bloque.

## 1. Qué se ejecutó

- **Conjuntos de evaluación:**
  - 001-2017 ejecución (ME): 14 participantes, 5 880 épocas, 7 clases.
  - 001-2017 imaginación (MI): 15 participantes, 6 300 épocas, 7 clases.
  - 001-2019 lesión medular, offline: 10 participantes, 3 598 épocas, 5 clases.
- **001-2020 (gel, agua, seco) no se ejecutó.** El servidor oficial de BNCI (lampx.tugraz.at, al que redirige bnci-horizon-2020.eu) no respondió en toda la sesión, ni desde el computador de trabajo ni desde Colab. Lo revisé de forma periódica entre las 19:30 y las 20:53. Las copias del Internet Archive están truncadas en 5 MiB y no encontré otro espejo público. Para 001-2020 solo queda la partición única v3 de David.
- **Datos:**
  - Fuentes públicas oficiales: Zenodo 834976 para 001-2017 y Zenodo 2222268 para 001-2019.
  - Las épocas se re-extrajeron con los manifiestos v2 de David (mismo archivo fuente, mismo `onset_sample` y misma clase), con ventana de −0,5 a 1,5 s.
  - No hubo filtrado previo, corrección de línea base ni rechazo por amplitud.
  - 001-2017: primeros 61 canales, remuestreo de 512 a 256 Hz.
  - 001-2019: primeros 61 canales y solo épocas con `finite_for_model`.
  - Todos los onsets del manifiesto están **a 1 muestra o menos** de una anotación del archivo crudo. Los datos crudos solo están en el computador de trabajo y en VMs efímeras de Colab ya liberadas.
- **Modelos:** CSP+LDA, FBCSP+SVM, MDM, EEGNet, ShallowConvNet y DeepConvNet, con hiperparámetros fijos idénticos al código v3 (detalle en `docs/MODELOS.md` del repositorio).
  - Redes en 001-2017: pérdida no ponderada y sin programador, como en la configuración establecida.
  - **Redes en 001-2019: ponderación por clase n/(K·nᶜ) y `CosineAnnealingLR(T_max = 20)`**, como en la configuración establecida (§1.1).
- **Particiones:** 10 particiones 80/20 disjuntas por participante en cada conjunto de evaluación, con semillas 20261113 + k (k = 0…9) y la fórmula de David (`default_rng(semilla + Σ ord(nombre))`, ⌈20 %⌉ de participantes en prueba). Los seis modelos usan las mismas particiones. **k = 0 es exactamente la partición de David.**
- **Control de fuga de información:** con k = 0 y k = 1, los 6 modelos × 3 conjuntos de evaluación se reentrenaron con las etiquetas de entrenamiento permutadas dentro de cada participante.
- **Salidas:** 216 corridas con predicciones por época. A partir de ellas se calcularon exactitud balanceada, F1 macro, kappa de Cohen, MCC, métricas por participante y matrices de confusión.
- **Cómputo:**
  - Redes en Colab Pro T4; modelos clásicos en Colab CPU y en el computador de trabajo.
  - **Unidades de cómputo totales: 3,21** (saldo de 591,39 a 588,18). De ellas, la reejecución de §1.1 usó **0,30** (588,48 → 588,18) en 1 sesión T4 (`w19`), de unos 22 minutos.
  - No queda ninguna sesión activa (`Active assignments: 0`).

### 1.1 Reejecución de las redes en 001-2019 con ponderación por clase y programador coseno
- **Motivo:** la configuración establecida (`CONFIGURACION_CONGELADA_V2_ANTES_CONTROL_PERMUTACION.json`) indica para 001-2019/001-2020 «ponderación por clase y scheduler coseno». El código v3 que produjo los números de David no los aplicaba.
- **Configuración aplicada**, igual a `train_net` del cuaderno fuente de David:
  - `CrossEntropyLoss(weight = n_entrenamiento / (K · conteo_clase))`, con los pesos calculados sobre las etiquetas de entrenamiento (las permutadas en el control).
  - `CosineAnnealingLR(T_max = 20)`, un paso por época.
  - Todo lo demás es idéntico: arquitecturas, 20 épocas, AdamW con lr 1e-3 y wd 1e-4, lote de 64, semillas y particiones.
- **Alcance:** EEGNet, ShallowConvNet y DeepConvNet en 001-2019, con 10 particiones de etiquetas reales y 2 de permutación (k = 0, 1). Las 36 corridas terminaron.

| Modelo (001-2019) | Exactitud balanceada, sin ponderación ni programador (v1) | **Exactitud balanceada, con ponderación + coseno (final)** | MCC (v1) | **MCC (final)** | Exactitud balanceada permutada (final, k = 0 / 1) |
|---|---|---|---|---|---|
| EEGNet | 0,196 ± 0,010 | **0,199 ± 0,013** | −0,005 ± 0,013 | **−0,001 ± 0,018** | 0,175 / 0,192 |
| ShallowConvNet | 0,202 ± 0,009 | **0,204 ± 0,011** | 0,002 ± 0,013 | **0,005 ± 0,015** | 0,175 / 0,191 |
| DeepConvNet | 0,199 ± 0,013 | **0,201 ± 0,015** | 0,000 ± 0,019 | **0,001 ± 0,019** | 0,207 / 0,197 |

**¿Cambia alguna conclusión? No.**
- Las tres redes siguen en el nivel de azar (0,200): p de Holm = 1 frente al azar.
- Los rangos con bloques = conjuntos de evaluación no cambian: MDM 1,67; CSP+LDA 2,17; FBCSP+SVM 3,17; ShallowConvNet 3,33; EEGNet 5,00; DeepConvNet 5,67. Por eso Friedman da el mismo resultado: χ² = 10,48, p asintótica = 0,063, p por permutación = 0,026, Iman-Davenport 0,019, DC = 4,35, y ningún par difiere según Nemenyi.
- Con bloques alternativos, Friedman pasa de p = 0,463 a 0,565 (partición × conjunto de evaluación) y de 0,605 a 0,518 (participante × conjunto de evaluación).
- En el análisis separado de 001-2019, Friedman pasa de p = 0,150 a 0,227 (exactitud balanceada) y de 0,122 a 0,199 (MCC).
- Ninguna comparación por pares sobrevive a Holm. El MDM sigue siendo el único modelo con exactitud balanceada consistentemente por encima del azar en 001-2019 (p = 0,005 sin corregir; p de Holm = 0,088).

### Verificación de reproducción (semilla k = 0 frente a la v3 de David)
- Los 9 valores clásicos (3 modelos × 3 conjuntos de evaluación) coinciden **exactamente** con la v3; por ejemplo, en 001-2019, CSP+LDA 0,2167, FBCSP+SVM 0,2083 y MDM 0,2236.
- Con la configuración v3 (sin ponderación), las redes difieren como máximo 0,0125 en exactitud balanceada, por el no determinismo de la GPU (`tables/reproduccion_semilla0_vs_david.csv`, calculado con las corridas v1).
- Con la configuración final, las redes de 001-2019 dan en k = 0: EEGNet 0,194, ShallowConvNet 0,213 y DeepConvNet 0,189.

## 2. Resultados con etiquetas reales (10 particiones por conjunto de evaluación)

**Tabla. Exactitud balanceada en prueba (media ± DE sobre particiones 80/20 disjuntas por participante; semillas 20261113 + k, k = 0…9).**

| Conjunto de evaluación | Azar | CSP+LDA | FBCSP+SVM | MDM | EEGNet | ShallowConvNet | DeepConvNet |
|---|---|---|---|---|---|---|---|
| 001-2017 ejecución (ME) | 0,143 | 0,144 ± 0,003 | 0,144 ± 0,003 | 0,144 ± 0,006 | 0,144 ± 0,004 | 0,143 ± 0,006 | 0,142 ± 0,003 |
| 001-2017 imaginación (MI) | 0,143 | 0,142 ± 0,004 | 0,142 ± 0,003 | 0,143 ± 0,001 | 0,142 ± 0,002 | 0,144 ± 0,005 | 0,142 ± 0,002 |
| 001-2019 LME (offline) | 0,200 | 0,211 ± 0,017 | 0,208 ± 0,019 | 0,220 ± 0,017 | 0,199 ± 0,013 | 0,204 ± 0,011 | 0,201 ± 0,015 |

**Tabla. F1 macro en prueba (media ± DE sobre particiones 80/20 disjuntas por participante; semillas 20261113 + k, k = 0…9).**

| Conjunto de evaluación | Azar | CSP+LDA | FBCSP+SVM | MDM | EEGNet | ShallowConvNet | DeepConvNet |
|---|---|---|---|---|---|---|---|
| 001-2017 ejecución (ME) | — | 0,073 ± 0,012 | 0,053 ± 0,026 | 0,078 ± 0,024 | 0,081 ± 0,021 | 0,101 ± 0,015 | 0,071 ± 0,018 |
| 001-2017 imaginación (MI) | — | 0,073 ± 0,014 | 0,050 ± 0,017 | 0,056 ± 0,012 | 0,076 ± 0,028 | 0,091 ± 0,021 | 0,062 ± 0,018 |
| 001-2019 LME (offline) | — | 0,170 ± 0,040 | 0,177 ± 0,022 | 0,184 ± 0,034 | 0,176 ± 0,016 | 0,186 ± 0,016 | 0,172 ± 0,028 |

**Tabla. Kappa de Cohen en prueba (media ± DE sobre particiones 80/20 disjuntas por participante; semillas 20261113 + k, k = 0…9).**

| Conjunto de evaluación | Azar | CSP+LDA | FBCSP+SVM | MDM | EEGNet | ShallowConvNet | DeepConvNet |
|---|---|---|---|---|---|---|---|
| 001-2017 ejecución (ME) | 0,000 | 0,001 ± 0,004 | 0,001 ± 0,004 | 0,001 ± 0,007 | 0,001 ± 0,004 | 0,000 ± 0,007 | -0,001 ± 0,004 |
| 001-2017 imaginación (MI) | 0,000 | 0,000 ± 0,004 | 0,000 ± 0,003 | 0,000 ± 0,001 | -0,001 ± 0,002 | 0,001 ± 0,006 | -0,001 ± 0,003 |
| 001-2019 LME (offline) | 0,000 | 0,013 ± 0,021 | 0,010 ± 0,023 | 0,025 ± 0,021 | -0,001 ± 0,017 | 0,005 ± 0,014 | 0,001 ± 0,018 |

**Tabla. MCC en prueba (media ± DE sobre particiones 80/20 disjuntas por participante; semillas 20261113 + k, k = 0…9).**

| Conjunto de evaluación | Azar | CSP+LDA | FBCSP+SVM | MDM | EEGNet | ShallowConvNet | DeepConvNet |
|---|---|---|---|---|---|---|---|
| 001-2017 ejecución (ME) | 0,000 | 0,002 ± 0,005 | 0,001 ± 0,006 | 0,001 ± 0,008 | 0,001 ± 0,005 | 0,000 ± 0,008 | -0,001 ± 0,006 |
| 001-2017 imaginación (MI) | 0,000 | 0,000 ± 0,008 | 0,000 ± 0,006 | 0,001 ± 0,006 | -0,002 ± 0,005 | 0,001 ± 0,007 | -0,001 ± 0,004 |
| 001-2019 LME (offline) | 0,000 | 0,014 ± 0,022 | 0,011 ± 0,025 | 0,027 ± 0,023 | -0,001 ± 0,018 | 0,005 ± 0,015 | 0,001 ± 0,019 |

## 3. Estadística

**Tabla. Prueba de Friedman y diferencia crítica de Nemenyi (k = 6 modelos, α = 0,05).**

| Métrica | Bloques | N | χ²(5) | p asintótica | p Iman-Davenport | p por permutación | DC | Rangos promedio (1 = mejor) |
|---|---|---|---|---|---|---|---|---|
| Exactitud balanceada | Conjuntos de evaluación (media de particiones) | 3 | 10,48 | 0,063 | 0,019 | 0,026 | 4,35 | MDM 1,67; CSP+LDA 2,17; FBCSP+SVM 3,17; ShallowConvNet 3,33; EEGNet 5,00; DeepConvNet 5,67 |
| Exactitud balanceada | Partición × conjunto de evaluación | 30 | 3,89 | 0,565 | 0,571 | 0,541 | 1,38 | MDM 3,03; CSP+LDA 3,33; FBCSP+SVM 3,48; ShallowConvNet 3,53; EEGNet 3,72; DeepConvNet 3,90 |
| Exactitud balanceada | Participante × conjunto de evaluación | 35 | 4,22 | 0,518 | 0,523 | 0,450 | 1,27 | MDM 3,17; ShallowConvNet 3,29; CSP+LDA 3,31; FBCSP+SVM 3,60; EEGNet 3,70; DeepConvNet 3,93 |
| Exactitud balanceada | Solo sanos (001-2017 ME/MI), partición × conjunto de evaluación | 20 | 0,69 | 0,983 | 0,985 | 0,979 | 1,69 | FBCSP+SVM 3,30; MDM 3,38; CSP+LDA 3,42; EEGNet 3,60; ShallowConvNet 3,60; DeepConvNet 3,70 |
| Exactitud balanceada | Solo LME (001-2019), particiones | 10 | 6,91 | 0,227 | 0,227 | 0,212 | 2,38 | MDM 2,35; CSP+LDA 3,15; ShallowConvNet 3,40; FBCSP+SVM 3,85; EEGNet 3,95; DeepConvNet 4,30 |
| MCC | Conjuntos de evaluación (media de particiones) | 3 | 10,81 | 0,055 | 0,013 | 0,023 | 4,35 | MDM 1,33; CSP+LDA 2,00; FBCSP+SVM 3,67; ShallowConvNet 3,67; EEGNet 5,00; DeepConvNet 5,33 |
| MCC | Partición × conjunto de evaluación | 30 | 4,05 | 0,542 | 0,548 | 0,532 | 1,38 | MDM 2,97; CSP+LDA 3,33; ShallowConvNet 3,53; FBCSP+SVM 3,65; EEGNet 3,72; DeepConvNet 3,80 |
| MCC | Participante × conjunto de evaluación | 35 | 2,94 | 0,709 | 0,715 | 0,683 | 1,27 | MDM 3,20; CSP+LDA 3,31; ShallowConvNet 3,36; EEGNet 3,61; FBCSP+SVM 3,71; DeepConvNet 3,80 |
| MCC | Solo sanos (001-2017 ME/MI), partición × conjunto de evaluación | 20 | 0,45 | 0,994 | 0,994 | 0,995 | 1,69 | MDM 3,30; CSP+LDA 3,40; FBCSP+SVM 3,52; DeepConvNet 3,55; EEGNet 3,60; ShallowConvNet 3,62 |
| MCC | Solo LME (001-2019), particiones | 10 | 7,30 | 0,199 | 0,197 | 0,193 | 2,38 | MDM 2,30; CSP+LDA 3,20; ShallowConvNet 3,35; FBCSP+SVM 3,90; EEGNet 3,95; DeepConvNet 4,30 |

**Tabla. Comparaciones por pares: Wilcoxon de rangos con signo (corrección de Holm) y prueba bayesiana de rangos con signo (ROPE = 0,01 en exactitud balanceada; 0,02 en MCC). Bloques = partición × conjunto de evaluación.**

| Métrica | Modelo A | Modelo B | Diferencia media (A − B) | p Wilcoxon | p Holm | P(A mejor) | P(ROPE) | P(B mejor) |
|---|---|---|---|---|---|---|---|---|
| Exactitud balanceada | CSP+LDA | FBCSP+SVM | 0,001 | 0,301 | 1,000 | 0,000 | 0,999 | 0,001 |
| Exactitud balanceada | CSP+LDA | MDM | -0,003 | 0,346 | 1,000 | 0,000 | 0,999 | 0,001 |
| Exactitud balanceada | CSP+LDA | EEGNet | 0,004 | 0,245 | 1,000 | 0,014 | 0,986 | 0,000 |
| Exactitud balanceada | CSP+LDA | ShallowConvNet | 0,002 | 0,548 | 1,000 | 0,001 | 0,999 | 0,000 |
| Exactitud balanceada | CSP+LDA | DeepConvNet | 0,004 | 0,136 | 1,000 | 0,005 | 0,995 | 0,000 |
| Exactitud balanceada | FBCSP+SVM | MDM | -0,004 | 0,044 | 0,614 | 0,000 | 0,998 | 0,002 |
| Exactitud balanceada | FBCSP+SVM | EEGNet | 0,003 | 0,665 | 1,000 | 0,013 | 0,986 | 0,000 |
| Exactitud balanceada | FBCSP+SVM | ShallowConvNet | 0,002 | 0,787 | 1,000 | 0,002 | 0,998 | 0,000 |
| Exactitud balanceada | FBCSP+SVM | DeepConvNet | 0,003 | 0,063 | 0,823 | 0,001 | 0,999 | 0,000 |
| Exactitud balanceada | MDM | EEGNet | 0,007 | 0,106 | 1,000 | 0,088 | 0,912 | 0,000 |
| Exactitud balanceada | MDM | ShallowConvNet | 0,006 | 0,198 | 1,000 | 0,013 | 0,987 | 0,000 |
| Exactitud balanceada | MDM | DeepConvNet | 0,007 | 0,016 | 0,238 | 0,115 | 0,885 | 0,000 |
| Exactitud balanceada | EEGNet | ShallowConvNet | -0,002 | 0,863 | 1,000 | 0,000 | 1,000 | 0,000 |
| Exactitud balanceada | EEGNet | DeepConvNet | 0,000 | 0,737 | 1,000 | 0,000 | 0,999 | 0,000 |
| Exactitud balanceada | ShallowConvNet | DeepConvNet | 0,002 | 0,270 | 1,000 | 0,000 | 0,999 | 0,000 |
| MCC | CSP+LDA | FBCSP+SVM | 0,002 | 0,387 | 1,000 | 0,000 | 1,000 | 0,000 |
| MCC | CSP+LDA | MDM | -0,005 | 0,358 | 1,000 | 0,000 | 1,000 | 0,000 |
| MCC | CSP+LDA | EEGNet | 0,006 | 0,198 | 1,000 | 0,000 | 1,000 | 0,000 |
| MCC | CSP+LDA | ShallowConvNet | 0,003 | 0,430 | 1,000 | 0,000 | 1,000 | 0,000 |
| MCC | CSP+LDA | DeepConvNet | 0,006 | 0,210 | 1,000 | 0,001 | 0,999 | 0,000 |
| MCC | FBCSP+SVM | MDM | -0,006 | 0,032 | 0,446 | 0,000 | 1,000 | 0,000 |
| MCC | FBCSP+SVM | EEGNet | 0,004 | 0,631 | 1,000 | 0,001 | 0,999 | 0,000 |
| MCC | FBCSP+SVM | ShallowConvNet | 0,002 | 0,787 | 1,000 | 0,000 | 1,000 | 0,000 |
| MCC | FBCSP+SVM | DeepConvNet | 0,004 | 0,078 | 0,936 | 0,000 | 1,000 | 0,000 |
| MCC | MDM | EEGNet | 0,011 | 0,045 | 0,591 | 0,013 | 0,988 | 0,000 |
| MCC | MDM | ShallowConvNet | 0,008 | 0,164 | 1,000 | 0,003 | 0,997 | 0,000 |
| MCC | MDM | DeepConvNet | 0,010 | 0,026 | 0,393 | 0,007 | 0,993 | 0,000 |
| MCC | EEGNet | ShallowConvNet | -0,003 | 0,567 | 1,000 | 0,000 | 1,000 | 0,000 |
| MCC | EEGNet | DeepConvNet | 0,000 | 1,000 | 1,000 | 0,000 | 1,000 | 0,000 |
| MCC | ShallowConvNet | DeepConvNet | 0,002 | 0,239 | 1,000 | 0,000 | 1,000 | 0,000 |

**Tabla. Exactitud balanceada frente al azar por conjunto de evaluación y modelo (Wilcoxon unilateral sobre particiones, corrección de Holm).**

| Conjunto de evaluación | Modelo | Particiones | Media | Azar | Media − azar | p | p Holm |
|---|---|---|---|---|---|---|---|
| 001-2017 ejecución (ME) | CSP+LDA | 10 | 0,144 | 0,143 | 0,001 | 0,156 | 1,000 |
| 001-2017 ejecución (ME) | DeepConvNet | 10 | 0,142 | 0,143 | 0,000 | 0,764 | 1,000 |
| 001-2017 ejecución (ME) | EEGNet | 10 | 0,144 | 0,143 | 0,001 | 0,238 | 1,000 |
| 001-2017 ejecución (ME) | FBCSP+SVM | 10 | 0,144 | 0,143 | 0,001 | 0,359 | 1,000 |
| 001-2017 ejecución (ME) | MDM | 10 | 0,144 | 0,143 | 0,001 | 0,305 | 1,000 |
| 001-2017 ejecución (ME) | ShallowConvNet | 10 | 0,143 | 0,143 | 0,000 | 0,699 | 1,000 |
| 001-2017 imaginación (MI) | CSP+LDA | 10 | 0,142 | 0,143 | 0,000 | 0,648 | 1,000 |
| 001-2017 imaginación (MI) | DeepConvNet | 10 | 0,142 | 0,143 | -0,001 | 0,859 | 1,000 |
| 001-2017 imaginación (MI) | EEGNet | 10 | 0,142 | 0,143 | -0,001 | 0,895 | 1,000 |
| 001-2017 imaginación (MI) | FBCSP+SVM | 10 | 0,142 | 0,143 | 0,000 | 0,719 | 1,000 |
| 001-2017 imaginación (MI) | MDM | 10 | 0,143 | 0,143 | 0,000 | 0,375 | 1,000 |
| 001-2017 imaginación (MI) | ShallowConvNet | 10 | 0,144 | 0,143 | 0,001 | 0,336 | 1,000 |
| 001-2019 LME (offline) | CSP+LDA | 10 | 0,211 | 0,200 | 0,011 | 0,092 | 1,000 |
| 001-2019 LME (offline) | DeepConvNet | 10 | 0,201 | 0,200 | 0,001 | 0,626 | 1,000 |
| 001-2019 LME (offline) | EEGNet | 10 | 0,199 | 0,200 | -0,001 | 0,602 | 1,000 |
| 001-2019 LME (offline) | FBCSP+SVM | 10 | 0,208 | 0,200 | 0,008 | 0,153 | 1,000 |
| 001-2019 LME (offline) | MDM | 10 | 0,220 | 0,200 | 0,020 | 0,005 | 0,088 |
| 001-2019 LME (offline) | ShallowConvNet | 10 | 0,204 | 0,200 | 0,004 | 0,281 | 1,000 |

**Tabla. Control con etiquetas de entrenamiento permutadas dentro de cada participante (exactitud balanceada en prueba).**

| Conjunto de evaluación | Modelo | Partición (semilla) | Exactitud balanceada | MCC |
|---|---|---|---|---|
| 001-2017 ejecución (ME) | CSP+LDA | 20261113 | 0,145 | 0,003 |
| 001-2017 ejecución (ME) | CSP+LDA | 20261114 | 0,148 | 0,009 |
| 001-2017 ejecución (ME) | DeepConvNet | 20261113 | 0,144 | 0,001 |
| 001-2017 ejecución (ME) | DeepConvNet | 20261114 | 0,143 | 0,000 |
| 001-2017 ejecución (ME) | EEGNet | 20261113 | 0,143 | 0,000 |
| 001-2017 ejecución (ME) | EEGNet | 20261114 | 0,142 | -0,001 |
| 001-2017 ejecución (ME) | FBCSP+SVM | 20261113 | 0,143 | 0,000 |
| 001-2017 ejecución (ME) | FBCSP+SVM | 20261114 | 0,143 | 0,000 |
| 001-2017 ejecución (ME) | MDM | 20261113 | 0,140 | -0,006 |
| 001-2017 ejecución (ME) | MDM | 20261114 | 0,140 | -0,007 |
| 001-2017 ejecución (ME) | ShallowConvNet | 20261113 | 0,140 | -0,004 |
| 001-2017 ejecución (ME) | ShallowConvNet | 20261114 | 0,140 | -0,003 |
| 001-2017 imaginación (MI) | CSP+LDA | 20261113 | 0,144 | 0,002 |
| 001-2017 imaginación (MI) | CSP+LDA | 20261114 | 0,133 | -0,014 |
| 001-2017 imaginación (MI) | DeepConvNet | 20261113 | 0,143 | 0,000 |
| 001-2017 imaginación (MI) | DeepConvNet | 20261114 | 0,142 | -0,011 |
| 001-2017 imaginación (MI) | EEGNet | 20261113 | 0,143 | 0,000 |
| 001-2017 imaginación (MI) | EEGNet | 20261114 | 0,142 | -0,001 |
| 001-2017 imaginación (MI) | FBCSP+SVM | 20261113 | 0,142 | -0,003 |
| 001-2017 imaginación (MI) | FBCSP+SVM | 20261114 | 0,141 | -0,004 |
| 001-2017 imaginación (MI) | MDM | 20261113 | 0,143 | 0,000 |
| 001-2017 imaginación (MI) | MDM | 20261114 | 0,143 | 0,000 |
| 001-2017 imaginación (MI) | ShallowConvNet | 20261113 | 0,140 | -0,005 |
| 001-2017 imaginación (MI) | ShallowConvNet | 20261114 | 0,149 | 0,008 |
| 001-2019 LME (offline) | CSP+LDA | 20261113 | 0,186 | -0,030 |
| 001-2019 LME (offline) | CSP+LDA | 20261114 | 0,201 | 0,001 |
| 001-2019 LME (offline) | DeepConvNet | 20261113 | 0,207 | 0,009 |
| 001-2019 LME (offline) | DeepConvNet | 20261114 | 0,197 | -0,005 |
| 001-2019 LME (offline) | EEGNet | 20261113 | 0,175 | -0,032 |
| 001-2019 LME (offline) | EEGNet | 20261114 | 0,192 | -0,012 |
| 001-2019 LME (offline) | FBCSP+SVM | 20261113 | 0,222 | 0,031 |
| 001-2019 LME (offline) | FBCSP+SVM | 20261114 | 0,184 | -0,022 |
| 001-2019 LME (offline) | MDM | 20261113 | 0,189 | -0,015 |
| 001-2019 LME (offline) | MDM | 20261114 | 0,204 | 0,005 |
| 001-2019 LME (offline) | ShallowConvNet | 20261113 | 0,175 | -0,032 |
| 001-2019 LME (offline) | ShallowConvNet | 20261114 | 0,191 | -0,012 |

**Tabla. Análisis separado de 001-2019 (lesión medular): Wilcoxon de rangos con signo (Holm) y prueba bayesiana de rangos con signo (ROPE = 0,01 en exactitud balanceada; 0,02 en MCC). Bloques = 10 particiones.**

| Métrica | Modelo A | Modelo B | Diferencia media (A − B) | p Wilcoxon | p Holm | P(A mejor) | P(ROPE) | P(B mejor) |
|---|---|---|---|---|---|---|---|---|
| Exactitud balanceada | CSP+LDA | FBCSP+SVM | 0,003 | 0,422 | 1,000 | 0,389 | 0,319 | 0,292 |
| Exactitud balanceada | CSP+LDA | MDM | -0,009 | 0,268 | 1,000 | 0,001 | 0,767 | 0,232 |
| Exactitud balanceada | CSP+LDA | EEGNet | 0,012 | 0,160 | 1,000 | 0,755 | 0,231 | 0,014 |
| Exactitud balanceada | CSP+LDA | ShallowConvNet | 0,007 | 0,496 | 1,000 | 0,259 | 0,740 | 0,001 |
| Exactitud balanceada | CSP+LDA | DeepConvNet | 0,010 | 0,322 | 1,000 | 0,595 | 0,371 | 0,033 |
| Exactitud balanceada | FBCSP+SVM | MDM | -0,012 | 0,016 | 0,219 | 0,000 | 0,371 | 0,629 |
| Exactitud balanceada | FBCSP+SVM | EEGNet | 0,009 | 0,492 | 1,000 | 0,676 | 0,245 | 0,079 |
| Exactitud balanceada | FBCSP+SVM | ShallowConvNet | 0,005 | 1,000 | 1,000 | 0,328 | 0,620 | 0,053 |
| Exactitud balanceada | FBCSP+SVM | DeepConvNet | 0,008 | 0,193 | 1,000 | 0,486 | 0,493 | 0,020 |
| Exactitud balanceada | MDM | EEGNet | 0,021 | 0,037 | 0,482 | 0,947 | 0,049 | 0,004 |
| Exactitud balanceada | MDM | ShallowConvNet | 0,016 | 0,064 | 0,773 | 0,835 | 0,165 | 0,001 |
| Exactitud balanceada | MDM | DeepConvNet | 0,019 | 0,014 | 0,205 | 0,954 | 0,046 | 0,000 |
| Exactitud balanceada | EEGNet | ShallowConvNet | -0,004 | 0,848 | 1,000 | 0,016 | 0,742 | 0,242 |
| Exactitud balanceada | EEGNet | DeepConvNet | -0,001 | 0,943 | 1,000 | 0,118 | 0,761 | 0,121 |
| Exactitud balanceada | ShallowConvNet | DeepConvNet | 0,003 | 0,557 | 1,000 | 0,361 | 0,528 | 0,111 |
| MCC | CSP+LDA | FBCSP+SVM | 0,003 | 0,422 | 1,000 | 0,068 | 0,885 | 0,047 |
| MCC | CSP+LDA | MDM | -0,013 | 0,182 | 1,000 | 0,000 | 0,819 | 0,181 |
| MCC | CSP+LDA | EEGNet | 0,015 | 0,193 | 1,000 | 0,362 | 0,637 | 0,001 |
| MCC | CSP+LDA | ShallowConvNet | 0,009 | 0,492 | 1,000 | 0,108 | 0,892 | 0,000 |
| MCC | CSP+LDA | DeepConvNet | 0,014 | 0,275 | 1,000 | 0,299 | 0,700 | 0,001 |
| MCC | FBCSP+SVM | MDM | -0,016 | 0,012 | 0,176 | 0,000 | 0,743 | 0,257 |
| MCC | FBCSP+SVM | EEGNet | 0,012 | 0,492 | 1,000 | 0,255 | 0,734 | 0,011 |
| MCC | FBCSP+SVM | ShallowConvNet | 0,006 | 1,000 | 1,000 | 0,124 | 0,871 | 0,005 |
| MCC | FBCSP+SVM | DeepConvNet | 0,010 | 0,193 | 1,000 | 0,068 | 0,931 | 0,001 |
| MCC | MDM | EEGNet | 0,028 | 0,037 | 0,482 | 0,739 | 0,261 | 0,000 |
| MCC | MDM | ShallowConvNet | 0,022 | 0,064 | 0,773 | 0,527 | 0,473 | 0,000 |
| MCC | MDM | DeepConvNet | 0,027 | 0,020 | 0,273 | 0,709 | 0,291 | 0,000 |
| MCC | EEGNet | ShallowConvNet | -0,006 | 0,734 | 1,000 | 0,001 | 0,972 | 0,026 |
| MCC | EEGNet | DeepConvNet | -0,001 | 0,922 | 1,000 | 0,017 | 0,950 | 0,033 |
| MCC | ShallowConvNet | DeepConvNet | 0,004 | 0,492 | 1,000 | 0,039 | 0,957 | 0,004 |

### Figuras (solo en español, 300 ppp; PNG, SVG y PDF)
- `figures/DC_exactitud_balanceada_bloques_conjuntos.*` y `figures/DC_MCC_bloques_conjuntos.*`: N = 3 conjuntos de evaluación (media de 10 particiones).
- `figures/DC_*_bloques_particion_conjunto.*`: N = 30 bloques, no independientes.
- `figures/DC_*_bloques_participante_conjunto.*`: N = 35 participantes que estuvieron en prueba al menos una vez.
- `figures/DC_*_solo_LME_2019.*`: análisis separado del conjunto de evaluación con lesión medular.
- `figures/Figura_cajas_exactitud_balanceada.*` y `figures/Figura_cajas_MCC.*`: distribución por partición.

## 4. Conclusiones

1. **001-2017 (ME y MI, 7 clases):** los seis modelos quedan en el nivel de azar en las 10 particiones.
   - La exactitud balanceada media va de 0,142 a 0,144 (azar: 0,143).
   - El MCC medio está entre −0,002 y 0,002.
   - Ninguna prueba frente al azar es significativa (p de Holm = 1).
   - Con bloques de solo sanos, Friedman da p ≈ 0,98.
2. **001-2019 (lesión medular, 5 clases):** solo el MDM supera al azar de forma consistente.
   - MDM: 0,220 ± 0,017 frente a 0,200; MCC 0,027 ± 0,023; Wilcoxon unilateral p = 0,005 sin corregir y p de Holm = 0,088.
   - CSP+LDA (0,211) y FBCSP+SVM (0,208) quedan por encima del azar, pero sin significancia.
   - Las redes, ya con ponderación por clase y programador coseno, quedan en el azar (0,199–0,204).
   - En el análisis separado, Friedman da p = 0,227 (exactitud balanceada) y 0,199 (MCC).
   - Por pares, el MDM supera a EEGNet y DeepConvNet con p sin corregir de 0,014–0,037 y P(MDM mejor) de 0,95 en exactitud balanceada, pero **ninguna comparación sobrevive a Holm** (p de Holm mínima = 0,205).
3. **Friedman y Nemenyi:**
   - Con bloques = conjuntos de evaluación (N = 3), los rangos favorecen a los clásicos: p asintótica = 0,063 y p por permutación = 0,026. Pero la DC = 4,35 es mayor que cualquier diferencia de rangos, así que **ningún par difiere según Nemenyi**.
   - Con N = 3 y dos conjuntos de evaluación en el nivel de azar, ese ordenamiento está dominado por ruido.
   - Con bloques alternativos, Friedman da p = 0,565 (N = 30) y 0,518 (N = 35).
   - La ventaja de rango de los clásicos en la partición única v3 (p = 0,053) no se sostiene como diferencia significativa con particiones repetidas.
4. **Wilcoxon + Holm (bloques partición × conjunto de evaluación):** ninguna diferencia significativa (p de Holm mínima = 0,238 en exactitud balanceada y 0,393 en MCC).
5. **Bayes con ROPE:** en los 15 pares, P(ROPE) ≥ 0,88 en exactitud balanceada (ROPE ± 0,01) y ≥ 0,99 en MCC (ROPE ± 0,02): las diferencias son prácticamente equivalentes. Esto se debe sobre todo a los 20 bloques de 001-2017, que están en el azar.
6. **Control por permutación:** con etiquetas permutadas, la exactitud balanceada queda en 0,133–0,149 en 001-2017 y en 0,175–0,222 en 001-2019, en el mismo rango que con etiquetas reales. No hay indicio de fuga de información.

## 5. Desviaciones respecto al protocolo de David y salvedades

- **001-2020 no se ejecutó** porque la fuente oficial no estaba disponible (§1). El análisis por conjuntos de evaluación tiene N = 3 en lugar de 6.
- **Entrenamiento de las redes:**
  - 001-2017 replica el código v3 ejecutado, que coincide con la configuración establecida (sin ponderación ni programador).
  - 001-2019 aplica ahora la configuración establecida (ponderación + coseno), que el código v3 no aplicaba (§1.1).
- **Particiones repetidas en 001-2019:** solo hay 45 particiones posibles. Las semillas 0 y 6 dan el mismo conjunto de prueba (P06, P10), igual que las semillas 2 y 8 (P02, P04): hay 8 conjuntos de prueba distintos. P07 (001-2019), S05 (001-2017 ME) y S03/S04 (001-2017 MI) nunca quedaron en prueba.
- **Independencia de los bloques:** los bloques partición × conjunto de evaluación y participante × conjunto de evaluación **no son independientes**, así que sus valores p son descriptivos. Los bloques = conjuntos de evaluación son solo 3, y ME y MI comparten el estudio 001-2017. Demšar (2006) recomienda 10 o más conjuntos independientes.
- **p por permutación de Friedman:** Monte Carlo, con 20 000 permutaciones para N = 3 y 5 000 en los demás casos.
- **Bayes:** prueba de rangos con signo no jerárquica (baycomp), con ROPE de 0,01 en exactitud balanceada y 0,02 en MCC.
- **Control por permutación:** 2 semillas, no 10.

## 6. Archivos

- `scripts/`:
  - Extracción (01), modelos clásicos (02), redes (03, con `--weighted_cosine`), análisis (04 y 04b), tablas (05) y figura de cajas (06).
  - `core.py` y `deep_core.py`.
  - `verificar_resultados.py`, `build_notebook.py` y `terminologia.py`.
- `TESIS_pipeline_reproducible.ipynb`: cuaderno reproducible, documentado y **sin ejecutar**.
- `preds/`: 216 predicciones por época (`.csv.gz`) más los JSON de metadatos.
- `tables/`: CSV de métricas, estadística y control; **`TABLAS_ES_listas_para_pegar.md`** y **`TABLAS_ES_estadistica.md`**.
- `figures/`: diagramas DC y figuras de cajas.
- `tables/anteriores_sin_ponderacion_ni_scheduler/`: resultados v1 completos, sin ponderación ni programador en 001-2019.
