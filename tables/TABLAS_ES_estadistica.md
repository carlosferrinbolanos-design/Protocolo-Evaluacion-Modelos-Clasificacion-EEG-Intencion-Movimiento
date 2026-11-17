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
