**Tabla. Prueba de Friedman y diferencia crítica de Nemenyi (k = 6 modelos, α = 0,05).**

| Métrica | Bloques | N | χ²(5) | p asintótica | p Iman-Davenport | p por permutación | DC | Rangos promedio (1 = mejor) |
|---|---|---|---|---|---|---|---|---|
| Exactitud balanceada | Estratos (media de particiones) | 3 | 10,48 | 0,063 | 0,019 | 0,026 | 4,35 | MDM riemanniano 1,67; CSP+LDA 2,17; FBCSP+SVM 3,17; ShallowConvNet 3,33; EEGNet 5,00; DeepConvNet 5,67 |
| Exactitud balanceada | Partición × estrato | 30 | 4,62 | 0,463 | 0,468 | 0,443 | 1,38 | MDM riemanniano 2,97; CSP+LDA 3,37; FBCSP+SVM 3,38; ShallowConvNet 3,65; DeepConvNet 3,80; EEGNet 3,83 |
| Exactitud balanceada | Participante × estrato | 35 | 3,62 | 0,605 | 0,611 | 0,546 | 1,27 | MDM riemanniano 3,11; CSP+LDA 3,30; ShallowConvNet 3,44; FBCSP+SVM 3,61; EEGNet 3,67; DeepConvNet 3,86 |
| Exactitud balanceada | Solo sanos (001-2017 ME/MI), partición × estrato | 20 | 0,69 | 0,983 | 0,985 | 0,979 | 1,69 | FBCSP+SVM 3,30; MDM riemanniano 3,38; CSP+LDA 3,42; EEGNet 3,60; ShallowConvNet 3,60; DeepConvNet 3,70 |
| Exactitud balanceada | Solo LME (001-2019), particiones | 10 | 8,11 | 0,150 | 0,144 | 0,149 | 2,38 | MDM riemanniano 2,15; CSP+LDA 3,25; FBCSP+SVM 3,55; ShallowConvNet 3,75; DeepConvNet 4,00; EEGNet 4,30 |
| MCC | Estratos (media de particiones) | 3 | 10,81 | 0,055 | 0,013 | 0,023 | 4,35 | MDM riemanniano 1,33; CSP+LDA 2,00; FBCSP+SVM 3,67; ShallowConvNet 3,67; EEGNet 5,00; DeepConvNet 5,33 |
| MCC | Partición × estrato | 30 | 4,93 | 0,425 | 0,429 | 0,410 | 1,38 | MDM riemanniano 2,90; CSP+LDA 3,33; FBCSP+SVM 3,55; ShallowConvNet 3,68; DeepConvNet 3,70; EEGNet 3,83 |
| MCC | Participante × estrato | 35 | 3,27 | 0,658 | 0,663 | 0,612 | 1,27 | MDM riemanniano 3,11; CSP+LDA 3,31; EEGNet 3,53; ShallowConvNet 3,53; FBCSP+SVM 3,69; DeepConvNet 3,83 |
| MCC | Solo sanos (001-2017 ME/MI), partición × estrato | 20 | 0,45 | 0,994 | 0,994 | 0,995 | 1,69 | MDM riemanniano 3,30; CSP+LDA 3,40; FBCSP+SVM 3,52; DeepConvNet 3,55; EEGNet 3,60; ShallowConvNet 3,62 |
| MCC | Solo LME (001-2019), particiones | 10 | 8,69 | 0,122 | 0,115 | 0,116 | 2,38 | MDM riemanniano 2,10; CSP+LDA 3,20; FBCSP+SVM 3,60; ShallowConvNet 3,80; DeepConvNet 4,00; EEGNet 4,30 |

**Tabla. Comparaciones por pares: Wilcoxon de rangos con signo (corrección de Holm) y prueba bayesiana de rangos con signo (ROPE = 0,01 en exactitud balanceada; 0,02 en MCC). Bloques = partición × estrato.**

| Métrica | Modelo A | Modelo B | Diferencia media (A − B) | p Wilcoxon | p Holm | P(A mejor) | P(ROPE) | P(B mejor) |
|---|---|---|---|---|---|---|---|---|
| Exactitud balanceada | CSP+LDA | FBCSP+SVM | 0,001 | 0,301 | 1,000 | 0,000 | 0,999 | 0,001 |
| Exactitud balanceada | CSP+LDA | MDM riemanniano | -0,003 | 0,346 | 1,000 | 0,000 | 0,999 | 0,001 |
| Exactitud balanceada | CSP+LDA | EEGNet | 0,005 | 0,106 | 1,000 | 0,006 | 0,994 | 0,000 |
| Exactitud balanceada | CSP+LDA | ShallowConvNet | 0,003 | 0,471 | 1,000 | 0,003 | 0,997 | 0,000 |
| Exactitud balanceada | CSP+LDA | DeepConvNet | 0,005 | 0,301 | 1,000 | 0,004 | 0,996 | 0,000 |
| Exactitud balanceada | FBCSP+SVM | MDM riemanniano | -0,004 | 0,044 | 0,570 | 0,000 | 0,998 | 0,002 |
| Exactitud balanceada | FBCSP+SVM | EEGNet | 0,004 | 0,471 | 1,000 | 0,007 | 0,993 | 0,000 |
| Exactitud balanceada | FBCSP+SVM | ShallowConvNet | 0,002 | 0,509 | 1,000 | 0,001 | 0,999 | 0,000 |
| Exactitud balanceada | FBCSP+SVM | DeepConvNet | 0,004 | 0,103 | 1,000 | 0,010 | 0,990 | 0,000 |
| Exactitud balanceada | MDM riemanniano | EEGNet | 0,008 | 0,031 | 0,438 | 0,128 | 0,872 | 0,000 |
| Exactitud balanceada | MDM riemanniano | ShallowConvNet | 0,006 | 0,084 | 1,000 | 0,054 | 0,946 | 0,000 |
| Exactitud balanceada | MDM riemanniano | DeepConvNet | 0,008 | 0,025 | 0,377 | 0,093 | 0,907 | 0,000 |
| Exactitud balanceada | EEGNet | ShallowConvNet | -0,002 | 0,440 | 1,000 | 0,000 | 1,000 | 0,000 |
| Exactitud balanceada | EEGNet | DeepConvNet | 0,000 | 0,603 | 1,000 | 0,000 | 1,000 | 0,000 |
| Exactitud balanceada | ShallowConvNet | DeepConvNet | 0,002 | 0,462 | 1,000 | 0,000 | 1,000 | 0,000 |
| MCC | CSP+LDA | FBCSP+SVM | 0,002 | 0,387 | 1,000 | 0,000 | 1,000 | 0,000 |
| MCC | CSP+LDA | MDM riemanniano | -0,005 | 0,358 | 1,000 | 0,000 | 0,999 | 0,001 |
| MCC | CSP+LDA | EEGNet | 0,007 | 0,070 | 0,769 | 0,001 | 0,999 | 0,000 |
| MCC | CSP+LDA | ShallowConvNet | 0,005 | 0,325 | 1,000 | 0,000 | 1,000 | 0,000 |
| MCC | CSP+LDA | DeepConvNet | 0,006 | 0,362 | 1,000 | 0,001 | 0,999 | 0,000 |
| MCC | FBCSP+SVM | MDM riemanniano | -0,006 | 0,032 | 0,446 | 0,000 | 1,000 | 0,000 |
| MCC | FBCSP+SVM | EEGNet | 0,006 | 0,361 | 1,000 | 0,000 | 1,000 | 0,000 |
| MCC | FBCSP+SVM | ShallowConvNet | 0,003 | 0,538 | 1,000 | 0,000 | 1,000 | 0,000 |
| MCC | FBCSP+SVM | DeepConvNet | 0,005 | 0,144 | 1,000 | 0,000 | 1,000 | 0,000 |
| MCC | MDM riemanniano | EEGNet | 0,012 | 0,012 | 0,176 | 0,028 | 0,972 | 0,000 |
| MCC | MDM riemanniano | ShallowConvNet | 0,009 | 0,064 | 0,763 | 0,001 | 0,999 | 0,000 |
| MCC | MDM riemanniano | DeepConvNet | 0,011 | 0,040 | 0,526 | 0,022 | 0,978 | 0,000 |
| MCC | EEGNet | ShallowConvNet | -0,003 | 0,299 | 1,000 | 0,000 | 1,000 | 0,000 |
| MCC | EEGNet | DeepConvNet | -0,001 | 0,871 | 1,000 | 0,000 | 1,000 | 0,000 |
| MCC | ShallowConvNet | DeepConvNet | 0,002 | 0,482 | 1,000 | 0,000 | 1,000 | 0,000 |

**Tabla. Exactitud balanceada frente al azar por estrato y modelo (Wilcoxon unilateral sobre particiones, corrección de Holm).**

| Estrato | Modelo | Particiones | Media | Azar | Media − azar | p | p Holm |
|---|---|---|---|---|---|---|---|
| 001-2017 ejecución (ME) | CSP+LDA | 10 | 0,144 | 0,143 | 0,001 | 0,156 | 1,000 |
| 001-2017 ejecución (ME) | DeepConvNet | 10 | 0,142 | 0,143 | 0,000 | 0,764 | 1,000 |
| 001-2017 ejecución (ME) | EEGNet | 10 | 0,144 | 0,143 | 0,001 | 0,238 | 1,000 |
| 001-2017 ejecución (ME) | FBCSP+SVM | 10 | 0,144 | 0,143 | 0,001 | 0,359 | 1,000 |
| 001-2017 ejecución (ME) | MDM riemanniano | 10 | 0,144 | 0,143 | 0,001 | 0,305 | 1,000 |
| 001-2017 ejecución (ME) | ShallowConvNet | 10 | 0,143 | 0,143 | 0,000 | 0,699 | 1,000 |
| 001-2017 imaginación (MI) | CSP+LDA | 10 | 0,142 | 0,143 | 0,000 | 0,648 | 1,000 |
| 001-2017 imaginación (MI) | DeepConvNet | 10 | 0,142 | 0,143 | -0,001 | 0,859 | 1,000 |
| 001-2017 imaginación (MI) | EEGNet | 10 | 0,142 | 0,143 | -0,001 | 0,895 | 1,000 |
| 001-2017 imaginación (MI) | FBCSP+SVM | 10 | 0,142 | 0,143 | 0,000 | 0,719 | 1,000 |
| 001-2017 imaginación (MI) | MDM riemanniano | 10 | 0,143 | 0,143 | 0,000 | 0,375 | 1,000 |
| 001-2017 imaginación (MI) | ShallowConvNet | 10 | 0,144 | 0,143 | 0,001 | 0,336 | 1,000 |
| 001-2019 LME (offline) | CSP+LDA | 10 | 0,211 | 0,200 | 0,011 | 0,092 | 1,000 |
| 001-2019 LME (offline) | DeepConvNet | 10 | 0,199 | 0,200 | -0,001 | 0,626 | 1,000 |
| 001-2019 LME (offline) | EEGNet | 10 | 0,196 | 0,200 | -0,004 | 0,820 | 1,000 |
| 001-2019 LME (offline) | FBCSP+SVM | 10 | 0,208 | 0,200 | 0,008 | 0,153 | 1,000 |
| 001-2019 LME (offline) | MDM riemanniano | 10 | 0,220 | 0,200 | 0,020 | 0,005 | 0,088 |
| 001-2019 LME (offline) | ShallowConvNet | 10 | 0,202 | 0,200 | 0,002 | 0,340 | 1,000 |

**Tabla. Control con etiquetas de entrenamiento permutadas dentro de cada participante (exactitud balanceada en prueba).**

| Estrato | Modelo | Partición (semilla) | Exactitud balanceada | MCC |
|---|---|---|---|---|
| 001-2017 ejecución (ME) | CSP+LDA | 20261113 | 0,145 | 0,003 |
| 001-2017 ejecución (ME) | CSP+LDA | 20261114 | 0,148 | 0,009 |
| 001-2017 ejecución (ME) | DeepConvNet | 20261113 | 0,144 | 0,001 |
| 001-2017 ejecución (ME) | DeepConvNet | 20261114 | 0,143 | 0,000 |
| 001-2017 ejecución (ME) | EEGNet | 20261113 | 0,143 | 0,000 |
| 001-2017 ejecución (ME) | EEGNet | 20261114 | 0,142 | -0,001 |
| 001-2017 ejecución (ME) | FBCSP+SVM | 20261113 | 0,143 | 0,000 |
| 001-2017 ejecución (ME) | FBCSP+SVM | 20261114 | 0,143 | 0,000 |
| 001-2017 ejecución (ME) | MDM riemanniano | 20261113 | 0,140 | -0,006 |
| 001-2017 ejecución (ME) | MDM riemanniano | 20261114 | 0,140 | -0,007 |
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
| 001-2017 imaginación (MI) | MDM riemanniano | 20261113 | 0,143 | 0,000 |
| 001-2017 imaginación (MI) | MDM riemanniano | 20261114 | 0,143 | 0,000 |
| 001-2017 imaginación (MI) | ShallowConvNet | 20261113 | 0,140 | -0,005 |
| 001-2017 imaginación (MI) | ShallowConvNet | 20261114 | 0,149 | 0,008 |
| 001-2019 LME (offline) | CSP+LDA | 20261113 | 0,186 | -0,030 |
| 001-2019 LME (offline) | CSP+LDA | 20261114 | 0,201 | 0,001 |
| 001-2019 LME (offline) | DeepConvNet | 20261113 | 0,181 | -0,025 |
| 001-2019 LME (offline) | DeepConvNet | 20261114 | 0,192 | -0,014 |
| 001-2019 LME (offline) | EEGNet | 20261113 | 0,200 | 0,000 |
| 001-2019 LME (offline) | EEGNet | 20261114 | 0,197 | -0,005 |
| 001-2019 LME (offline) | FBCSP+SVM | 20261113 | 0,222 | 0,031 |
| 001-2019 LME (offline) | FBCSP+SVM | 20261114 | 0,184 | -0,022 |
| 001-2019 LME (offline) | MDM riemanniano | 20261113 | 0,189 | -0,015 |
| 001-2019 LME (offline) | MDM riemanniano | 20261114 | 0,204 | 0,005 |
| 001-2019 LME (offline) | ShallowConvNet | 20261113 | 0,204 | 0,005 |
| 001-2019 LME (offline) | ShallowConvNet | 20261114 | 0,212 | 0,017 |

**Tabla. Análisis separado de 001-2019 (lesión medular): Wilcoxon de rangos con signo (Holm) y prueba bayesiana de rangos con signo (ROPE = 0,01 en exactitud balanceada; 0,02 en MCC). Bloques = 10 particiones.**

| Métrica | Modelo A | Modelo B | Diferencia media (A − B) | p Wilcoxon | p Holm | P(A mejor) | P(ROPE) | P(B mejor) |
|---|---|---|---|---|---|---|---|---|
| Exactitud balanceada | CSP+LDA | FBCSP+SVM | 0,003 | 0,422 | 1,000 | 0,388 | 0,321 | 0,291 |
| Exactitud balanceada | CSP+LDA | MDM riemanniano | -0,009 | 0,268 | 1,000 | 0,001 | 0,768 | 0,231 |
| Exactitud balanceada | CSP+LDA | EEGNet | 0,015 | 0,037 | 0,445 | 0,755 | 0,245 | 0,000 |
| Exactitud balanceada | CSP+LDA | ShallowConvNet | 0,009 | 0,301 | 1,000 | 0,447 | 0,552 | 0,001 |
| Exactitud balanceada | CSP+LDA | DeepConvNet | 0,012 | 0,375 | 1,000 | 0,523 | 0,471 | 0,007 |
| Exactitud balanceada | FBCSP+SVM | MDM riemanniano | -0,012 | 0,016 | 0,219 | 0,000 | 0,373 | 0,627 |
| Exactitud balanceada | FBCSP+SVM | EEGNet | 0,012 | 0,264 | 1,000 | 0,547 | 0,453 | 0,000 |
| Exactitud balanceada | FBCSP+SVM | ShallowConvNet | 0,006 | 0,375 | 1,000 | 0,333 | 0,655 | 0,012 |
| Exactitud balanceada | FBCSP+SVM | DeepConvNet | 0,009 | 0,275 | 1,000 | 0,561 | 0,417 | 0,022 |
| Exactitud balanceada | MDM riemanniano | EEGNet | 0,024 | 0,014 | 0,205 | 0,973 | 0,027 | 0,000 |
| Exactitud balanceada | MDM riemanniano | ShallowConvNet | 0,018 | 0,020 | 0,254 | 0,958 | 0,042 | 0,000 |
| Exactitud balanceada | MDM riemanniano | DeepConvNet | 0,021 | 0,037 | 0,445 | 0,951 | 0,049 | 0,000 |
| Exactitud balanceada | EEGNet | ShallowConvNet | -0,006 | 0,160 | 1,000 | 0,000 | 0,871 | 0,129 |
| Exactitud balanceada | EEGNet | DeepConvNet | -0,003 | 0,846 | 1,000 | 0,000 | 0,934 | 0,066 |
| Exactitud balanceada | ShallowConvNet | DeepConvNet | 0,003 | 0,539 | 1,000 | 0,009 | 0,991 | 0,000 |
| MCC | CSP+LDA | FBCSP+SVM | 0,003 | 0,422 | 1,000 | 0,069 | 0,883 | 0,048 |
| MCC | CSP+LDA | MDM riemanniano | -0,013 | 0,182 | 1,000 | 0,000 | 0,815 | 0,185 |
| MCC | CSP+LDA | EEGNet | 0,019 | 0,037 | 0,445 | 0,309 | 0,691 | 0,000 |
| MCC | CSP+LDA | ShallowConvNet | 0,012 | 0,275 | 1,000 | 0,167 | 0,833 | 0,000 |
| MCC | CSP+LDA | DeepConvNet | 0,015 | 0,375 | 1,000 | 0,288 | 0,712 | 0,000 |
| MCC | FBCSP+SVM | MDM riemanniano | -0,016 | 0,012 | 0,176 | 0,000 | 0,741 | 0,259 |
| MCC | FBCSP+SVM | EEGNet | 0,016 | 0,275 | 1,000 | 0,185 | 0,815 | 0,000 |
| MCC | FBCSP+SVM | ShallowConvNet | 0,009 | 0,322 | 1,000 | 0,078 | 0,922 | 0,001 |
| MCC | FBCSP+SVM | DeepConvNet | 0,011 | 0,322 | 1,000 | 0,257 | 0,738 | 0,006 |
| MCC | MDM riemanniano | EEGNet | 0,032 | 0,014 | 0,191 | 0,857 | 0,143 | 0,000 |
| MCC | MDM riemanniano | ShallowConvNet | 0,025 | 0,014 | 0,191 | 0,713 | 0,287 | 0,000 |
| MCC | MDM riemanniano | DeepConvNet | 0,028 | 0,037 | 0,445 | 0,818 | 0,182 | 0,000 |
| MCC | EEGNet | ShallowConvNet | -0,007 | 0,160 | 1,000 | 0,000 | 0,997 | 0,003 |
| MCC | EEGNet | DeepConvNet | -0,005 | 0,846 | 1,000 | 0,000 | 0,994 | 0,006 |
| MCC | ShallowConvNet | DeepConvNet | 0,002 | 0,695 | 1,000 | 0,000 | 1,000 | 0,000 |
