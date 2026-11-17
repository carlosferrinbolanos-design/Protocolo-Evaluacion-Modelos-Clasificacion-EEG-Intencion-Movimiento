"""Tablas listas para pegar en el artículo (español, coma decimal) y tablas estadísticas desde stats.json."""
import pandas as pd, json, numpy as np
import os
from pathlib import Path
R=os.environ.get('RESULTADOS',str(Path(__file__).resolve().parents[1])); T=f'{R}/tables'
A=pd.read_csv(f'{T}/summary_mean_sd.csv'); A=A[A.control=='true_labels']
MODELS=['CSP_LDA','FBCSP_SVM','Riemannian_MDM','EEGNet','ShallowConvNet','DeepConvNet']
ES={'CSP_LDA':'CSP+LDA','FBCSP_SVM':'FBCSP+SVM','Riemannian_MDM':'MDM','EEGNet':'EEGNet','ShallowConvNet':'ShallowConvNet','DeepConvNet':'DeepConvNet'}
STR_ES={'001-2017_ME':'001-2017 ejecución (ME)','001-2017_MI':'001-2017 imaginación (MI)','001-2019_offline':'001-2019 LME (offline)','001-2020_gel':'001-2020 gel','001-2020_water':'001-2020 agua','001-2020_dry':'001-2020 seco'}
CH={'001-2017_ME':1/7,'001-2017_MI':1/7,'001-2019_offline':.2,'001-2020_gel':.5,'001-2020_water':.5,'001-2020_dry':.5}
MET={'balanced_accuracy':('Exactitud balanceada','Balanced accuracy'),'macro_f1':('F1 macro','Macro-F1'),'cohen_kappa':('Kappa de Cohen','Cohen\'s kappa'),'mcc':('MCC','MCC')}
def f(x,es=True): x=0.0 if abs(x)<0.0005 else x; s=f'{x:.3f}'; return s.replace('.',',') if es else s
def f2(x): return f'{x:.2f}'.replace('.',',')
out_es=[]; out_en=[]
for m,(es,en) in MET.items():
    strata=[s for s in STR_ES if s in set(A.stratum)]
    n=int(A[f'{m}_count'].min()) if len(A) else 0
    he='| Conjunto de evaluación | Azar | '+' | '.join(ES[x] for x in MODELS)+' |'; hn='| Stratum | chance | '+' | '.join(MODELS)+' |'
    sep='|'+'---|'*(len(MODELS)+2)
    le=[f'**Tabla. {es} en prueba (media ± DE sobre particiones 80/20 disjuntas por participante; semillas 20261113 + k, k = 0…9).**','',he,sep]
    ln=[f'**{en} on held-out participants (mean ± SD across repeated participant-disjoint splits).**','',hn,sep]
    for s in strata:
        ch=CH[s] if m=='balanced_accuracy' else (0.0 if m in('cohen_kappa','mcc') else np.nan)
        re_=[];rn=[]
        for x in MODELS:
            r=A[(A.stratum==s)&(A.model==x)]
            if len(r)==0: re_.append('n/d'); rn.append('n/a'); continue
            r=r.iloc[0]; k=int(r[f'{m}_count'])
            re_.append(f"{f(r[f'{m}_mean'])} ± {f(r[f'{m}_std'])}"+(f' (n={k})' if k<10 else '')); rn.append(f"{r[f'{m}_mean']:.3f} ± {r[f'{m}_std']:.3f}"+(f' (n={k})' if k<10 else ''))
        chs='—' if np.isnan(ch) else f(ch); chn='—' if np.isnan(ch) else f'{ch:.3f}'
        le.append(f'| {STR_ES[s]} | {chs} | '+' | '.join(re_)+' |'); ln.append(f'| {s} | {chn} | '+' | '.join(rn)+' |')
    out_es+=le+['']; out_en+=ln+['']
open(f'{T}/TABLAS_ES_listas_para_pegar.md','w').write('\n'.join(out_es))
# per-split wide csv for Word import
S=pd.read_csv(f'{T}/metrics_per_split.csv'); S[S.control=='true_labels'].to_csv(f'{T}/metrics_per_split_real_labels.csv',index=False)
print('\n'.join(out_es[:12]))
# ---- tablas estadísticas en español
st=json.load(open(f'{T}/stats.json'))['stats']
BL={'blocks_strata':'Conjuntos de evaluación (media de particiones)','blocks_split_x_stratum':'Partición × conjunto de evaluación','blocks_participant_x_stratum':'Participante × conjunto de evaluación','healthy_only_blocks_split':'Solo sanos (001-2017 ME/MI), partición × conjunto de evaluación','SCI_2019_only_blocks_split':'Solo LME (001-2019), particiones'}
MN={'balanced_accuracy':'Exactitud balanceada','mcc':'MCC'}
o=['**Tabla. Prueba de Friedman y diferencia crítica de Nemenyi (k = 6 modelos, α = 0,05).**','','| Métrica | Bloques | N | χ²(5) | p asintótica | p Iman-Davenport | p por permutación | DC | Rangos promedio (1 = mejor) |','|---|---|---|---|---|---|---|---|---|']
for m,res in st.items():
    for b,v in res.items():
        if b=='pairwise': continue
        rk='; '.join(f"{ES[x]} {f2(v['avg_ranks'][x])}" for x in sorted(v['avg_ranks'],key=v['avg_ranks'].get))
        pid='—' if v['p_iman_davenport'] is None or (isinstance(v['p_iman_davenport'],float) and np.isnan(v['p_iman_davenport'])) else f(v['p_iman_davenport'])
        o.append(f"| {MN[m]} | {BL.get(b,b)} | {v['N']} | {f2(v['chi2'])} | {f(v['p_asymptotic'])} | {pid} | {f(v['p_permutation'])} | {f2(v['CD'])} | {rk} |")
o+=['','**Tabla. Comparaciones por pares: Wilcoxon de rangos con signo (corrección de Holm) y prueba bayesiana de rangos con signo (ROPE = 0,01 en exactitud balanceada; 0,02 en MCC). Bloques = partición × conjunto de evaluación.**','','| Métrica | Modelo A | Modelo B | Diferencia media (A − B) | p Wilcoxon | p Holm | P(A mejor) | P(ROPE) | P(B mejor) |','|---|---|---|---|---|---|---|---|---|']
for m in st:
    pw=pd.read_csv(f'{T}/pairwise_wilcoxon_holm_bayes_{m}.csv')
    for _,r in pw.iterrows():
        o.append(f"| {MN[m]} | {ES[r.model_a]} | {ES[r.model_b]} | {f(r.mean_diff)} | {f(r.wilcoxon_p)} | {f(r.wilcoxon_p_holm)} | {f(r.p_left_a_better)} | {f(r.p_rope)} | {f(r.p_right_b_better)} |")
vc=pd.read_csv(f'{T}/vs_chance.csv')
o+=['','**Tabla. Exactitud balanceada frente al azar por conjunto de evaluación y modelo (Wilcoxon unilateral sobre particiones, corrección de Holm).**','','| Conjunto de evaluación | Modelo | Particiones | Media | Azar | Media − azar | p | p Holm |','|---|---|---|---|---|---|---|---|']
for _,r in vc.iterrows():
    o.append(f"| {STR_ES[r.stratum]} | {ES[r.model]} | {r.n_splits} | {f(r.mean_ba)} | {f(r.chance)} | {f(r.mean_minus_chance)} | {'—' if pd.isna(r.wilcoxon_greater_p) else f(r.wilcoxon_greater_p)} | {f(r.p_holm)} |")
pc=pd.read_csv(f'{T}/permutation_control.csv')
if len(pc):
    o+=['','**Tabla. Control con etiquetas de entrenamiento permutadas dentro de cada participante (exactitud balanceada en prueba).**','','| Conjunto de evaluación | Modelo | Partición (semilla) | Exactitud balanceada | MCC |','|---|---|---|---|---|']
    for _,r in pc.sort_values(['stratum','model','seed_idx']).iterrows():
        o.append(f"| {STR_ES[r.stratum]} | {ES[r.model]} | {20261113+int(r.seed_idx)} | {f(r.balanced_accuracy)} | {f(r.mcc)} |")
open(f'{T}/TABLAS_ES_estadistica.md','w').write('\n'.join(o)); print('stats tables ok')
import os
if os.path.exists(f'{T}/pairwise_LME_2019_wilcoxon_holm_bayes.csv'):
    pw=pd.read_csv(f'{T}/pairwise_LME_2019_wilcoxon_holm_bayes.csv')
    o=['**Tabla. Análisis separado de 001-2019 (lesión medular): Wilcoxon de rangos con signo (Holm) y prueba bayesiana de rangos con signo (ROPE = 0,01 en exactitud balanceada; 0,02 en MCC). Bloques = 10 particiones.**','','| Métrica | Modelo A | Modelo B | Diferencia media (A − B) | p Wilcoxon | p Holm | P(A mejor) | P(ROPE) | P(B mejor) |','|---|---|---|---|---|---|---|---|---|']
    for _,r in pw.iterrows():
        o.append(f"| {MN[r.metric]} | {ES[r.model_a]} | {ES[r.model_b]} | {f(r.mean_diff)} | {f(r.wilcoxon_p)} | {f(r.wilcoxon_p_holm)} | {f(r.p_left_a_better)} | {f(r.p_rope)} | {f(r.p_right_b_better)} |")
    open(f'{T}/TABLAS_ES_estadistica.md','a').write('\n\n'+'\n'.join(o)+'\n')
