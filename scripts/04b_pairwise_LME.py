"""Análisis separado del conjunto de evaluación con lesión medular (001-2019): Wilcoxon+Holm y bayesiano por pares, bloques = particiones."""
import pandas as pd, numpy as np, itertools, baycomp
from scipy import stats
import os
from pathlib import Path
T=os.path.join(os.environ.get('RESULTADOS',str(Path(__file__).resolve().parents[1])),'tables')
S=pd.read_csv(f'{T}/metrics_per_split.csv'); S=S[(S.control=='true_labels')&(S.stratum=='001-2019_offline')]
MODELS=['CSP_LDA','FBCSP_SVM','Riemannian_MDM','EEGNet','ShallowConvNet','DeepConvNet']
out=[]
for metric,rope in [('balanced_accuracy',0.01),('mcc',0.02)]:
    w=S.pivot(index='seed_idx',columns='model',values=metric)[MODELS]; rows=[]
    for a,b in itertools.combinations(MODELS,2):
        x=w[a].values;y=w[b].values; p=stats.wilcoxon(x,y).pvalue
        pl,pe,pr=baycomp.two_on_multiple(x,y,rope=rope,random_state=20261113)  # semilla fija del muestreo bayesiano
        rows.append(dict(metric=metric,model_a=a,model_b=b,mean_diff=float(np.mean(x-y)),wilcoxon_p=p,p_left_a_better=pl,p_rope=pe,p_right_b_better=pr))
    r=pd.DataFrame(rows); ps=r.wilcoxon_p.values; o=np.argsort(ps); adj=np.empty(len(ps)); run=0
    for i,idx in enumerate(o): run=max(run,(len(ps)-i)*ps[idx]); adj[idx]=min(1,run)
    r['wilcoxon_p_holm']=adj; out.append(r)
pd.concat(out).to_csv(f'{T}/pairwise_LME_2019_wilcoxon_holm_bayes.csv',index=False); print(pd.concat(out).round(3).to_string())
