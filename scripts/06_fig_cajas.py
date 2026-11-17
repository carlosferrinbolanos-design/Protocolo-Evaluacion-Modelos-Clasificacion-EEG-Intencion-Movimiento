"""Figura: distribución de la exactitud balanceada y el MCC por conjunto de evaluación y modelo sobre las 10 particiones (solo español)."""
import pandas as pd, numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams["font.family"]="DejaVu Sans"
import os
from pathlib import Path
R=os.environ.get('RESULTADOS',str(Path(__file__).resolve().parents[1]))
S=pd.read_csv(f'{R}/tables/metrics_per_split.csv'); S=S[S.control=='true_labels']
MODELS=['CSP_LDA','FBCSP_SVM','Riemannian_MDM','EEGNet','ShallowConvNet','DeepConvNet']
LBL=['CSP+\nLDA','FBCSP+\nSVM','MDM','EEGNet','Shallow-\nConvNet','Deep-\nConvNet']
ST=[('001-2017_ME','001-2017 ejecución (7 clases)',1/7),('001-2017_MI','001-2017 imaginación (7 clases)',1/7),('001-2019_offline','001-2019 lesión medular (5 clases)',.2)]
for met,ylab,fn in [('balanced_accuracy','Exactitud balanceada','Figura_cajas_exactitud_balanceada'),('mcc','MCC','Figura_cajas_MCC')]:
    fig,axs=plt.subplots(1,3,figsize=(7.2,2.6),dpi=300)
    for ax,(s,t,ch) in zip(axs,ST):
        d=[S[(S.stratum==s)&(S.model==m)][met].values for m in MODELS]
        ax.boxplot(d,widths=.6,medianprops=dict(color='#b00000'),flierprops=dict(markersize=2))
        for i,v in enumerate(d): ax.scatter(np.full(len(v),i+1)+np.random.default_rng(i).uniform(-.15,.15,len(v)),v,s=3,color='k',alpha=.5,zorder=3)
        ax.axhline(ch if met=='balanced_accuracy' else 0,ls='--',color='gray',lw=.8)
        ax.set_title(t,fontsize=7); ax.set_xticks(range(1,7)); ax.set_xticklabels(LBL,fontsize=5.2,rotation=0); ax.tick_params(axis='y',labelsize=6)
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x,p: f'{x:.3f}'.replace('.',',')))
    axs[0].set_ylabel(ylab,fontsize=7)
    fig.text(.5,.005,'Cada punto es una partición 80/20 disjunta por participante (10 semillas); línea discontinua = nivel de azar' if met=='balanced_accuracy' else 'Cada punto es una partición 80/20 disjunta por participante (10 semillas); línea discontinua = MCC de 0 (azar)',ha='center',fontsize=6,style='italic')
    plt.tight_layout(rect=(0,.04,1,1))
    for e in ('png','svg','pdf'): fig.savefig(f'{R}/figures/{fn}.{e}',dpi=300)
print('ok')
