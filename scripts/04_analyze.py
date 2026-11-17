"""Metrics + statistics + CD figures from per-epoch prediction files."""
import glob, os, json, itertools, warnings
import numpy as np, pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score, cohen_kappa_score, matthews_corrcoef, confusion_matrix
from scipy import stats
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')
from pathlib import Path
R=os.environ.get('RESULTADOS',str(Path(__file__).resolve().parents[1])); P=f'{R}/preds'; T=f'{R}/tables'; FG=f'{R}/figures'; os.makedirs(T,exist_ok=True); os.makedirs(FG,exist_ok=True)
MODELS=['CSP_LDA','FBCSP_SVM','Riemannian_MDM','EEGNet','ShallowConvNet','DeepConvNet']
LBL={'CSP_LDA':'CSP+LDA','FBCSP_SVM':'FBCSP+SVM','Riemannian_MDM':'MDM','EEGNet':'EEGNet','ShallowConvNet':'ShallowConvNet','DeepConvNet':'DeepConvNet'}
CHANCE={'001-2017_ME':1/7,'001-2017_MI':1/7,'001-2019_offline':1/5,'001-2020_gel':.5,'001-2020_water':.5,'001-2020_dry':.5}
def met(d):
    yt,yp=d.y_true,d.y_pred; labs=sorted(set(yt))
    return dict(balanced_accuracy=balanced_accuracy_score(yt,yp),macro_f1=f1_score(yt,yp,average='macro',labels=labs,zero_division=0),
                cohen_kappa=cohen_kappa_score(yt,yp),mcc=matthews_corrcoef(yt,yp),n=len(d))
rows=[];prow=[];cms=[]
for f in sorted(glob.glob(f'{P}/*.csv.gz')):
    d=pd.read_csv(f); k=d.iloc[0]; key=dict(stratum=k.stratum,model=k.model,control=k.control,seed_idx=int(k.seed_idx),seed=int(k.seed))
    js=json.load(open(f.replace('.csv.gz','.json')))
    rows.append({**key,**met(d),'test_subjects':' '.join(js['test_subjects']),'runtime_s':js['runtime_s']})
    for sub,g in d.groupby('subject'): prow.append({**key,'subject':sub,**met(g)})
    labs=sorted(set(d.y_true)); cm=confusion_matrix(d.y_true,d.y_pred,labels=labs)
    for i,a in enumerate(labs):
        for j,b in enumerate(labs): cms.append({**key,'true':a,'pred':b,'count':int(cm[i,j])})
S=pd.DataFrame(rows); S['chance']=S.stratum.map(CHANCE); S.to_csv(f'{T}/metrics_per_split.csv',index=False)
PP=pd.DataFrame(prow); PP.to_csv(f'{T}/metrics_per_participant_split.csv',index=False)
pd.DataFrame(cms).to_csv(f'{T}/confusion_matrices_per_split.csv',index=False)
M=['balanced_accuracy','macro_f1','cohen_kappa','mcc']
agg=S.groupby(['control','stratum','model'])[M].agg(['mean','std','count']); agg.columns=['_'.join(c) for c in agg.columns]; agg=agg.reset_index(); agg.to_csv(f'{T}/summary_mean_sd.csv',index=False)
real=S[S.control=='true_labels']
# completeness: only use (stratum,seed) blocks where all 6 models exist
cnt=real.groupby(['stratum','seed_idx']).model.nunique(); full=cnt[cnt==6].index
RB=real.set_index(['stratum','seed_idx']).loc[full].reset_index()
strata=sorted(RB.stratum.unique())
out={'strata_used':strata,'n_split_blocks':int(len(full)),'blocks_per_stratum':RB.groupby('stratum').seed_idx.nunique().to_dict()}
def ranks(mat):  # mat: blocks x models, higher better -> rank 1 best
    return np.vstack([stats.rankdata(-r) for r in mat])
def friedman(mat, nperm=20000, rng=np.random.default_rng(20261113)):
    N,k=mat.shape; Rk=ranks(mat)
    def chi(Rr): Rj=Rr.mean(0); return 12*N/(k*(k+1))*np.sum((Rj-(k+1)/2)**2)
    c=chi(Rk); p_asym=1-stats.chi2.cdf(c,k-1)
    ff=(N-1)*c/(N*(k-1)-c) if N*(k-1)-c>0 else np.inf; p_ff=1-stats.f.cdf(ff,k-1,(k-1)*(N-1))
    ge=0
    for _ in range(nperm):
        Rp=np.vstack([rng.permutation(r) for r in Rk]); ge+=chi(Rp)>=c-1e-12
    return dict(N=N,k=k,chi2=c,p_asymptotic=p_asym,iman_davenport_F=ff,p_iman_davenport=p_ff,p_permutation=(ge+1)/(nperm+1),avg_ranks=dict(zip(MODELS,Rk.mean(0))))
QA={6:2.850}
MTAG={'balanced_accuracy':'exactitud_balanceada','mcc':'MCC'}
def cd(N,k=6): return QA[k]*np.sqrt(k*(k+1)/(6*N))
def holm(ps):
    ps=np.asarray(ps); o=np.argsort(ps); m=len(ps); adj=np.empty(m); run=0
    for i,idx in enumerate(o): run=max(run,(m-i)*ps[idx]); adj[idx]=min(1,run)
    return adj
plt.rcParams["font.family"]="DejaVu Sans"
def fc(x,n=2): return f"{x:.{n}f}".replace(".",",")
def cd_plot(avg, cdv, title, fn, N, nota=None):
    # estilo de las figuras del artículo (make_figs.py, función cd_diagram); solo español
    names=sorted(avg,key=avg.get); ranks=np.array([avg[n] for n in names]); names=[LBL[n] for n in names]
    k=len(names); lo,hi=1,k
    fig,ax=plt.subplots(figsize=(7.2,2.1),dpi=300); ax.set_xlim(0,10); ax.set_ylim(2.3,6.4); ax.axis("off")
    def X(r): return 2.2+(r-lo)/(hi-lo)*5.6
    y0=4.8
    ax.plot([X(lo),X(hi)],[y0,y0],color="k",lw=1)
    for t in np.arange(lo,hi+0.01,0.5):
        ax.plot([X(t)]*2,[y0,y0+(0.18 if t%1==0 else 0.1)],color="k",lw=0.8)
        if t%1==0: ax.text(X(t),y0+0.28,f"{int(t)}",ha="center",fontsize=8)
    ax.text(X(hi)+0.25,y0+0.28,"rango promedio (1 = mejor)",ha="left",fontsize=6.5,style="italic")
    ax.plot([X(lo),X(lo+cdv)],[5.65,5.65],color="k",lw=1.6)
    for xx in (X(lo),X(lo+cdv)): ax.plot([xx,xx],[5.55,5.75],color="k",lw=1)
    ax.text((X(lo)+X(lo+cdv))/2,5.82,f"DC = {fc(cdv)}",ha="center",fontsize=7.5)
    half=(k+1)//2
    for i,(n,r) in enumerate(zip(names,ranks)):
        if i<half:
            yy=y0-0.55-i*0.55; ax.plot([X(r),X(r)],[y0,yy],color="k",lw=0.8); ax.plot([X(r),2.0],[yy,yy],color="k",lw=0.8)
            ax.text(1.95,yy,f"{n} ({fc(r)})",ha="right",va="center",fontsize=7.5)
        else:
            j=k-1-i; yy=y0-0.55-j*0.55; ax.plot([X(r),X(r)],[y0,yy],color="k",lw=0.8); ax.plot([X(r),8.0],[yy,yy],color="k",lw=0.8)
            ax.text(8.05,yy,f"({fc(r)}) {n}",ha="left",va="center",fontsize=7.5)
    cl=[]
    for i in range(k):
        j=i
        while j+1<k and ranks[j+1]-ranks[i]<cdv: j+=1
        if j>i and not any(a<=i and j<=b for a,b in cl): cl.append((i,j))
    for m,(a,b) in enumerate(cl):
        yy=y0-0.25-m*0.14; ax.plot([X(ranks[a])-0.05,X(ranks[b])+0.05],[yy,yy],color="#b00000",lw=2.6,solid_capstyle="round")
    ax.text(5,6.25,title,ha="center",fontsize=6.8,fontweight="bold")
    if nota: ax.text(5,2.4,nota,ha="center",fontsize=6.6,style="italic")
    plt.tight_layout(pad=0.2)
    for e in ("png","svg","pdf"): fig.savefig(f"{fn}.{e}",dpi=300)
    plt.close(fig)
def mat_of(df,metric,blocks):
    w=df.pivot_table(index=blocks,columns='model',values=metric); return w[MODELS].dropna()
stats_out={}
for metric in (['balanced_accuracy','mcc'] if len(full) else []):
    res={}
    # (a) blocks = strata (mean over splits)
    m1=RB.groupby(['stratum','model'])[metric].mean().unstack()[MODELS]
    fa=friedman(m1.values); fa['CD']=cd(len(m1)); res['blocks_strata']=fa
    cd_plot(fa['avg_ranks'],fa['CD'],f'Diferencia crítica (Nemenyi) — {'exactitud balanceada' if metric=='balanced_accuracy' else 'MCC'}; bloques = conjuntos de evaluación (media de las particiones)',f'{FG}/DC_{MTAG[metric]}_bloques_conjuntos',len(m1),nota=f'Friedman χ²({fa["k"]-1}) = {fc(fa["chi2"])}; p asintótica = {fc(fa["p_asymptotic"],3)}; p por permutación = {fc(fa["p_permutation"],3)}; N = {fa["N"]} conjuntos de evaluación, k = 6 modelos, α = 0,05')
    # (b) blocks = split x stratum
    m2=mat_of(RB,metric,['stratum','seed_idx']); fb=friedman(m2.values,nperm=5000); fb['CD']=cd(len(m2)); res['blocks_split_x_stratum']=fb
    cd_plot(fb['avg_ranks'],fb['CD'],f'Diferencia crítica (Nemenyi) — {'exactitud balanceada' if metric=='balanced_accuracy' else 'MCC'}; bloques = partición × conjunto de evaluación (no independientes)',f'{FG}/DC_{MTAG[metric]}_bloques_particion_conjunto',len(m2),nota=f'Friedman χ²({fb["k"]-1}) = {fc(fb["chi2"])}; p asintótica = {fc(fb["p_asymptotic"],3)}; p por permutación = {fc(fb["p_permutation"],3)}; N = {fb["N"]} bloques, k = 6 modelos, α = 0,05')
    # (c) blocks = participant x stratum (mean over splits where participant was in test)
    pp=PP[(PP.control=='true_labels')]; pp=pp.set_index(['stratum','seed_idx']).loc[[i for i in full if i in pp.set_index(['stratum','seed_idx']).index.unique()]].reset_index()
    m3=pp.groupby(['stratum','subject','model'])[metric].mean().unstack()[MODELS].dropna(); fp=friedman(m3.values,nperm=5000); fp['CD']=cd(len(m3)); res['blocks_participant_x_stratum']=fp
    cd_plot(fp['avg_ranks'],fp['CD'],f'Diferencia crítica (Nemenyi) — {'exactitud balanceada' if metric=='balanced_accuracy' else 'MCC'}; bloques = participante × conjunto de evaluación',f'{FG}/DC_{MTAG[metric]}_bloques_participante_conjunto',len(m3),nota=f'Friedman χ²({fp["k"]-1}) = {fc(fp["chi2"])}; p asintótica = {fc(fp["p_asymptotic"],3)}; p por permutación = {fc(fp["p_permutation"],3)}; N = {fp["N"]} bloques, k = 6 modelos, α = 0,05')
    # (d) healthy only (2017 ME/MI [+2020]) and SCI only (2019), blocks = split
    for tag,sel in [('healthy_only',[s for s in strata if not s.startswith('001-2019')]),('SCI_2019_only',[s for s in strata if s.startswith('001-2019')])]:
        sub=RB[RB.stratum.isin(sel)]
        if len(sub)==0: continue
        md=mat_of(sub,metric,['stratum','seed_idx']); fd=friedman(md.values,nperm=5000); fd['CD']=cd(len(md)); fd['strata']=sel; res[f'{tag}_blocks_split']=fd
        if tag=='SCI_2019_only': cd_plot(fd['avg_ranks'],fd['CD'],f'Diferencia crítica (Nemenyi) — {'exactitud balanceada' if metric=='balanced_accuracy' else 'MCC'}; solo 001-2019 (lesión medular), bloques = partición',f'{FG}/DC_{MTAG[metric]}_solo_LME_2019',len(md),nota=f'Friedman χ²({fd["k"]-1}) = {fc(fd["chi2"])}; p asintótica = {fc(fd["p_asymptotic"],3)}; p por permutación = {fc(fd["p_permutation"],3)}; N = {fd["N"]} particiones, k = 6 modelos, α = 0,05')
    # pairwise Wilcoxon + Holm and Bayesian signed-rank (blocks = split x stratum)
    pw=[]
    import baycomp
    for a,b in itertools.combinations(MODELS,2):
        x=m2[a].values; y=m2[b].values
        try: p=stats.wilcoxon(x,y,zero_method='wilcox').pvalue
        except ValueError: p=1.0
        pl,pe,pr=baycomp.two_on_multiple(x,y,rope=0.01 if metric=='balanced_accuracy' else 0.02,random_state=20261113)  # semilla fija del muestreo bayesiano
        pw.append(dict(metric=metric,model_a=a,model_b=b,mean_diff=float(np.mean(x-y)),median_diff=float(np.median(x-y)),wilcoxon_p=p,p_left_a_better=pl,p_rope=pe,p_right_b_better=pr))
    pw=pd.DataFrame(pw); pw['wilcoxon_p_holm']=holm(pw.wilcoxon_p.values)
    # baycomp devuelve (P(A mejor), P(ROPE), P(B mejor)); comprobado: diferencia media positiva ⇔ P(A mejor) alta
    res['pairwise']=pw.to_dict('records'); stats_out[metric]=res
    pw.to_csv(f'{T}/pairwise_wilcoxon_holm_bayes_{metric}.csv',index=False)
# vs chance: one-sample Wilcoxon of BA-chance across splits per stratum x model; and vs permutation control
vc=[]
for (st,mo),g in real.groupby(['stratum','model']):
    dlt=g.balanced_accuracy.values-CHANCE[st]
    p=stats.wilcoxon(dlt,alternative='greater').pvalue if len(dlt)>=5 and np.any(dlt!=0) else np.nan
    vc.append(dict(stratum=st,model=mo,n_splits=len(g),mean_ba=g.balanced_accuracy.mean(),chance=CHANCE[st],mean_minus_chance=dlt.mean(),wilcoxon_greater_p=p))
vc=pd.DataFrame(vc); vc['p_holm']=holm(vc.wilcoxon_greater_p.fillna(1).values); vc.to_csv(f'{T}/vs_chance.csv',index=False)
perm=S[S.control!='true_labels']; perm.groupby(['stratum','model','seed_idx'])[M].mean().reset_index().to_csv(f'{T}/permutation_control.csv',index=False)
json.dump({'meta':out,'stats':stats_out},open(f'{T}/stats.json','w'),indent=1,default=float)
print(json.dumps(out)); print(agg[agg.control=='true_labels'][['stratum','model','balanced_accuracy_mean','balanced_accuracy_std','mcc_mean','mcc_std','balanced_accuracy_count']].to_string())
for metric in stats_out:
    for k,v in stats_out[metric].items():
        if k!='pairwise': print(metric,k,{kk:(round(vv,4) if isinstance(vv,float) else vv) for kk,vv in v.items() if kk!='avg_ranks'}, {m:round(r,2) for m,r in v['avg_ranks'].items()})
