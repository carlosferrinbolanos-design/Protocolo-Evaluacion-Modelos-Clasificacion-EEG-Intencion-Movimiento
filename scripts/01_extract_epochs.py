"""Re-extract David's v2/v3 epochs from public raw sources using his frozen manifests.
Selection = exactly the manifest rows (source file + onset_sample + class), window -0.5..1.5 s,
no filtering, no baseline, no amplitude rejection (same as David's pipeline).
001-2017: first 61 channels of GDF (512 Hz) -> epoch 1025 samples -> resample_poly(1,2) -> 513 (as in v3 loader).
001-2019: first 61 channels of GDF (256 Hz), 513 samples; finite_for_model==True only.
001-2020: 11 canonical sensorimotor channels (FC3,FCZ,FC4,C3,C1,CZ,C2,C4,CP3,CPZ,CP4) from MAT signal.
Each manifest onset is verified against an event in the raw file."""
import sys, re, glob, json, os
from pathlib import Path
import numpy as np, pandas as pd, mne
from scipy.signal import resample_poly
mne.set_log_level('ERROR')
TES=Path(os.environ.get('TES',str(Path(__file__).resolve().parents[1]/'manifests')))
RAW=Path(os.environ.get('RAW','datos_crudos')); OUT=Path(os.environ.get('EP','epocas'))
D3=['FC3','FCZ','FC4','C3','C1','CZ','C2','C4','CP3','CPZ','CP4']
def canon(x): return re.sub('[^A-Z0-9]+','',str(x).upper())
SPEC={'001-2017_ME':(TES/'001-2017_cue_aligned_raw'/'dataset1_me_cue_epoch_manifest_reextracted_v2.csv','subject'),
 '001-2017_MI':(TES/'001-2017_cue_aligned_raw'/'dataset1_mi_cue_epoch_manifest_reextracted_v2.csv','subject'),
 '001-2019_offline':(TES/'dataset2_offline_epoch_manifest_model_input_v2.csv','subject'),
 '001-2020_gel':(TES/'dataset3_gel_epoch_manifest_v2.csv','participant'),
 '001-2020_water':(TES/'dataset3_water_epoch_manifest_v2.csv','participant'),
 '001-2020_dry':(TES/'dataset3_dry_epoch_manifest_v2.csv','participant')}
def find2019(src):
    base=Path(src).name; c=glob.glob(str(RAW/'2019'/'**'/base),recursive=True); c=[x for x in c if '__MACOSX' not in x]
    assert c, src; return c[0]
def run(name):
    man,sc=SPEC[name]; d=pd.read_csv(man)
    if 'finite_for_model' in d.columns: d=d[d.finite_for_model.astype(bool)].copy()
    d=d.sort_values(['epoch_file','epoch_index']).reset_index(drop=True)
    srccol='source_gdf' if 'source_gdf' in d.columns else 'source_file'
    X=[None]*len(d); report=[]
    for src,b in d.groupby(srccol,sort=False):
        if name.startswith('001-2017'):
            raw=mne.io.read_raw_gdf(RAW/'2017'/Path(src).name,preload=True); fs=raw.info['sfreq']; assert fs==512
            sig=raw.get_data()[:61].astype(np.float32); ev=np.round(raw.annotations.onset*fs).astype(int); pre=256; n=1025
        elif name.startswith('001-2019'):
            raw=mne.io.read_raw_gdf(find2019(src),preload=True); fs=raw.info['sfreq']; assert fs==256
            sig=raw.get_data()[:61].astype(np.float32); ev=np.round(raw.annotations.onset*fs).astype(int); pre=128; n=513
        else:
            import scipy.io
            m=scipy.io.loadmat(RAW/'2020'/src,struct_as_record=False,squeeze_me=True); h=m['header']
            labels=[canon(x) for x in np.ravel(h.channels_labels)]; idx=[labels.index(c) for c in D3]
            sig=np.asarray(m['signal'])[idx].astype(np.float32); ev=np.ravel(m['events'].positions).astype(int)-1; pre=128; n=513
        on=b.onset_sample.to_numpy(int); matched=np.isin(on,ev).mean()
        for i,o in zip(b.index,on):
            e=sig[:,o-pre:o-pre+n]; assert e.shape[1]==n,(src,o)
            if name.startswith('001-2017'): e=resample_poly(e,1,2,axis=-1).astype(np.float32)
            X[i]=e
        report.append({'source':src,'n':len(b),'onset_matched_raw_event_frac':float(matched)})
    X=np.stack(X).astype(np.float32); fin=np.isfinite(X).reshape(len(X),-1).all(1)
    y=np.asarray(d.movement_class.astype(str).tolist(),dtype=str); s=np.asarray(d[sc].astype(str).tolist(),dtype=str)
    np.savez(OUT/f'{name}.npz',X=X,y=y,s=s)
    r={'stratum':name,'n_epochs':len(d),'shape':list(X.shape),'all_finite':bool(fin.all()),'n_nonfinite':int((~fin).sum()),'subjects':int(len(set(s))),'min_onset_match':min(x['onset_matched_raw_event_frac'] for x in report),'mean_onset_match':float(np.mean([x['onset_matched_raw_event_frac'] for x in report]))}
    json.dump({'summary':r,'per_source':report},open(OUT/f'{name}_extract_report.json','w'),indent=1); print(json.dumps(r),flush=True)
for n in sys.argv[1:]: run(n)
