"""Build a self-contained bootstrap .py for a Colab VM (no file upload needed): embeds scripts + David's
manifests, downloads public raw data from Zenodo inside the VM, extracts epochs, runs the given task list in background."""
import base64, gzip, sys, json
from pathlib import Path
S=Path(__file__).resolve().parents[1]; TES=Path(__file__).resolve().parents[2]/'manifests'
MAN={'001-2017_ME':'001-2017_cue_aligned_raw/dataset1_me_cue_epoch_manifest_reextracted_v2.csv','001-2017_MI':'001-2017_cue_aligned_raw/dataset1_mi_cue_epoch_manifest_reextracted_v2.csv','001-2019_offline':'dataset2_offline_epoch_manifest_model_input_v2.csv'}
def b64(p): return base64.b64encode(gzip.compress(Path(p).read_bytes())).decode()
def build(tag, strata, tasks):
    files={'core.py':b64(S/'core.py'),'deep_core.py':b64(S/'deep_core.py'),'01_extract_epochs.py':b64(S/'01_extract_epochs.py'),'02_run_classical.py':b64(S/'02_run_classical.py'),'03_run_deep.py':b64(S/'03_run_deep.py')}
    mans={MAN[s]:b64(TES/MAN[s]) for s in strata}
    lines=['set -x','cd /content/w','export TES=/content/w/tes RAW=/content/raw EP=/content/ep PREDS=/content/out','mkdir -p $EP $PREDS /content/raw/2017 /content/raw/2019','python dl.py '+' '.join(strata)+' > dl.log 2>&1','python 01_extract_epochs.py '+' '.join(strata)+' > extract.log 2>&1']
    lines+=tasks+['echo ALLDONE > /content/out/ALLDONE_'+tag]
    pipeline='\n'.join(lines)
    dl=r'''
import sys,json,os,subprocess,urllib.request,concurrent.futures as cf,pandas as pd,glob,zipfile
os.chdir('/content/w')
need17=set()
for s in sys.argv[1:]:
    if s.startswith('001-2017'):
        p='tes/001-2017_cue_aligned_raw/dataset1_%s_cue_epoch_manifest_reextracted_v2.csv'%s[-2:].lower()
        need17|=set(os.path.basename(x) for x in pd.read_csv(p).source_gdf)
meta={f['key']:f['size'] for f in json.load(urllib.request.urlopen('https://zenodo.org/api/records/834976'))['files']}
def get(key,rec,dest,size):
    for a in range(8):
        if os.path.exists(dest) and os.path.getsize(dest)==size: return key,'ok'
        subprocess.run(['curl','-sL','--retry','5','--max-time','1500','-o',dest+'.part',f'https://zenodo.org/records/{rec}/files/{key}?download=1'])
        if os.path.exists(dest+'.part') and os.path.getsize(dest+'.part')==size: os.replace(dest+'.part',dest); return key,'ok'
    return key,'FAIL'
jobs=[(k,'834976','/content/raw/2017/'+k,meta[k]) for k in sorted(need17)]
if '001-2019_offline' in sys.argv[1:]:
    m19={f['key']:f['size'] for f in json.load(urllib.request.urlopen('https://zenodo.org/api/records/2222268'))['files']}
    jobs+=[(f'P{i:02d}.zip','2222268',f'/content/raw/2019/P{i:02d}.zip',m19[f'P{i:02d}.zip']) for i in range(1,11)]
with cf.ThreadPoolExecutor(32) as ex:
    res=list(ex.map(lambda j:get(*j),jobs))
print('FAILS',[r for r in res if r[1]!='ok'],flush=True)
for z in glob.glob('/content/raw/2019/*.zip'):
    zipfile.ZipFile(z).extractall(z[:-4]); os.remove(z)
print('DLDONE',len(res),flush=True)
'''
    src=f'''import base64,gzip,os,subprocess
from pathlib import Path
W=Path('/content/w'); (W/'tes'/'001-2017_cue_aligned_raw').mkdir(parents=True,exist_ok=True)
F={json.dumps(files)}
M={json.dumps(mans)}
for k,v in F.items(): (W/k).write_bytes(gzip.decompress(base64.b64decode(v)))
for k,v in M.items(): (W/'tes'/k).write_bytes(gzip.decompress(base64.b64decode(v)))
(W/'dl.py').write_text({dl!r})
(W/'pipeline.sh').write_text({pipeline!r})
subprocess.Popen('nohup bash /content/w/pipeline.sh > /content/w/pipeline.log 2>&1 &',shell=True)
print('LAUNCHED {tag}')
'''
    Path(S/f'boot_{tag}.py').write_text(src); print(tag, len(src))
CL='python 02_run_classical.py {s} --prep && '
plans={
 'A':(['001-2017_ME','001-2019_offline'],[
   '(python 03_run_deep.py 001-2017_ME --seeds 0-9 && python 03_run_deep.py 001-2017_ME --seeds 0-1 --control labels_permuted_within_train_participant && python 03_run_deep.py 001-2019_offline --seeds 0-9 && python 03_run_deep.py 001-2019_offline --seeds 0-1 --control labels_permuted_within_train_participant) > deep.log 2>&1 &',
   'python 02_run_classical.py 001-2017_ME --prep > cl.log 2>&1; python 02_run_classical.py 001-2017_ME --seeds 0-4 >> cl.log 2>&1; python 02_run_classical.py 001-2017_ME --seeds 0-0 --control labels_permuted_within_train_participant >> cl.log 2>&1','wait']),
 'B':(['001-2017_ME'],['python 02_run_classical.py 001-2017_ME --prep > cl.log 2>&1; python 02_run_classical.py 001-2017_ME --seeds 5-9 >> cl.log 2>&1; python 02_run_classical.py 001-2017_ME --seeds 1-1 --control labels_permuted_within_train_participant >> cl.log 2>&1']),
 'C':(['001-2017_MI'],[
   '(python 03_run_deep.py 001-2017_MI --seeds 0-9 && python 03_run_deep.py 001-2017_MI --seeds 0-1 --control labels_permuted_within_train_participant) > deep.log 2>&1 &',
   'python 02_run_classical.py 001-2017_MI --prep > cl.log 2>&1; python 02_run_classical.py 001-2017_MI --seeds 0-4 >> cl.log 2>&1; python 02_run_classical.py 001-2017_MI --seeds 0-0 --control labels_permuted_within_train_participant >> cl.log 2>&1','wait']),
 'D':(['001-2017_MI'],['python 02_run_classical.py 001-2017_MI --prep > cl.log 2>&1; python 02_run_classical.py 001-2017_MI --seeds 5-9 >> cl.log 2>&1; python 02_run_classical.py 001-2017_MI --seeds 1-1 --control labels_permuted_within_train_participant >> cl.log 2>&1']),
}
for t,(st,ts) in plans.items(): build(t,st,ts)
