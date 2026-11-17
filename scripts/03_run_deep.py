"""Deep models (EEGNet, ShallowConvNet, DeepConvNet) on GPU for repeated splits; David's v3 training config."""
import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
from core import *
from deep_core import run_deep
from sklearn.preprocessing import LabelEncoder
ap=argparse.ArgumentParser(); ap.add_argument('stratum'); ap.add_argument('--seeds',default='0-9'); ap.add_argument('--control',default='true_labels')
ap.add_argument('--models',default='EEGNet,ShallowConvNet,DeepConvNet'); ap.add_argument('--weighted_cosine',action='store_true'); a=ap.parse_args(); name=a.stratum
EP=os.environ.get('EP','epocas'); OUTD=os.environ.get('PREDS',os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),'preds'))
lo,hi=map(int,a.seeds.split('-'))
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); torch.backends.cudnn.benchmark=True
z=np.load(f'{EP}/{name}.npz',allow_pickle=True); X=z['X']; y=z['y']; s=z['s']
le=LabelEncoder(); yy=le.fit_transform(y); cls=le.classes_
for k in range(lo,hi+1):
  tr,te,ts=split(name,s,k)
  for m in a.models.split(','):
    if done(OUTD,name,m,a.control,k): continue
    ytrain=yy if a.control=='true_labels' else le.transform(permute_train(y,s,tr,seed_of(k)+sum(map(ord,m+name))))
    pred,rt=run_deep(name,m,a.control,X,yy,ytrain,tr,te,seed_of(k),device,weighted_cosine=a.weighted_cosine)
    save_preds(OUTD,name,m,a.control,k,s,te,yy[te],pred,cls,rt,ts); print('DONE',name,m,a.control,k,round(rt,1),flush=True)
