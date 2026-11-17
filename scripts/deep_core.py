# Deep models copied from David's executed v3 deep script (<deep_models_v2>): identical architectures/training.
import time, gc, json, random
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
class EEGNet(nn.Module):
 def __init__(self,c,k):
  super().__init__();self.t=nn.Conv2d(1,8,(1,64),padding=(0,32),bias=False);self.b1=nn.BatchNorm2d(8);self.s=nn.Conv2d(8,16,(c,1),groups=8,bias=False);self.b2=nn.BatchNorm2d(16);self.d=nn.Conv2d(16,16,(1,16),groups=16,padding=(0,8),bias=False);self.p=nn.Conv2d(16,16,1,bias=False);self.b3=nn.BatchNorm2d(16);self.o=nn.Dropout(.25);self.f=nn.Linear(16,k)
 def forward(self,x):
  x=F.elu(self.b1(self.t(x)));x=F.elu(self.b2(self.s(x)));x=self.o(F.avg_pool2d(x,(1,4)));x=F.elu(self.b3(self.p(self.d(x))));x=self.o(F.avg_pool2d(x,(1,8)));return self.f(F.adaptive_avg_pool2d(x,1).flatten(1))
class ShallowConvNet(nn.Module):
 def __init__(self,c,k):
  super().__init__();self.t=nn.Conv2d(1,40,(1,25),padding=(0,12),bias=False);self.s=nn.Conv2d(40,40,(c,1),bias=False);self.b=nn.BatchNorm2d(40);self.o=nn.Dropout(.5);self.f=nn.Linear(40,k)
 def forward(self,x):
  x=self.b(self.s(self.t(x)));x=x*x;x=F.avg_pool2d(x,(1,75),stride=(1,15));x=torch.log(torch.clamp(x,min=1e-6));x=self.o(x);return self.f(F.adaptive_avg_pool2d(x,1).flatten(1))
class DeepConvNet(nn.Module):
 def __init__(self,c,k):
  super().__init__();self.a=nn.Sequential(nn.Conv2d(1,25,(1,10),padding=(0,5),bias=False),nn.Conv2d(25,25,(c,1),bias=False),nn.BatchNorm2d(25),nn.ELU(),nn.MaxPool2d((1,3),stride=(1,3)),nn.Dropout(.5))
  def B(i,o):return nn.Sequential(nn.Conv2d(i,o,(1,10),padding=(0,5),bias=False),nn.BatchNorm2d(o),nn.ELU(),nn.MaxPool2d((1,3),stride=(1,3)),nn.Dropout(.5))
  self.b=B(25,50);self.c=B(50,100);self.d=B(100,200);self.f=nn.Linear(200,k)
 def forward(self,x):return self.f(F.adaptive_avg_pool2d(self.d(self.c(self.b(self.a(x)))),1).flatten(1))
def make(m,c,k):return {'EEGNet':EEGNet,'ShallowConvNet':ShallowConvNet,'DeepConvNet':DeepConvNet}[m](c,k)
def run_deep(name,m,ctl,X,yy,ytrain,tr,te,seed,device,weighted_cosine=False):
 # yy: true int labels; ytrain: labels used for training (true or permuted)
 base=seed+sum(map(ord,m+name+ctl))
 mu=X[tr].mean(axis=(0,2),dtype=np.float64).astype(np.float32);sd=np.maximum(X[tr].std(axis=(0,2),dtype=np.float64).astype(np.float32),1e-6)
 xn=((X-mu[None,:,None])/sd[None,:,None]).astype(np.float32)
 g=torch.Generator();g.manual_seed(base)
 dl=DataLoader(TensorDataset(torch.from_numpy(xn[tr]).unsqueeze(1),torch.from_numpy(ytrain[tr]).long()),batch_size=64,shuffle=True,generator=g,num_workers=0)
 tl=DataLoader(TensorDataset(torch.from_numpy(xn[te]).unsqueeze(1),torch.from_numpy(yy[te]).long()),batch_size=128,shuffle=False,num_workers=0)
 torch.manual_seed(base);np.random.seed(base%2**32);random.seed(base)
 K=int(yy.max())+1;net=make(m,xn.shape[1],K).to(device);op=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.0001);t=time.time();net.train()
 if weighted_cosine:
  # David's frozen 2019/2020 config (source notebook train_net): weights = n/(K*counts) on training labels; CosineAnnealingLR(T_max=20) stepped per epoch
  counts=np.bincount(ytrain[tr],minlength=K);wts=torch.tensor((tr.sum()/(K*counts)).astype(np.float32),device=device)
  crit=nn.CrossEntropyLoss(weight=wts);sch=torch.optim.lr_scheduler.CosineAnnealingLR(op,T_max=20)
 else:
  crit=nn.CrossEntropyLoss();sch=None
 for ep in range(20):
  net.train()
  for xb,yb in dl:
   xb=xb.to(device,non_blocking=True);yb=yb.to(device,non_blocking=True);op.zero_grad(set_to_none=True);loss=crit(net(xb),yb);loss.backward();op.step()
  if sch is not None: sch.step()
 net.eval();pr=[]
 with torch.no_grad():
  for xb,yb in tl:pr.extend(net(xb.to(device,non_blocking=True)).argmax(1).cpu().numpy())
 rt=time.time()-t
 del net,op,dl,tl,xn;torch.cuda.empty_cache();gc.collect()
 return np.asarray(pr),rt
