"""Classical pipelines (CSP+LDA, FBCSP+SVM, Riemannian MDM) exactly as in David's v3 cell 2 / cell 12,
for repeated participant-disjoint splits. Filtering is per-epoch (zero-phase), done once per stratum
(identical to David's code which filtered all epochs before splitting)."""
import sys, os, time, gc, argparse
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from core import *
from scipy.signal import butter, sosfiltfilt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from mne.decoding import CSP
from pyriemann.estimation import Covariances
from pyriemann.classification import MDM
import mne; mne.set_log_level('ERROR')
ap = argparse.ArgumentParser(); ap.add_argument('stratum'); ap.add_argument('--seeds', default='0-9'); ap.add_argument('--models', default='CSP_LDA,FBCSP_SVM,Riemannian_MDM')
ap.add_argument('--control', default='true_labels'); ap.add_argument('--out', default=os.environ.get('PREDS',os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),'preds'))); ap.add_argument('--prep', action='store_true')
a = ap.parse_args(); name = a.stratum
lo, hi = map(int, a.seeds.split('-')); seeds = range(lo, hi + 1)
EP = os.environ.get('EP','epocas'); BANDS = [(8, 12), (12, 16), (16, 20), (20, 24), (24, 28), (28, 32)]
z = np.load(f'{EP}/{name}.npz', allow_pickle=True); y = z['y']; s = z['s']
def bandfile(tag): return f'{EP}/{name}__{tag}.npy'
if a.prep:
    X = z['X']
    for tag, (l, h) in [('b8_30', (8, 30))] + [(f'b{l}_{h}', (l, h)) for l, h in BANDS]:
        if not os.path.exists(bandfile(tag)):
            np.save(bandfile(tag) + '.tmp.npy', sosfiltfilt(butter(4, [l, h], btype='bandpass', fs=256, output='sos'), X, axis=-1).astype(np.float32))
            os.replace(bandfile(tag) + '.tmp.npy', bandfile(tag)); print('prep', tag, flush=True)
    sys.exit(0)
le = LabelEncoder(); yi = le.fit_transform(y); cls = le.classes_
XB = np.load(bandfile('b8_30'), mmap_mode='r')
for k in seeds:
    tr, te, ts = split(name, s, k)
    ytr = yi if a.control == 'true_labels' else le.transform(permute_train(y, s, tr, seed_of(k)))
    for m in a.models.split(','):
        if done(a.out, name, m, a.control, k): continue
        t = time.time()
        if m == 'CSP_LDA':
            Xtr = np.asarray(XB[tr]); Xte = np.asarray(XB[te])
            csp = CSP(n_components=12, reg='ledoit_wolf', log=True, norm_trace=False)
            F1 = csp.fit_transform(Xtr, ytr[tr]); F2 = csp.transform(Xte)
            pred = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto').fit(F1, ytr[tr]).predict(F2)
        elif m == 'Riemannian_MDM':
            cov = Covariances(estimator='oas'); C1 = cov.transform(np.asarray(XB[tr])); C2 = cov.transform(np.asarray(XB[te]))
            pred = MDM(metric='riemann').fit(C1, ytr[tr]).predict(C2)
        elif m == 'FBCSP_SVM':
            f1 = []; f2 = []
            for l, h in BANDS:
                B = np.load(bandfile(f'b{l}_{h}'), mmap_mode='r')
                csp = CSP(n_components=4, reg='ledoit_wolf', log=True, norm_trace=False)
                f1.append(csp.fit_transform(np.asarray(B[tr]), ytr[tr])); f2.append(csp.transform(np.asarray(B[te]))); del B, csp; gc.collect()
            F1 = np.hstack(f1); F2 = np.hstack(f2); sc = StandardScaler().fit(F1)
            pred = SVC(kernel='rbf', C=1.0, gamma='scale', class_weight='balanced').fit(sc.transform(F1), ytr[tr]).predict(sc.transform(F2))
        save_preds(a.out, name, m, a.control, k, s, te, yi[te], pred, cls, time.time() - t, ts)
        print('DONE', name, m, a.control, k, round(time.time() - t, 1), flush=True); gc.collect()
