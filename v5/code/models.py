"""The six model families of the thesis, fixed before any evaluation (pipeline v5).

Same settings for every stratum (the v2 run used different CSP regularisation and network
training settings for 001-2017; v5 removes that inconsistency):
  CSP_LDA    8-30 Hz Butterworth (order 4, zero phase) on the epoch; mne CSP, Ledoit-Wolf,
             log average power; 12 components (6 when the stratum has 11 channels);
             LDA lsqr with automatic shrinkage.
  FBCSP_SVM  six 4-Hz bands from 8 to 32 Hz; CSP Ledoit-Wolf with 4 components per band;
             standardised features; SVM RBF C=1, gamma=scale, balanced class weights.
  MDM        8-30 Hz; OAS covariance; Riemannian minimum distance to mean (pyriemann).
  EEGNet, ShallowConvNet, DeepConvNet  braindecode EEGNet / ShallowFBCSPNet / Deep4Net on the
             0.5 Hz high-passed epoch; per-channel z-score fitted on training data only;
             20 epochs, AdamW lr 1e-3, weight decay 1e-4, batch 64, class-weighted
             cross-entropy, cosine schedule. No early stopping (no data seen from test).
"""
import numpy as np
from scipy.signal import butter, sosfiltfilt
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

FS = 256.0
import mne  # noqa: E402
mne.set_log_level("ERROR")


class Bandpass(BaseEstimator, TransformerMixin):
    def __init__(self, lo, hi):
        self.lo, self.hi = lo, hi

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        sos = butter(4, [self.lo, self.hi], btype="bandpass", fs=FS, output="sos")
        return sosfiltfilt(sos, X, axis=-1).astype(np.float32)


def _csp(n):
    from mne.decoding import CSP
    return CSP(n_components=n, reg="ledoit_wolf", log=True, norm_trace=False,
               transform_into="average_power")


def make_classical(name, n_ch):
    if name == "CSP_LDA":
        return make_pipeline(Bandpass(8, 30), _csp(12 if n_ch > 11 else 6),
                             LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"))
    if name == "FBCSP_SVM":
        bands = [(8, 12), (12, 16), (16, 20), (20, 24), (24, 28), (28, 32)]
        fb = make_union(*[make_pipeline(Bandpass(lo, hi), _csp(4)) for lo, hi in bands])
        return make_pipeline(fb, StandardScaler(),
                             SVC(kernel="rbf", C=1.0, gamma="scale", class_weight="balanced"))
    if name == "MDM":
        from pyriemann.classification import MDM
        from pyriemann.estimation import Covariances
        return make_pipeline(Bandpass(8, 30), Covariances(estimator="oas"), MDM(metric="riemann"))
    raise ValueError(name)


class TorchNet(BaseEstimator, ClassifierMixin):
    def __init__(self, arch, n_epochs=20, lr=1e-3, wd=1e-4, batch=64, seed=0, device="cuda"):
        self.arch, self.n_epochs, self.lr, self.wd = arch, n_epochs, lr, wd
        self.batch, self.seed, self.device = batch, seed, device

    def _build(self, n_ch, n_cls, n_t):
        from braindecode.models import Deep4Net, EEGNet, ShallowFBCSPNet
        kw = dict(n_chans=n_ch, n_outputs=n_cls, n_times=n_t)
        if self.arch == "EEGNet":
            return EEGNet(**kw)
        if self.arch == "ShallowConvNet":
            return ShallowFBCSPNet(**kw, final_conv_length="auto")
        if self.arch == "DeepConvNet":
            return Deep4Net(**kw, final_conv_length="auto")
        raise ValueError(self.arch)

    def fit(self, X, y):
        import torch
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        self.classes_ = np.unique(y)
        yi = np.searchsorted(self.classes_, y)
        self.mu_ = X.mean(axis=(0, 2), keepdims=True)
        self.sd_ = X.std(axis=(0, 2), keepdims=True) + 1e-8
        Xn = ((X - self.mu_) / self.sd_).astype(np.float32)
        self.net_ = self._build(X.shape[1], len(self.classes_), X.shape[2]).to(self.device)
        cnt = np.bincount(yi, minlength=len(self.classes_)).astype(np.float32)
        w = torch.tensor(cnt.sum() / (len(cnt) * cnt), device=self.device)
        lossf = torch.nn.CrossEntropyLoss(weight=w)
        opt = torch.optim.AdamW(self.net_.parameters(), lr=self.lr, weight_decay=self.wd)
        steps = self.n_epochs * int(np.ceil(len(yi) / self.batch))
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
        Xt = torch.tensor(Xn, device=self.device)
        yt = torch.tensor(yi, device=self.device)
        self.net_.train()
        for _ in range(self.n_epochs):
            order = rng.permutation(len(yi))
            for i in range(0, len(order), self.batch):
                b = torch.tensor(order[i:i + self.batch], device=self.device)
                opt.zero_grad()
                loss = lossf(self.net_(Xt[b]), yt[b])
                loss.backward()
                opt.step()
                sch.step()
        return self

    def predict(self, X):
        import torch
        Xn = ((X - self.mu_) / self.sd_).astype(np.float32)
        self.net_.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(Xn), 128):
                out.append(self.net_(torch.tensor(Xn[i:i + 128], device=self.device)).argmax(1).cpu().numpy())
        return self.classes_[np.concatenate(out)]


CLASSICAL = ["CSP_LDA", "FBCSP_SVM", "MDM"]
NETS = ["EEGNet", "ShallowConvNet", "DeepConvNet"]


def make_model(name, n_ch, seed=0):
    if name in CLASSICAL:
        return make_classical(name, n_ch)
    return TorchNet(name, seed=seed)
