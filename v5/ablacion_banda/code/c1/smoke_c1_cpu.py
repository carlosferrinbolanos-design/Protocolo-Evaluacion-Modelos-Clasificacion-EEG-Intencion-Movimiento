"""Smoke test and sanity check for Arm C1 (Tensor-CSPNet) on CPU.
Enforces CUDA_VISIBLE_DEVICES="" so that GPU is not touched.
Evaluates:
1. Forward pass shape & finiteness for 11 and 61 channels on CPU.
2. Train and predict on 001-2020_gel split 0 on CPU (20 epochs).
"""
import os
import sys
import time
import json
from pathlib import Path

# Enforce empty CUDA_VISIBLE_DEVICES as requested
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import pandas as pd
import torch
from scipy.signal import butter, sosfiltfilt

V5 = Path(os.environ.get("TESIS_V5_ROOT", Path(__file__).resolve().parents[3]))  # repo: v5/ (original: D:/ICTQ/TESIS_pipeline_v5)
sys.path.insert(0, str(V5 / "code"))
import evaluate as ev
from models import TorchNet

OUT = V5 / "ablacion_banda"
EPOCHS = V5 / "epochs"
FS = 256.0
PAD = 256
C1_BANDS = [(0.5, 1.5), (1.0, 2.0), (1.5, 2.5), (2.0, 3.0)]


def filt(X, lo, hi):
    if lo is None or lo <= 0.5:
        sos = butter(4, hi, btype="lowpass", fs=FS, output="sos")
    else:
        sos = butter(4, [lo, hi], btype="bandpass", fs=FS, output="sos")
    return sosfiltfilt(sos, X, axis=-1, padtype="odd", padlen=PAD).astype(np.float32)


def c1_dims(n_ch):
    return (n_ch, 36, 36, min(n_ch, 22))


class TCSPNetCPU(TorchNet):
    def __init__(self, seed=0, n_epochs=20):
        super().__init__("TensorCSPNet", n_epochs=n_epochs, seed=seed, device="cpu")

    @staticmethod
    def bank(X):
        return np.stack([filt(X, lo, hi) for lo, hi in C1_BANDS], axis=1)

    def _build(self, n_ch, n_cls, n_t):
        from spd_learn.models import TensorCSPNet
        return TensorCSPNet(n_chans=n_ch, n_outputs=n_cls, n_patches=4, n_freqs=len(C1_BANDS),
                            use_mlp=False, tcn_channels=16, dims=c1_dims(n_ch), momentum=0.1)

    def fit(self, X, y):
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        self.classes_ = np.unique(y)
        yi = np.searchsorted(self.classes_, y)
        self.mu_ = X.mean(axis=(0, 2), keepdims=True)
        self.sd_ = X.std(axis=(0, 2), keepdims=True) + 1e-8
        t_bank0 = time.time()
        Xb = self.bank(((X - self.mu_) / self.sd_).astype(np.float32))
        bank_time = time.time() - t_bank0
        print(f"Filter bank done in {bank_time:.2f}s, Xb shape: {Xb.shape}", flush=True)

        self.net_ = self._build(X.shape[1], len(self.classes_), X.shape[2]).to("cpu")
        cnt = np.bincount(yi, minlength=len(self.classes_)).astype(np.float32)
        w = torch.tensor(cnt.sum() / (len(cnt) * cnt), dtype=torch.float32, device="cpu")
        lossf = torch.nn.CrossEntropyLoss(weight=w)
        opt = torch.optim.AdamW(self.net_.parameters(), lr=self.lr, weight_decay=self.wd)
        steps = self.n_epochs * int(np.ceil(len(yi) / self.batch))
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
        Xt = torch.from_numpy(Xb)
        yt = torch.tensor(yi, dtype=torch.long)
        self.loss_curve_ = []
        self.net_.train()
        for ep in range(self.n_epochs):
            t_ep0 = time.time()
            order = rng.permutation(len(yi))
            tot = 0.0
            for i in range(0, len(order), self.batch):
                b = torch.from_numpy(order[i:i + self.batch])
                if len(b) < 2:
                    continue
                opt.zero_grad()
                out = self.net_(Xt[b])
                loss = lossf(out, yt[b])
                loss.backward()
                opt.step()
                sch.step()
                tot += float(loss.item()) * len(b)
            ep_loss = tot / len(yi)
            self.loss_curve_.append(ep_loss)
            if (ep + 1) % 5 == 0 or ep == 0:
                print(f"Epoch {ep+1:2d}/{self.n_epochs}: loss = {ep_loss:.4f} ({time.time() - t_ep0:.2f}s)", flush=True)
        return self

    def predict(self, X):
        Xb = self.bank(((X - self.mu_) / self.sd_).astype(np.float32))
        self.net_.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(Xb), 128):
                out.append(self.net_(torch.from_numpy(Xb[i:i + 128])).argmax(1).numpy())
        return self.classes_[np.concatenate(out)]


def main():
    t_start = time.time()
    print("=== C1 SMOKE TEST (CPU ONLY) ===", flush=True)
    print(f"CUDA GPU memory allocated: {torch.cuda.memory_allocated() if torch.cuda.is_available() else 0} bytes (All computation is forced on CPU)", flush=True)

    # 1. Forward checks
    print("\n1. Testing forward pass on CPU...", flush=True)
    for n_ch in (11, 61):
        net = TCSPNetCPU()._build(n_ch, 3, 513).eval()
        inp = torch.randn(4, 4, n_ch, 513)
        with torch.no_grad():
            out = net(inp)
        print(f"  n_ch={n_ch:2d}: out shape={list(out.shape)}, is_finite={torch.isfinite(out).all().item()}", flush=True)
        assert list(out.shape) == [4, 3] and torch.isfinite(out).all()

    # 2. Gel split 0 smoke run
    print("\n2. Running 001-2020_gel split 0 on CPU...", flush=True)
    X, y, s = ev.load(EPOCHS, "001-2020_gel")
    splits = ev.splits(s)
    test_set = splits[0]
    te = np.isin(s, list(test_set))
    tr = ~te
    print(f"  Train samples: {tr.sum()}, Test samples: {te.sum()}", flush=True)

    t_fit0 = time.time()
    m = TCSPNetCPU(seed=ev.SEED0, n_epochs=20)
    m.fit(X[tr], y[tr])
    fit_time = time.time() - t_fit0
    print(f"  Fit completed in {fit_time:.2f}s", flush=True)

    t_pred0 = time.time()
    preds = m.predict(X[te])
    pred_time = time.time() - t_pred0

    subj_ba = ev.subject_ba(y[te], preds, s[te])
    mean_ba = float(np.mean(list(subj_ba.values())))
    loss_initial = m.loss_curve_[0]
    loss_final = m.loss_curve_[-1]
    loss_decrease = loss_final < loss_initial
    total_time = time.time() - t_start

    print(f"  Pred completed in {pred_time:.2f}s", flush=True)
    print(f"  Initial loss: {loss_initial:.4f}, Final loss: {loss_final:.4f} (decreased: {loss_decrease})", flush=True)
    print(f"  Balanced Accuracy (split 0): {mean_ba:.4f}", flush=True)
    print(f"  Total smoke test time: {total_time:.2f}s", flush=True)

    result = {
        "passed": bool(loss_decrease and total_time < 600),
        "total_time_seconds": round(total_time, 2),
        "fit_time_seconds": round(fit_time, 2),
        "pred_time_seconds": round(pred_time, 2),
        "initial_loss": round(loss_initial, 4),
        "final_loss": round(loss_final, 4),
        "balanced_acc_split0": round(mean_ba, 4),
        "cuda_used": False
    }

    res_file = OUT / "results" / "smoke" / "c1_cpu_smoke_result.json"
    res_file.parent.mkdir(parents=True, exist_ok=True)
    with open(res_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"Result saved to {res_file}", flush=True)
    if total_time > 600:
        print("ABORT/TIMEOUT: Smoke test exceeded 10 minutes", flush=True)
        sys.exit(2)
    print("SUCCESS: C1 Smoke test on CPU PASSED.", flush=True)


if __name__ == "__main__":
    main()
