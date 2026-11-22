"""Sequential GPU runner for Arm C1 (Tensor-CSPNet), v2.
Copy of c1_runner.py with: faulthandler to a crash file, per-step progress logging,
stratum order 2020 -> 2019 -> 2017, and --strata / --max-splits for a foreground test.
Writes strictly to results/c1/ (same checkpoints, so it resumes). Log: c1_run_v2.log.
"""
import argparse
import faulthandler
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt

V5 = Path(os.environ.get("TESIS_V5_ROOT", Path(__file__).resolve().parents[3]))  # repo: v5/ (original: D:/ICTQ/TESIS_pipeline_v5)
sys.path.insert(0, str(V5 / "code"))
import evaluate as ev
from models import TorchNet

OUT_C1 = V5 / "ablacion_banda" / "results" / "c1"
CHECKPOINTS = OUT_C1 / "checkpoints"
PREDICTIONS = OUT_C1 / "predictions"
TIMING = OUT_C1 / "timing"
LOGFILE = OUT_C1 / "c1_run_v2.log"

EPOCHS = V5 / "epochs"
FS = 256.0
PAD = 256
C1_BANDS = [(0.5, 1.5), (1.0, 2.0), (1.5, 2.5), (2.0, 3.0)]


def log(msg):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    LOGFILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LOGFILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def filt(X, lo, hi):
    if lo is None or lo <= 0.5:
        sos = butter(4, hi, btype="lowpass", fs=FS, output="sos")
    else:
        sos = butter(4, [lo, hi], btype="bandpass", fs=FS, output="sos")
    return sosfiltfilt(sos, X, axis=-1, padtype="odd", padlen=PAD).astype(np.float32)


def c1_dims(n_ch):
    return (n_ch, 36, 36, min(n_ch, 22))


class TCSPNet(TorchNet):
    def __init__(self, seed=0, device="cuda", n_epochs=20):
        super().__init__("TensorCSPNet", n_epochs=n_epochs, seed=seed, device=device)

    @staticmethod
    def bank(X):
        return np.stack([filt(X, lo, hi) for lo, hi in C1_BANDS], axis=1)

    def _build(self, n_ch, n_cls, n_t):
        from spd_learn.models import TensorCSPNet
        return TensorCSPNet(n_chans=n_ch, n_outputs=n_cls, n_patches=4, n_freqs=len(C1_BANDS),
                            use_mlp=False, tcn_channels=16, dims=c1_dims(n_ch), momentum=0.1)

    def fit(self, X, y):
        import torch
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        self.classes_ = np.unique(y)
        yi = np.searchsorted(self.classes_, y)
        self.mu_ = X.mean(axis=(0, 2), keepdims=True)
        self.sd_ = X.std(axis=(0, 2), keepdims=True) + 1e-8
        t0 = time.time()
        Xb = self.bank(((X - self.mu_) / self.sd_).astype(np.float32))
        log(f"    filter bank train {Xb.shape} in {time.time() - t0:.1f}s")
        self.net_ = self._build(X.shape[1], len(self.classes_), X.shape[2]).to(self.device)
        cnt = np.bincount(yi, minlength=len(self.classes_)).astype(np.float32)
        w = torch.tensor(cnt.sum() / (len(cnt) * cnt), device=self.device)
        lossf = torch.nn.CrossEntropyLoss(weight=w)
        opt = torch.optim.AdamW(self.net_.parameters(), lr=self.lr, weight_decay=self.wd)
        steps = self.n_epochs * int(np.ceil(len(yi) / self.batch))
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
        Xt = torch.from_numpy(Xb)
        yt = torch.tensor(yi)
        self.loss_curve_ = []
        self.net_.train()
        t0 = time.time()
        for ep in range(self.n_epochs):
            if ep in (1, 5, 10, 15):
                log(f"    epoch {ep}/{self.n_epochs} at {time.time() - t0:.1f}s, last loss {loss.item():.4f}")
            order = rng.permutation(len(yi))
            for i in range(0, len(order), self.batch):
                b = torch.from_numpy(order[i:i + self.batch])
                if len(b) < 2:
                    continue
                opt.zero_grad()
                loss = lossf(self.net_(Xt[b].to(self.device)), yt[b].to(self.device))
                loss.backward()
                opt.step()
                sch.step()
        return self

    def predict(self, X):
        import torch
        Xb = self.bank(((X - self.mu_) / self.sd_).astype(np.float32))
        self.net_.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(Xb), 128):
                out.append(self.net_(torch.from_numpy(Xb[i:i + 128]).to(self.device)).argmax(1).cpu().numpy())
        return self.classes_[np.concatenate(out)]


def run_c1_stratum(st, max_splits=None):
    merged_file = PREDICTIONS / f"{st}__C1.csv"
    if merged_file.exists():
        log(f"Stratum {st} already merged ({merged_file}). Skipping.")
        return

    ckpt_dir = CHECKPOINTS / st
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    PREDICTIONS.mkdir(parents=True, exist_ok=True)
    TIMING.mkdir(parents=True, exist_ok=True)

    X, y, s = ev.load(EPOCHS, st)
    spl = ev.splits(s)
    timing_file = TIMING / "timing_C1.csv"

    t_strat0 = time.time()
    log(f"--- START Stratum {st} (n={len(s)} epochs, {X.shape[1]} ch) ---")

    for k, test_subj in enumerate(spl):
        if max_splits is not None and k >= max_splits:
            break
        split_file = ckpt_dir / f"split_{k}.csv"
        if split_file.exists():
            log(f"  {st} split {k} already exists. Skipping.")
            continue

        te = np.isin(s, list(test_subj))
        tr = ~te
        t0 = time.time()
        m = TCSPNet(seed=ev.SEED0 + k, device="cuda", n_epochs=20)
        m.fit(X[tr], y[tr])
        pred = m.predict(X[te])
        dur = round(time.time() - t0, 2)

        tmp_file = ckpt_dir / f"split_{k}.csv.tmp"
        df_split = pd.DataFrame({
            "split": k,
            "epoch": np.where(te)[0],
            "subject": s[te],
            "y": y[te],
            "pred": pred
        })
        df_split.to_csv(tmp_file, index=False)
        tmp_file.replace(split_file)

        df_t = pd.DataFrame([{
            "arm": "C1",
            "model": "TensorCSPNet",
            "stratum": st,
            "split": k,
            "seconds": dur
        }])
        df_t.to_csv(timing_file, mode="a", header=not timing_file.exists(), index=False)
        log(f"  {st} split {k}/9 done in {dur:.1f}s")

    # Merge splits
    parts = [ckpt_dir / f"split_{k}.csv" for k in range(ev.N_SPLITS)]
    if all(p.exists() for p in parts):
        df_all = pd.concat([pd.read_csv(p, dtype={"subject": str, "y": str, "pred": str}) for p in parts])
        df_all.to_csv(merged_file, index=False)
        total_strat_dur = round(time.time() - t_strat0, 1)
        log(f"--- DONE Stratum {st} merged to {merged_file.name} in {total_strat_dur:.1f}s ---")
    else:
        log(f"WARNING: Stratum {st} incomplete, merge skipped.")


ORDER = ["001-2020_gel", "001-2020_water", "001-2020_dry", "001-2019_offline",
         "001-2017_ME", "001-2017_MI"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strata", nargs="*", default=ORDER)
    ap.add_argument("--max-splits", type=int, default=None)
    a = ap.parse_args()
    assert set(a.strata) <= set(ev.STRATA)
    OUT_C1.mkdir(parents=True, exist_ok=True)
    crash = open(OUT_C1 / "c1_v2_faulthandler.txt", "a")
    faulthandler.enable(file=crash, all_threads=True)
    log(f"=== C1 GPU RUNNER v2 START (PID {os.getpid()}) strata={a.strata} max_splits={a.max_splits} ===")
    import torch
    log(f"PyTorch version: {torch.__version__}, CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        log(f"GPU device: {torch.cuda.get_device_name(0)}")
    try:
        for st in a.strata:
            run_c1_stratum(st, a.max_splits)
    except BaseException:
        import traceback
        log("ERROR:\n" + traceback.format_exc())
        raise
    log("=== C1 GPU RUNNER v2 FINISHED ===")


if __name__ == "__main__":
    main()
