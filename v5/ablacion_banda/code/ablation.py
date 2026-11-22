"""Band ablation on the v5 protocol (config v5-ablacion).

Commands:
  check_splits                         v5 splits reproduce splits_*.csv
  cross <arm> <stratum> [k ...]        train/predict; one checkpoint CSV per split
  merge <arm> <stratum>                merge split checkpoints into predictions/<stratum>__<arm>.csv
  a0check                              rerun EEGNet broadband on 001-2020_gel split 0, compare with v5
  c1sanity                             Tensor-CSPNet sanity checks
  permtrain <arm> <stratum> <n>        training-label permutation, resumable
  score                                summary table, kappa, Holm, permutation results
The v5 code is imported read-only; nothing is written outside ablacion_banda.
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt

V5 = Path(os.environ.get("TESIS_V5_ROOT", Path(__file__).resolve().parents[2]))  # repo: v5/ (original: D:/ICTQ/TESIS_pipeline_v5)
OUT = V5 / "ablacion_banda"
sys.path.insert(0, str(V5 / "code"))
import evaluate as ev  # noqa: E402
from models import FS, TorchNet  # noqa: E402

EPOCHS = V5 / "epochs"
PAD = 256
C1_BANDS = [(0.5, 1.5), (1.0, 2.0), (1.5, 2.5), (2.0, 3.0)]
ARMS = {"A0": "EEGNet", "A1": "EEGNet", "A2": "EEGNet", "B1": "MDM", "C1": "TensorCSPNet"}


def filt(X, lo, hi):
    """Butterworth order 4, zero phase, odd padding 1 s. lo=None -> low-pass."""
    if lo is None or lo <= 0.5:
        sos = butter(4, hi, btype="lowpass", fs=FS, output="sos")
    else:
        sos = butter(4, [lo, hi], btype="bandpass", fs=FS, output="sos")
    return sosfiltfilt(sos, X, axis=-1, padtype="odd", padlen=PAD).astype(np.float32)


def arm_input(arm, X):
    if arm == "A0":
        return X
    if arm in ("A1", "B1"):
        return filt(X, None, 3.0)
    if arm == "A2":
        return filt(X, 8.0, 30.0)
    if arm == "C1":
        return X  # filter bank applied inside the model wrapper, after z-scoring
    raise ValueError(arm)


# ------------------------------------------------------------------ models
class MDMLow:
    def __init__(self):
        from pyriemann.classification import MDM
        from pyriemann.estimation import Covariances
        from sklearn.pipeline import make_pipeline
        self.p = make_pipeline(Covariances(estimator="oas"), MDM(metric="riemann"))

    def fit(self, X, y):
        self.p.fit(X.astype(np.float64), y); return self

    def predict(self, X):
        return self.p.predict(X.astype(np.float64))


def c1_dims(n_ch):
    return (n_ch, 36, 36, min(n_ch, 22))


class TCSPNet(TorchNet):
    """Same training loop as the v5 TorchNet; input = z-scored epoch passed through the
    0.5-3 Hz filter bank, shape (n, 4, C, T). Data stay on CPU; batches go to the GPU."""

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
        Xb = self.bank(((X - self.mu_) / self.sd_).astype(np.float32))
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
        for _ in range(self.n_epochs):
            order = rng.permutation(len(yi)); tot = 0.0
            for i in range(0, len(order), self.batch):
                b = torch.from_numpy(order[i:i + self.batch])
                if len(b) < 2:
                    continue  # batch norm needs >= 2 samples
                opt.zero_grad()
                loss = lossf(self.net_(Xt[b].to(self.device)), yt[b].to(self.device))
                loss.backward(); opt.step(); sch.step()
                tot += float(loss) * len(b)
            self.loss_curve_.append(tot / len(yi))
        return self

    def predict(self, X):
        import torch
        Xb = self.bank(((X - self.mu_) / self.sd_).astype(np.float32))
        self.net_.eval(); out = []
        with torch.no_grad():
            for i in range(0, len(Xb), 128):
                out.append(self.net_(torch.from_numpy(Xb[i:i + 128]).to(self.device)).argmax(1).cpu().numpy())
        return self.classes_[np.concatenate(out)]


def make(arm, n_ch, seed):
    if arm in ("A0", "A1", "A2"):
        return TorchNet("EEGNet", seed=seed)
    if arm == "B1":
        return MDMLow()
    if arm == "C1":
        return TCSPNet(seed=seed)
    raise ValueError(arm)


def fit_predict(arm, X, y, s, test_set, seed, permute_rng=None):
    te = np.isin(s, list(test_set)); tr = ~te
    ytr = y[tr] if permute_rng is None else ev.permute_within(y[tr], s[tr], permute_rng)
    m = make(arm, X.shape[1], seed)
    m.fit(X[tr], ytr)
    return te, m.predict(X[te]), m


# ------------------------------------------------------------------ commands
def load(arm, stratum):
    X, y, s = ev.load(EPOCHS, stratum)
    return arm_input(arm, X), y, s


def check_splits():
    for st in ev.STRATA:
        z = np.load(EPOCHS / f"{st}.npz", allow_pickle=False)
        s = z["subject"][z["finite"]].astype(str)
        mine = [" ".join(sorted(t)) for t in ev.splits(s)]
        ref = pd.read_csv(V5 / "results" / f"splits_{st}.csv", dtype=str).test_subjects.tolist()
        if mine != ref:
            raise SystemExit(f"split mismatch {st}")
        print("splits ok", st, flush=True)


def cross(arm, stratum, ks=None):
    d = OUT / "predictions" / arm / stratum; d.mkdir(parents=True, exist_ok=True)
    ks = list(range(ev.N_SPLITS)) if ks is None else ks
    todo = [k for k in ks if not (d / f"split_{k}.csv").exists()]
    if not todo:
        print("skip", arm, stratum, flush=True); return
    X, y, s = load(arm, stratum)
    spl = ev.splits(s)
    tf = OUT / "timing" / f"timing_{arm}.csv"; tf.parent.mkdir(exist_ok=True)
    for k in todo:
        t0 = time.time()
        te, pred, _ = fit_predict(arm, X, y, s, spl[k], seed=ev.SEED0 + k)
        sec = time.time() - t0
        tmp = d / f"split_{k}.csv.tmp"
        pd.DataFrame({"split": k, "epoch": np.where(te)[0], "subject": s[te], "y": y[te], "pred": pred}
                     ).to_csv(tmp, index=False)
        tmp.replace(d / f"split_{k}.csv")
        pd.DataFrame([{"arm": arm, "model": ARMS[arm], "stratum": stratum, "split": k,
                       "seconds": round(sec, 2)}]).to_csv(tf, mode="a", header=not tf.exists(), index=False)
        print(arm, stratum, "split", k, f"{sec:.1f}s", flush=True)


def merge(arm, stratum):
    d = OUT / "predictions" / arm / stratum
    parts = [d / f"split_{k}.csv" for k in range(ev.N_SPLITS)]
    if not all(p.exists() for p in parts):
        raise SystemExit(f"incomplete {arm} {stratum}")
    f = OUT / "predictions" / f"{stratum}__{arm}.csv"
    pd.concat([pd.read_csv(p, dtype={"subject": str, "y": str, "pred": str}) for p in parts]).to_csv(f, index=False)


def a0check():
    st = "001-2020_gel"
    X, y, s = load("A0", st)
    spl = ev.splits(s)
    te, pred, _ = fit_predict("A0", X, y, s, spl[0], seed=ev.SEED0)
    ref = pd.read_csv(V5 / "results" / "predictions" / f"{st}__EEGNet.csv", dtype={"pred": str})
    ref = ref[ref.split == 0].sort_values("epoch")
    same_idx = np.array_equal(ref.epoch.to_numpy(), np.where(te)[0])
    agree = float(np.mean(ref.pred.to_numpy(dtype=str) == pred.astype(str))) if same_idx else float("nan")
    res = {"stratum": st, "split": 0, "same_test_epochs": bool(same_idx), "prediction_agreement": agree,
           "identical": bool(same_idx and agree == 1.0)}
    (OUT / "checks").mkdir(exist_ok=True)
    (OUT / "checks" / "a0_reproducibility.json").write_text(json.dumps(res, indent=2))
    print(res, flush=True)
    return res["identical"]


def c1sanity():
    import torch
    res = {}
    for n_ch in (11, 61):
        net = TCSPNet()._build(n_ch, 3, 513).cuda().eval()
        with torch.no_grad():
            out = net(torch.randn(8, 4, n_ch, 513, device="cuda"))
        res[f"forward_{n_ch}ch"] = {"shape": list(out.shape), "finite": bool(torch.isfinite(out).all())}
    # synthetic low-frequency two-class problem, 11 channels
    rng = np.random.default_rng(0)
    n, C, T = 400, 11, 513
    t = np.arange(T) / FS
    yy = rng.integers(0, 2, n)
    pat = rng.standard_normal((2, C))
    X = rng.standard_normal((n, C, T)).astype(np.float32) * 0.5
    X += (pat[yy][:, :, None] * np.sin(2 * np.pi * 1.5 * t)[None, None, :]).astype(np.float32)
    m = TCSPNet(seed=0, n_epochs=10).fit(X[:300], yy[:300].astype(str))
    res["synthetic_ba"] = ev.bal_acc(yy[300:].astype(str), m.predict(X[300:]))
    # real data: loss decreases, 001-2020_gel split 0
    Xr, y, s = load("C1", "001-2020_gel")
    te, pred, m = fit_predict("C1", Xr, y, s, ev.splits(s)[0], seed=ev.SEED0)
    res["gel_split0_loss_first_last"] = [m.loss_curve_[0], m.loss_curve_[-1]]
    res["gel_split0_ba"] = float(np.mean(list(ev.subject_ba(y[te], pred, s[te]).values())))
    res["passed"] = bool(all(v["finite"] and v["shape"] == [8, 3] for k, v in res.items() if k.startswith("forward"))
                         and res["synthetic_ba"] >= 0.9 and m.loss_curve_[-1] < m.loss_curve_[0])
    (OUT / "checks").mkdir(exist_ok=True)
    (OUT / "checks" / "c1_sanity.json").write_text(json.dumps(res, indent=2))
    print(res, flush=True)
    return res["passed"]


def permtrain(arm, stratum, n_perm):
    X, y, s = load(arm, stratum)
    spl = ev.splits(s)
    d = OUT / "permtrain"; d.mkdir(exist_ok=True)
    f = d / f"{stratum}__{arm}_null.csv"
    done = set(pd.read_csv(f).perm) if f.exists() else set()
    for b in [b for b in range(n_perm) if b not in done]:
        rng = np.random.default_rng(10_000 + b)  # same permutation seeds as v5
        sc = []
        for k, t in enumerate(spl):
            te, pred, _ = fit_predict(arm, X, y, s, t, seed=ev.SEED0 + k, permute_rng=rng)
            sc.append(np.mean(list(ev.subject_ba(y[te], pred, s[te]).values())))
        pd.DataFrame({"perm": [b], "ba": [float(np.mean(sc))]}).to_csv(f, mode="a", header=not f.exists(), index=False)
        print(stratum, arm, "perm", b + 1, "/", n_perm, flush=True)


# ------------------------------------------------------------------ scoring
def kappa(y, p):
    y, p = np.asarray(y, str), np.asarray(p, str)
    cl = np.unique(np.concatenate([y, p]))
    po = np.mean(y == p)
    pe = sum(np.mean(y == c) * np.mean(p == c) for c in cl)
    return float((po - pe) / (1 - pe)) if pe < 1 else 0.0


def participant_scores(f):
    d = pd.read_csv(f, dtype={"subject": str, "y": str, "pred": str})
    ba, ka = {}, {}
    for (_, u), g in d.groupby(["split", "subject"]):
        ba.setdefault(u, []).append(ev.bal_acc(g.y, g.pred)); ka.setdefault(u, []).append(kappa(g.y, g.pred))
    return ({u: np.mean(v) for u, v in ba.items()}, {u: np.mean(v) for u, v in ka.items()},
            d.y.nunique(), d)


def signflip(a, b, n=10000, seed=0):
    u = sorted(set(a) & set(b)); dlt = np.array([a[k] - b[k] for k in u])
    rng = np.random.default_rng(seed)
    null = np.abs((rng.choice([-1, 1], (n, len(dlt))) * dlt).mean(1))
    return float(dlt.mean()), float((1 + (null >= abs(dlt.mean())).sum()) / (1 + n)), len(u)


def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); m = len(p)
    adj = np.maximum.accumulate([(m - i) * p[j] for i, j in enumerate(o)])
    out = np.empty(m); out[o] = np.minimum(adj, 1.0); return out


def score():
    rows, part = [], {}
    pred_dir = OUT / "predictions"
    a0_reuse = json.loads((OUT / "checks" / "a0_reproducibility.json").read_text()).get("identical", False) \
        if (OUT / "checks" / "a0_reproducibility.json").exists() else False
    for st in ev.STRATA:
        for arm in ARMS:
            f = pred_dir / f"{st}__{arm}.csv"
            src = "ablation"
            if arm == "A0" and not f.exists() and a0_reuse:
                f, src = V5 / "results" / "predictions" / f"{st}__EEGNet.csv", "v5 (reused)"
            if not f.exists():
                continue
            r = ev.score_predictions(f)
            ba, ka, ncls, d = participant_scores(f)
            part[(st, arm)] = ba
            kap_split = [np.mean([kappa(g.y, g.pred) for _, g in gs.groupby("subject")]) for _, gs in d.groupby("split")]
            rows.append({"stratum": st, "arm": arm, "model": ARMS[arm], "source": src, "chance_ba": 1 / ncls,
                         **r, "kappa_mean_splits": float(np.mean(kap_split)),
                         "kappa_participant_mean": float(np.mean(list(ka.values())))})
        for ref, name in (("MDM", "REF_MDM_8-30_v5"),):
            f = V5 / "results" / "predictions" / f"{st}__{ref}.csv"
            if f.exists():
                r = ev.score_predictions(f); _, ka, ncls, d = participant_scores(f)
                rows.append({"stratum": st, "arm": name, "model": ref, "source": "v5 (reference, not tested)",
                             "chance_ba": 1 / ncls, **r, "kappa_participant_mean": float(np.mean(list(ka.values())))})
    df = pd.DataFrame(rows)
    df["ci95_participant_boot"] = df["ci95_participant_boot"].map(lambda v: f"{v[0]:.3f}-{v[1]:.3f}")
    fam = []
    for _, r in df[~df.arm.str.startswith("REF")].iterrows():
        fam.append({"test": f"{r.arm} vs chance", "stratum": r.stratum, "estimate": r.ba_mean_splits,
                    "p": r.p_test_label_perm, "method": "test-label permutation (1000)"})
    for st in ev.STRATA:
        for other in ("A2", "A0"):
            if (st, "A1") in part and (st, other) in part:
                m, p, n = signflip(part[(st, "A1")], part[(st, other)])
                fam.append({"test": f"A1 vs {other}", "stratum": st, "estimate": m, "p": p,
                            "method": f"paired sign-flip (10000), n={n} participants"})
    fam = pd.DataFrame(fam); fam["p_holm"] = holm(fam.p); fam["family_size"] = len(fam)
    pt = []
    for f in sorted((OUT / "permtrain").glob("*_null.csv")) if (OUT / "permtrain").exists() else []:
        st, arm = f.stem.replace("_null", "").split("__")
        null = pd.read_csv(f).drop_duplicates("perm").ba.to_numpy()
        obs = df[(df.stratum == st) & (df.arm == arm)].ba_mean_splits
        if len(obs):
            o = float(obs.iloc[0])
            pt.append({"stratum": st, "arm": arm, "n_perm": len(null), "observed": o, "null_mean": null.mean(),
                       "null_95": np.percentile(null, 95), "null_max": null.max(),
                       "p_train_label_perm": (1 + (null >= o).sum()) / (1 + len(null))})
    (OUT / "summary").mkdir(exist_ok=True)
    df.to_csv(OUT / "summary" / "ablation_summary.csv", index=False)
    fam.to_csv(OUT / "summary" / "ablation_holm_family.csv", index=False)
    pd.DataFrame(pt).to_csv(OUT / "summary" / "ablation_permtrain.csv", index=False)
    print(df[["stratum", "arm", "ba_mean_splits", "kappa_mean_splits", "p_test_label_perm"]].to_string(), flush=True)
    print(fam.to_string(), flush=True)


if __name__ == "__main__":
    c = sys.argv[1]
    if c == "check_splits":
        check_splits()
    elif c == "cross":
        cross(sys.argv[2], sys.argv[3], [int(k) for k in sys.argv[4:]] or None)
    elif c == "merge":
        merge(sys.argv[2], sys.argv[3])
    elif c == "a0check":
        sys.exit(0 if a0check() else 3)
    elif c == "c1sanity":
        sys.exit(0 if c1sanity() else 3)
    elif c == "permtrain":
        permtrain(sys.argv[2], sys.argv[3], int(sys.argv[4]))
    elif c == "score":
        score()
