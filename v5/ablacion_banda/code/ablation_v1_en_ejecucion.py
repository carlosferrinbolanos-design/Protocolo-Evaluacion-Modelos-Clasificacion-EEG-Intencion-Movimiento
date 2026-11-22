"""Band ablation on the v5 protocol (config ../v5-ablacion.json).

Commands (run with ~/.venvs/tesis5, from this folder):
  smoke                 one 2020 split per arm (quick check)
  cross                 A1, A2, B1 on all six strata (resumable)
  permtrain             200 training-label permutations, A1 and A2, 2020 strata only (resumable)
  score                 metrics, kappa, contrasts, Holm -> ../results/ablation_summary*.csv
  all                   cross + permtrain + score
Reuses the v5 code read-only (epochs, splits, seeds, models, scoring).
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
sys.path.insert(0, str(V5 / "code"))
import evaluate as ev  # noqa: E402  (v5 code, read-only)
from models import make_model, make_classical  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "results"
EPOCHS = V5 / "epochs"
FS = 256.0

ARMS = {  # arm -> (v5 model name, band)
    "A1_EEGNet_low": ("EEGNet", "low"),
    "A2_EEGNet_mu_beta": ("EEGNet", "mu_beta"),
    "B1_MDM_low": ("MDM_low", "low"),
}


def band(X, kind):
    if kind == "low":
        sos = butter(4, 3.0, btype="lowpass", fs=FS, output="sos")
    else:
        sos = butter(4, [8.0, 30.0], btype="bandpass", fs=FS, output="sos")
    return sosfiltfilt(sos, X, axis=-1).astype(np.float32)


def mdm_low():
    from pyriemann.classification import MDM
    from pyriemann.estimation import Covariances
    from sklearn.pipeline import make_pipeline
    return make_pipeline(Covariances(estimator="oas"), MDM(metric="riemann"))  # input already 0.5-3 Hz


def fit_predict(arm, X, y, s, test_set, seed, permute_rng=None):
    model, _ = ARMS[arm]
    te = np.isin(s, list(test_set)); tr = ~te
    ytr = y[tr] if permute_rng is None else ev.permute_within(y[tr], s[tr], permute_rng)
    m = mdm_low() if model == "MDM_low" else make_model(model, X.shape[1], seed=seed)
    m.fit(X[tr], ytr)
    return te, m.predict(X[te])


def load_band(st, kind):
    X, y, s = ev.load(EPOCHS, st)
    return band(X, kind), y, s


def log(msg):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(OUT / "ablation.log", "a", encoding="utf-8") as f:
        f.write(line + "\n")


def cmd_cross(strata=ev.STRATA, max_splits=None, tag=""):
    (OUT / "predictions").mkdir(parents=True, exist_ok=True)
    timing = []
    for st in strata:
        cache = {}
        spl = ev.splits(ev.load(EPOCHS, st)[2])
        for arm, (model, kind) in ARMS.items():
            f = OUT / "predictions" / f"{st}__{arm}{tag}.csv"
            if f.exists():
                log(f"skip {st} {arm}{tag}"); continue
            if kind not in cache:
                cache[kind] = load_band(st, kind)
            X, y, s = cache[kind]
            parts = []
            for k, t in enumerate(spl[:max_splits] if max_splits else spl):
                t0 = time.time()
                te, pred = fit_predict(arm, X, y, s, t, seed=ev.SEED0 + k)
                idx = np.where(te)[0]
                parts.append(pd.DataFrame({"split": k, "epoch": idx, "subject": s[te], "y": y[te], "pred": pred}))
                timing.append({"stratum": st, "arm": arm, "split": k, "seconds": round(time.time() - t0, 1)})
            pd.concat(parts).to_csv(f, index=False)
            log(f"done {st} {arm}{tag}")
        if timing:
            pd.DataFrame(timing).to_csv(OUT / f"timing_{tag or 'cross'}_{int(time.time())}.csv", index=False)
            timing = []


def _perm_rep(arm, X, y, s, spl, b):
    rng = np.random.default_rng(10_000 + b)
    sc = []
    for k, t in enumerate(spl):
        te, pred = fit_predict(arm, X, y, s, t, seed=ev.SEED0 + k, permute_rng=rng)
        sc.append(np.mean(list(ev.subject_ba(y[te], pred, s[te]).values())))
    return float(np.mean(sc))


def cmd_permtrain(n_perm=200):
    d = OUT / "permtrain"; d.mkdir(exist_ok=True)
    for st in ["001-2020_gel", "001-2020_water", "001-2020_dry"]:
        for arm in ["A1_EEGNet_low", "A2_EEGNet_mu_beta"]:
            X, y, s = load_band(st, ARMS[arm][1])
            spl = ev.splits(s)
            f = d / f"{st}__{arm}_null.csv"
            done = set(pd.read_csv(f).perm) if f.exists() else set()
            for b in range(n_perm):
                if b in done:
                    continue
                v = _perm_rep(arm, X, y, s, spl, b)
                pd.DataFrame({"perm": [b], "ba": [v]}).to_csv(f, mode="a", header=not f.exists(), index=False)
                if (b + 1) % 10 == 0:
                    log(f"permtrain {st} {arm} {b + 1}/{n_perm}")
            log(f"permtrain done {st} {arm}")


def kappa(y, p):
    y, p = np.asarray(y, str), np.asarray(p, str)
    po = np.mean(y == p)
    pe = sum(np.mean(y == c) * np.mean(p == c) for c in np.unique(np.concatenate([y, p])))
    return float((po - pe) / (1 - pe)) if pe < 1 else 0.0


def participant_ba(pred_csv):
    d = pd.read_csv(pred_csv, dtype={"subject": str, "y": str, "pred": str})
    per = {}
    for k, g in d.groupby("split"):
        for u, v in ev.subject_ba(g.y.to_numpy(), g.pred.to_numpy(), g.subject.to_numpy()).items():
            per.setdefault(u, []).append(v)
    return pd.Series({u: np.mean(v) for u, v in per.items()})


def signflip_p(diff, n=10000, seed=0):
    diff = np.asarray(diff); rng = np.random.default_rng(seed)
    obs = abs(diff.mean())
    null = np.abs((rng.choice([-1, 1], size=(n, len(diff))) * diff).mean(axis=1))
    return float((1 + (null >= obs).sum()) / (1 + n))


def holm(p):
    p = np.asarray(p); o = np.argsort(p); m = len(p); adj = np.empty(m); run = 0.0
    for i, j in enumerate(o):
        run = max(run, min(1.0, (m - i) * p[j])); adj[j] = run
    return adj


def cmd_score():
    rows = []
    for st in ev.STRATA:
        for arm in ARMS:
            f = OUT / "predictions" / f"{st}__{arm}.csv"
            if not f.exists():
                continue
            r = ev.score_predictions(f)
            d = pd.read_csv(f, dtype={"subject": str, "y": str, "pred": str})
            ks = [np.mean([kappa(gg.y, gg.pred) for _, gg in g.groupby("subject")]) for _, g in d.groupby("split")]
            rows.append({"stratum": st, "arm": arm, "ba_mean_splits": r["ba_mean_splits"],
                         "ba_ci95_lo": r["ci95_participant_boot"][0], "ba_ci95_hi": r["ci95_participant_boot"][1],
                         "kappa_mean_splits": float(np.mean(ks)), "chance": 1 / d.y.nunique(),
                         "n_participants": r["n_participants_tested"], "p_raw": r["p_test_label_perm"],
                         "test": "arm vs chance (test-label permutation)"})
        a1 = OUT / "predictions" / f"{st}__A1_EEGNet_low.csv"
        for other, ref in [("A2_EEGNet_mu_beta", OUT / "predictions" / f"{st}__A2_EEGNet_mu_beta.csv"),
                           ("A0_EEGNet_broadband_v5", V5 / "results" / "predictions" / f"{st}__EEGNet.csv")]:
            if a1.exists() and ref.exists():
                x, z = participant_ba(a1), participant_ba(ref)
                common = x.index.intersection(z.index)
                diff = (x[common] - z[common]).to_numpy()
                rows.append({"stratum": st, "arm": f"A1_minus_{other}", "ba_mean_splits": float(diff.mean()),
                             "n_participants": len(common), "p_raw": signflip_p(diff),
                             "test": "paired sign-flip permutation, participant level"})
    t = pd.DataFrame(rows)
    t["holm_family_size"] = len(t)
    t["p_holm"] = holm(t.p_raw.to_numpy())
    t.round(4).to_csv(OUT / "ablation_summary.csv", index=False)
    pt = []
    for f in sorted((OUT / "permtrain").glob("*_null.csv")):
        st, arm = f.stem.replace("_null", "").split("__")
        null = pd.read_csv(f).drop_duplicates("perm").ba.to_numpy()
        obs = t[(t.stratum == st) & (t.arm == arm)].ba_mean_splits
        if len(obs):
            o = float(obs.iloc[0])
            pt.append({"stratum": st, "arm": arm, "n_perm": len(null), "observed": o, "null_mean": null.mean(),
                       "null_95": np.percentile(null, 95), "null_max": null.max(),
                       "p_train_label_perm": (1 + (null >= o).sum()) / (1 + len(null))})
    pd.DataFrame(pt).round(4).to_csv(OUT / "ablation_permtrain.csv", index=False)
    print(t.round(3).to_string()); print(pd.DataFrame(pt).round(3).to_string())


if __name__ == "__main__":
    c = sys.argv[1]
    OUT.mkdir(parents=True, exist_ok=True)
    if c == "smoke":
        OUT_SMOKE = OUT / "smoke"; OUT_SMOKE.mkdir(exist_ok=True)
        OUT = OUT_SMOKE  # noqa: F811
        cmd_cross(strata=["001-2020_gel"], max_splits=1, tag="_smoke")
    elif c == "cross":
        cmd_cross()
    elif c == "permtrain":
        cmd_permtrain()
    elif c == "score":
        cmd_score()
    elif c == "all":
        cmd_cross(); cmd_permtrain(); cmd_score()
        log("ALL DONE")
