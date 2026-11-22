"""Cross-subject evaluation and within-subject sanity check (pipeline v5).

Commands:
  sanity   <epochs_dir> <out_dir>            within-subject control, every stratum
  cross    <epochs_dir> <out_dir> <models>   10 participant-disjoint 80/20 splits
  permtrain <epochs_dir> <out_dir> <stratum> <model> <n_perm>
                                             refit with training labels permuted within
                                             participant (same 10 splits)

Design fixed before running (Carlos, 2026-11-14):
  * Splits: 10 random participant-disjoint splits, round(20 %) of participants in test,
    seeds 20261114 + k. Same splits for every model.
  * Metric: balanced accuracy computed per test participant, averaged over test participants
    (split score) and over splits (model score). Per-epoch predictions are saved.
  * Inference: (a) test-label permutation within participant, 1000 draws, p for the
    split-averaged score; (b) 200 refits with training labels permuted within participant
    for selected pipelines; (c) participant bootstrap 95 % CI of the participant-level score.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).parent))
from models import CLASSICAL, Bandpass, make_classical, make_model  # noqa: E402

STRATA = ["001-2017_ME", "001-2017_MI", "001-2019_offline",
          "001-2020_gel", "001-2020_water", "001-2020_dry"]
N_SPLITS, SEED0 = 10, 20261114


def load(epochs_dir, stratum):
    z = np.load(Path(epochs_dir) / f"{stratum}.npz", allow_pickle=False)
    keep = z["finite"]
    return (z["X"][keep].astype(np.float32), z["y"][keep].astype(str),
            z["subject"][keep].astype(str))


def splits(subjects):
    subs = np.array(sorted(set(subjects)))
    n_test = max(1, int(round(0.2 * len(subs))))
    out = []
    for k in range(N_SPLITS):
        rng = np.random.default_rng(SEED0 + k)
        out.append(set(rng.choice(subs, n_test, replace=False).tolist()))
    return out


def bal_acc(y, p):
    """Balanced accuracy (mean recall over classes present in y), same definition as sklearn.
    Own implementation: sklearn 1.9.1 + numpy 2.5.4 segfault on pandas object arrays."""
    y, p = np.asarray(y, dtype=str), np.asarray(p, dtype=str)
    return float(np.mean([np.mean(p[y == c] == c) for c in np.unique(y)]))


def subject_ba(y, pred, s):
    y, pred, s = (np.asarray(v, dtype=str) for v in (y, pred, s))
    return {u: bal_acc(y[s == u], pred[s == u]) for u in sorted(set(s))}


def permute_within(y, s, rng):
    yp = y.copy()
    for u in np.unique(s):
        i = np.where(s == u)[0]
        yp[i] = rng.permutation(yp[i])
    return yp


def fit_predict(model, X, y, s, test_set, seed, permute_rng=None):
    te = np.isin(s, list(test_set))
    tr = ~te
    ytr = y[tr] if permute_rng is None else permute_within(y[tr], s[tr], permute_rng)
    m = make_model(model, X.shape[1], seed=seed)
    m.fit(X[tr], ytr)
    return te, m.predict(X[te])


# ---------------------------------------------------------------- sanity (within subject)
def mrcp_features(X):
    """0.5-3 Hz, decimated to 16 Hz: time-domain MRCP amplitudes (Ofner et al. 2017, 2019)."""
    Xf = Bandpass(0.5, 3.0).transform(X)
    return Xf[:, :, ::16].reshape(len(X), -1)


def sanity_subject(X, y, model):
    skf = StratifiedKFold(5, shuffle=False)  # contiguous, chronological blocks
    pred = np.empty_like(y)
    for tr, te in skf.split(X, y):
        if model == "MRCP_sLDA":
            clf = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
            clf.fit(mrcp_features(X[tr]), y[tr]); pred[te] = clf.predict(mrcp_features(X[te]))
        else:
            clf = make_classical(model, X.shape[1]); clf.fit(X[tr], y[tr]); pred[te] = clf.predict(X[te])
    return balanced_accuracy_score(y, pred)


def cmd_sanity(epochs_dir, out_dir):
    rows = []
    for st in STRATA:
        X, y, s = load(epochs_dir, st)
        subs = sorted(set(s))
        for model in ("MRCP_sLDA", "CSP_LDA"):
            res = Parallel(n_jobs=14)(delayed(sanity_subject)(X[s == u], y[s == u], model) for u in subs)
            for u, ba in zip(subs, res):
                rows.append({"stratum": st, "model": model, "subject": u, "balanced_accuracy": ba,
                             "chance": 1 / len(set(y)), "n_epochs": int((s == u).sum())})
            print(st, model, np.round(np.mean(res), 3), flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(Path(out_dir) / "sanity_within_subject_by_subject.csv", index=False)
    summ = df.groupby(["stratum", "model"]).agg(mean_ba=("balanced_accuracy", "mean"),
                                                sd_ba=("balanced_accuracy", "std"),
                                                min_ba=("balanced_accuracy", "min"),
                                                max_ba=("balanced_accuracy", "max"),
                                                chance=("chance", "first"),
                                                n_subjects=("subject", "count")).reset_index()
    summ.to_csv(Path(out_dir) / "sanity_within_subject_summary.csv", index=False)
    print(summ.to_string())


# ---------------------------------------------------------------- cross-subject
def _one_split(model, X, y, s, k, test_set):
    t = time.time()
    te, pred = fit_predict(model, X, y, s, test_set, seed=SEED0 + k)
    return k, te, pred, time.time() - t


def cmd_cross(epochs_dir, out_dir, models):
    out_dir = Path(out_dir); (out_dir / "predictions").mkdir(parents=True, exist_ok=True)
    rows = []
    for st in STRATA:
        X, y, s = load(epochs_dir, st)
        spl = splits(s)
        pd.DataFrame([{"split": k, "test_subjects": " ".join(sorted(t))} for k, t in enumerate(spl)]
                     ).to_csv(out_dir / f"splits_{st}.csv", index=False)
        for model in models:
            f = out_dir / "predictions" / f"{st}__{model}.csv"
            if f.exists():
                print("skip", st, model, flush=True); continue
            if model in CLASSICAL:
                res = Parallel(n_jobs=3, max_nbytes="1M")(delayed(_one_split)(model, X, y, s, k, t) for k, t in enumerate(spl))
            else:
                res = [_one_split(model, X, y, s, k, t) for k, t in enumerate(spl)]
            pr = []
            for k, te, pred, sec in res:
                idx = np.where(te)[0]
                pr.append(pd.DataFrame({"split": k, "epoch": idx, "subject": s[te], "y": y[te], "pred": pred}))
                rows.append({"stratum": st, "model": model, "split": k, "seconds": round(sec, 1)})
            pd.concat(pr).to_csv(f, index=False)
            print(st, model, "done", flush=True)
    pd.DataFrame(rows).to_csv(out_dir / f"timing_{int(time.time())}.csv", index=False)


def score_predictions(pred_csv, n_perm=1000, n_boot=2000, seed=0):
    d = pd.read_csv(pred_csv, dtype={"subject": str, "y": str, "pred": str})
    rng = np.random.default_rng(seed)
    split_scores, subj = [], {}
    null = np.zeros(n_perm)
    for k, g in d.groupby("split"):
        y, p, s = (g[c].to_numpy(dtype=str) for c in ("y", "pred", "subject"))
        ba = subject_ba(y, p, s)
        split_scores.append(np.mean(list(ba.values())))
        for u, v in ba.items():
            subj.setdefault(u, []).append(v)
        for b in range(n_perm):
            yp = permute_within(y, s, rng)
            null[b] += np.mean(list(subject_ba(yp, p, s).values()))
    null /= d.split.nunique()
    obs = float(np.mean(split_scores))
    subj_mean = np.array([np.mean(v) for v in subj.values()])
    boot = [rng.choice(subj_mean, len(subj_mean)).mean() for _ in range(n_boot)]
    return {"ba_mean_splits": obs, "ba_sd_splits": float(np.std(split_scores, ddof=1)),
            "ba_min_split": float(np.min(split_scores)), "ba_max_split": float(np.max(split_scores)),
            "ba_participant_mean": float(subj_mean.mean()),
            "ci95_participant_boot": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
            "n_participants_tested": int(len(subj_mean)),
            "null_test_perm_mean": float(null.mean()), "null_test_perm_95": float(np.percentile(null, 95)),
            "p_test_label_perm": float((1 + (null >= obs).sum()) / (1 + n_perm))}


def cmd_score(out_dir):
    rows = []
    for f in sorted((Path(out_dir) / "predictions").glob("*.csv")):
        st, model = f.stem.split("__")
        r = {"stratum": st, "model": model, **score_predictions(f)}
        rows.append(r); print(st, model, round(r["ba_mean_splits"], 3), r["p_test_label_perm"], flush=True)
    df = pd.DataFrame(rows)
    df["ci95_participant_boot"] = df["ci95_participant_boot"].map(lambda v: f"{v[0]:.3f}-{v[1]:.3f}")
    df.to_csv(Path(out_dir) / "cross_subject_summary.csv", index=False)


def _perm_rep(model, X, y, s, spl, b):
    rng = np.random.default_rng(10_000 + b)
    scores = []
    for k, t in enumerate(spl):
        te, pred = fit_predict(model, X, y, s, t, seed=SEED0 + k, permute_rng=rng)
        scores.append(np.mean(list(subject_ba(y[te], pred, s[te]).values())))
    return float(np.mean(scores))


def cmd_permtrain(epochs_dir, out_dir, stratum, model, n_perm):
    """Resumable: each finished permutation is appended to <stratum>__<model>_null.csv."""
    X, y, s = load(epochs_dir, stratum)
    spl = splits(s)
    jobs = 4 if model in CLASSICAL else 1  # parallel CUDA processes crash on Windows (access violation)
    d = Path(out_dir) / "permtrain"; d.mkdir(exist_ok=True)
    f = d / f"{stratum}__{model}_null.csv"
    done = pd.read_csv(f) if f.exists() else pd.DataFrame(columns=["perm", "ba"])
    todo = [b for b in range(n_perm) if b not in set(done.perm)]
    with Parallel(n_jobs=jobs) as par:
        for i in range(0, len(todo), jobs):
            chunk = todo[i:i + jobs]
            vals = par(delayed(_perm_rep)(model, X, y, s, spl, b) for b in chunk)
            pd.DataFrame({"perm": chunk, "ba": vals}).to_csv(f, mode="a", header=not f.exists(), index=False)
            print(stratum, model, "perm", chunk[-1] + 1, "/", n_perm, flush=True)
    null = pd.read_csv(f).drop_duplicates("perm").ba.to_numpy()
    obs = pd.read_csv(Path(out_dir) / "cross_subject_summary.csv")
    obs = float(obs[(obs.stratum == stratum) & (obs.model == model)].ba_mean_splits.iloc[0])
    res = {"stratum": stratum, "model": model, "n_perm": int(len(null)), "observed": obs,
           "null_mean": float(null.mean()), "null_95": float(np.percentile(null, 95)),
           "null_max": float(null.max()), "p_train_label_perm": float((1 + (null >= obs).sum()) / (1 + len(null)))}
    (d / f"{stratum}__{model}.json").write_text(json.dumps(res, indent=2))
    print(res)


if __name__ == "__main__":
    c = sys.argv[1]
    if c == "sanity":
        cmd_sanity(sys.argv[2], sys.argv[3])
    elif c == "cross":
        cmd_cross(sys.argv[2], sys.argv[3], sys.argv[4].split(","))
    elif c == "score":
        cmd_score(sys.argv[2])
    elif c == "permtrain":
        cmd_permtrain(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], int(sys.argv[6]))
