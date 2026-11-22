import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd

V5 = Path(os.environ.get("TESIS_V5_ROOT", Path(__file__).resolve().parents[3]))  # repo: v5/ (original: D:/ICTQ/TESIS_pipeline_v5)
sys.path.insert(0, str(V5 / "code"))
import evaluate as ev  # noqa: E402

AB = V5 / "ablacion_banda" / "results"


def kappa(y, p):
    y, p = np.asarray(y, str), np.asarray(p, str)
    po = np.mean(y == p)
    pe = sum(np.mean(y == c) * np.mean(p == c) for c in np.unique(np.concatenate([y, p])))
    return float((po - pe) / (1 - pe)) if pe < 1 else 0.0


def score(f):
    r = ev.score_predictions(f, n_perm=1000, n_boot=2000, seed=0)
    d = pd.read_csv(f, dtype={"subject": str, "y": str, "pred": str})
    ks = [np.mean([kappa(gg.y, gg.pred) for _, gg in g.groupby("subject")]) for _, g in d.groupby("split")]
    k = int(round(r["p_test_label_perm"] * 1001))
    return {"ba_mean_splits": r["ba_mean_splits"], "ba_sd_splits": r["ba_sd_splits"],
            "ba_min_split": r["ba_min_split"], "ba_max_split": r["ba_max_split"],
            "ci95_lo": r["ci95_participant_boot"][0], "ci95_hi": r["ci95_participant_boot"][1],
            "kappa_mean_splits": float(np.mean(ks)), "chance": 1 / d.y.nunique(),
            "n_participants": r["n_participants_tested"], "null_95": r["null_test_perm_95"],
            "p_frac": f"{k}/1001", "p_dec": r["p_test_label_perm"]}


rows = []
for st in ["001-2017_ME", "001-2017_MI", "001-2019_offline", "001-2020_gel", "001-2020_water", "001-2020_dry"]:
    rows.append({"stratum": st, **score(AB / "c1" / "predictions" / f"{st}__C1.csv")})
pd.DataFrame(rows).round(4).to_csv(sys.stdout, index=False)

print("\nVALIDATION 001-2017_ME")
summ = pd.read_csv(AB / "ablation_summary.csv")
for arm in ["A1_EEGNet_low", "B1_MDM_low"]:
    r = score(AB / "predictions" / f"001-2017_ME__{arm}.csv")
    ref = summ[(summ.stratum == "001-2017_ME") & (summ.arm == arm)].iloc[0]
    pairs = [("ba_mean_splits", "ba_mean_splits"), ("ci95_lo", "ba_ci95_lo"), ("ci95_hi", "ba_ci95_hi"),
             ("kappa_mean_splits", "kappa_mean_splits"), ("chance", "chance"),
             ("n_participants", "n_participants"), ("p_dec", "p_raw")]
    ok = all(abs(round(r[a], 4) - ref[b]) < 1e-9 for a, b in pairs)
    print(arm, {a: round(r[a], 4) for a, _ in pairs}, "MATCH" if ok else "MISMATCH",
          {b: ref[b] for _, b in pairs})
