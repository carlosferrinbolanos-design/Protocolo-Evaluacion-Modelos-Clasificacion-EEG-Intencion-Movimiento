"""Score arm C1 (Tensor-CSPNet) with exactly the same function used for A1, A2 and B1.

Reproduces the in-memory scoring run on 2026-11-21 (America/Bogota): for each stratum it
calls evaluate.score_predictions() from the v5 code (n_perm = 1000 test-label permutations
within participant, n_boot = 2000 participant-bootstrap draws, default_rng(0)) on the merged
predictions written by c1_runner_v2.py. It writes nothing and never prints participant codes.
The raw p values printed here enter the single Holm family of the ablation (36 tests:
4 arms x 6 strata vs chance + 2 contrasts x 6 strata); see ablacion_banda/code/ablation*.py.

Usage: python score_c1.py            (TESIS_V5_ROOT defaults to the v5/ folder of this repo)
"""
import os
import sys
from pathlib import Path

V5 = Path(os.environ.get("TESIS_V5_ROOT", Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(V5 / "code"))
import evaluate as ev  # noqa: E402

PRED = V5 / "ablacion_banda" / "results" / "c1" / "predictions"

for st in ev.STRATA:
    r = ev.score_predictions(PRED / f"{st}__C1.csv")
    r = dict(r)
    print(st, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items() if k != "subject"})
