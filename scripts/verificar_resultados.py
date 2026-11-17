"""Verificación de reproducibilidad: compara las tablas regeneradas con las publicadas en el repositorio.

Uso:
    python scripts/verificar_resultados.py --regenerado /content/tesis_eeg/tablas [--reportado tables] [--modo completo|rapido]
                                           [--tol-clasico 5e-4] [--tol-red 0.02] [--tol-estadistica 1e-3]

Qué compara:
  1. Siempre: con k = 0, la exactitud balanceada de los modelos clásicos frente a la partición congelada v3 del estudio
     (deterministas; tolerancia --tol-clasico).
  2. En modo completo: medias de exactitud balanceada, F1 macro, kappa y MCC por estrato × modelo
     (summary_mean_sd.csv). Tolerancias: --tol-clasico para los clásicos y --tol-red para las redes (GPU no determinista).
  3. En modo completo: χ² y p asintótica de Friedman (stats.json) y p de Holm frente al azar (vs_chance.csv).
     Solo deben coincidir exactamente si se regeneraron desde las mismas predicciones; con reentrenamiento
     en GPU se informan como diferencia, sin fallar, salvo que se use --estricto.
Devuelve código 0 si todo está dentro de la tolerancia y 1 en caso contrario.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REDES = {"EEGNet", "ShallowConvNet", "DeepConvNet"}
METRICAS = ["balanced_accuracy", "macro_f1", "cohen_kappa", "mcc"]
CONTROL_REAL = "true_labels"
# Exactitud balanceada con k = 0 de la ejecución v3 del estudio (partición congelada, modelos clásicos).
DAVID_V3_K0 = {
    ("001-2017_ME", "CSP_LDA"): 0.1468, ("001-2017_ME", "FBCSP_SVM"): 0.1524, ("001-2017_ME", "Riemannian_MDM"): 0.1349,
    ("001-2017_MI", "CSP_LDA"): 0.1413, ("001-2017_MI", "FBCSP_SVM"): 0.1429, ("001-2017_MI", "Riemannian_MDM"): 0.1421,
    ("001-2019_offline", "CSP_LDA"): 0.2167, ("001-2019_offline", "FBCSP_SVM"): 0.2083, ("001-2019_offline", "Riemannian_MDM"): 0.2236,
}


def comparar_k0(regen: Path, tol: float) -> list[str]:
    """Compara la exactitud balanceada de los clásicos con k = 0 frente a la ejecución v3."""
    s = pd.read_csv(regen / "metrics_per_split.csv")
    s = s[(s.control == CONTROL_REAL) & (s.seed_idx == 0)].set_index(["stratum", "model"]).balanced_accuracy
    fallos = []
    for clave, ref in DAVID_V3_K0.items():
        if clave not in s.index:
            continue
        d = abs(s[clave] - ref)
        estado = "OK" if d <= tol else "FALLA"
        print(f"[k=0 v3] {clave[0]:18s} {clave[1]:15s} regenerado={s[clave]:.4f} v3={ref:.4f} |Δ|={d:.5f} {estado}")
        if d > tol:
            fallos.append(f"k0 {clave}")
    return fallos


def comparar_medias(regen: Path, rep: Path, tol_c: float, tol_r: float) -> list[str]:
    """Compara las medias por estrato × modelo de las cuatro métricas."""
    a = pd.read_csv(regen / "summary_mean_sd.csv")
    b = pd.read_csv(rep / "summary_mean_sd.csv")
    a, b = (x[x.control == CONTROL_REAL].set_index(["stratum", "model"]) for x in (a, b))
    fallos = []
    for clave in b.index:
        if clave not in a.index:
            print(f"[medias] falta {clave} en lo regenerado"); fallos.append(f"falta {clave}"); continue
        tol = tol_r if clave[1] in REDES else tol_c
        for m in METRICAS:
            d = abs(a.loc[clave, f"{m}_mean"] - b.loc[clave, f"{m}_mean"])
            if d > tol:
                fallos.append(f"{clave} {m} |Δ|={d:.4f}")
        print(f"[medias] {clave[0]:18s} {clave[1]:15s} BA {a.loc[clave, 'balanced_accuracy_mean']:.4f} vs {b.loc[clave, 'balanced_accuracy_mean']:.4f}"
              f" | MCC {a.loc[clave, 'mcc_mean']:.4f} vs {b.loc[clave, 'mcc_mean']:.4f} (tol {tol})")
    return fallos


def comparar_estadistica(regen: Path, rep: Path, tol: float, estricto: bool) -> list[str]:
    """Compara Friedman (χ², p asintótica, rangos) y la p de Holm frente al azar."""
    a = json.load(open(regen / "stats.json"))["stats"]
    b = json.load(open(rep / "stats.json"))["stats"]
    fallos = []
    for met, bloques in b.items():
        for blq, v in bloques.items():
            if blq == "pairwise" or blq not in a.get(met, {}):
                continue
            w = a[met][blq]
            dchi, dp = abs(w["chi2"] - v["chi2"]), abs(w["p_asymptotic"] - v["p_asymptotic"])
            drk = max(abs(w["avg_ranks"][m] - v["avg_ranks"][m]) for m in v["avg_ranks"])
            ok = dchi <= 10 * tol and dp <= tol and drk <= tol
            print(f"[Friedman] {met:17s} {blq:30s} χ² {w['chi2']:.3f} vs {v['chi2']:.3f} | p {w['p_asymptotic']:.4f} vs {v['p_asymptotic']:.4f}"
                  f" | máx Δrango {drk:.3f} {'OK' if ok else 'DIFIERE'}")
            if not ok and estricto:
                fallos.append(f"Friedman {met} {blq}")
    va, vb = (pd.read_csv(x / "vs_chance.csv").set_index(["stratum", "model"]) for x in (regen, rep))
    for clave in vb.index.intersection(va.index):
        d = abs(va.loc[clave, "p_holm"] - vb.loc[clave, "p_holm"])
        if d > tol and estricto:
            fallos.append(f"vs azar {clave}")
    return fallos


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--regenerado", required=True, type=Path, help="carpeta de tablas regeneradas (p. ej., <RUTA_TRABAJO>/tablas)")
    ap.add_argument("--reportado", default=Path(__file__).resolve().parents[1] / "tables", type=Path, help="carpeta de tablas publicadas")
    ap.add_argument("--modo", choices=["completo", "rapido"], default="completo")
    ap.add_argument("--tol-clasico", type=float, default=5e-4)
    ap.add_argument("--tol-red", type=float, default=0.02)
    ap.add_argument("--tol-estadistica", type=float, default=1e-3)
    ap.add_argument("--estricto", action="store_true", help="falla también si difiere la estadística (usar al regenerar desde preds/)")
    a = ap.parse_args()
    fallos = comparar_k0(a.regenerado, a.tol_clasico)
    if a.modo == "completo":
        fallos += comparar_medias(a.regenerado, a.reportado, a.tol_clasico, a.tol_red)
        fallos += comparar_estadistica(a.regenerado, a.reportado, a.tol_estadistica, a.estricto)
    print("\nRESULTADO:", "VERIFICADO" if not fallos else f"{len(fallos)} discrepancias")
    for f in fallos:
        print("  -", f)
    return 0 if not fallos else 1


if __name__ == "__main__":
    sys.exit(main())
