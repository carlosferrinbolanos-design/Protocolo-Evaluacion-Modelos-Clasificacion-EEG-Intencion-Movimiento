"""Regenerate model-input epochs for the three BNCI datasets from the public source files.

Pipeline v5 (decisions ratified by Carlos on 2026-11-14):
  * EEG channels selected by name (never by index).
  * 0.5 Hz zero-phase FIR high-pass on the CONTINUOUS recording, before epoching.
  * Resampling of the continuous recording to 256 Hz (2017 is recorded at 512 Hz).
  * Epochs -0.5 to 1.5 s around the event (513 samples), no baseline, no amplitude rejection.
  * 001-2017: cue codes 0x600-0x606 (7 classes), ME and MI separate, S01_ME excluded.
  * 001-2019: offline attempted-movement runs only, 5 classes; P01 from the Zenodo archive.
  * 001-2020: movement-onset codes, 2 classes, analysed per system, 11 common channels,
    events closer than 4 s to the previous movement onset excluded.
Amplitudes are kept in the units MNE reports; no physical unit is claimed.

Usage: python extract_epochs.py <001-2017|001-2019|001-2020> <extracted_root> <out_dir>
"""
import json
import re
import sys
from pathlib import Path

import mne
import numpy as np
import pandas as pd

FS = 256.0
TMIN, TMAX = -0.5, 1.5
HP = 0.5
mne.set_log_level("ERROR")

D17_CH = ("F3 F1 Fz F2 F4 FFC5h FFC3h FFC1h FFC2h FFC4h FFC6h FC5 FC3 FC1 FCz FC2 FC4 FC6 "
          "FTT7h FCC5h FCC3h FCC1h FCC2h FCC4h FCC6h FTT8h C5 C3 C1 Cz C2 C4 C6 TTP7h CCP5h "
          "CCP3h CCP1h CCP2h CCP4h CCP6h TTP8h CP5 CP3 CP1 CPz CP2 CP4 CP6 CPP5h CPP3h CPP1h "
          "CPP2h CPP4h CPP6h P3 P1 Pz P2 P4 PPO1h PPO2h").split()
D17_CODES = {"1536": "elbow_flexion", "1537": "elbow_extension", "1538": "supination",
             "1539": "pronation", "1540": "hand_close", "1541": "hand_open", "1542": "rest"}
D19_CODES = {"776": "supination", "777": "pronation", "779": "hand_open",
             "925": "palmar_grasp", "926": "lateral_grasp"}
D20_CH = ["FC3", "FCZ", "FC4", "C3", "C1", "CZ", "C2", "C4", "CP3", "CPZ", "CP4"]
D20_CODES = {503587: "palmar_grasp", 503588: "lateral_grasp"}


def canon(x):
    return re.sub(r"[^A-Z0-9]", "", str(x).upper())


def pick_by_name(raw, wanted):
    idx = {canon(c): c for c in raw.ch_names}
    missing = [w for w in wanted if canon(w) not in idx]
    if missing:
        raise RuntimeError(f"missing channels {missing}")
    names = [idx[canon(w)] for w in wanted]
    raw.pick(names)
    raw.reorder_channels(names)
    raw.set_channel_types({n: "eeg" for n in names})
    return names


def preprocess(raw):
    raw.load_data()
    # GDF files end with a block of NaN padding; a filter would spread it over the whole
    # recording. Crop the trailing NaN block; any NaN elsewhere is an error.
    fin = np.isfinite(raw._data).all(axis=0)
    last = int(np.where(fin)[0][-1])
    if not fin[:last + 1].all():
        raise RuntimeError("NaN inside the recording")
    if last + 1 < raw.n_times:
        raw.crop(tmax=last / raw.info["sfreq"])
    raw.filter(l_freq=HP, h_freq=None, phase="zero", fir_design="firwin", picks="all")
    if raw.info["sfreq"] != FS:
        raw.resample(FS, npad="auto")
    return raw


def cut(raw, onsets_s, codes):
    """Cut epochs at onset times (seconds from recording start)."""
    samp = np.round(np.asarray(onsets_s) * FS).astype(int) + raw.first_samp
    ev = np.column_stack([samp, np.zeros_like(samp), np.asarray(codes, int)])
    ep = mne.Epochs(raw, ev, tmin=TMIN, tmax=TMAX, baseline=None, preload=True,
                    reject_by_annotation=False, event_repeated="error")
    if len(ep) != len(ev):
        raise RuntimeError(f"dropped epochs: {len(ep)}/{len(ev)} {ep.drop_log}")
    X = ep.get_data().astype(np.float32)
    assert X.shape[-1] == 513, X.shape
    return X


def run_2017(root):
    out = {"001-2017_ME": [], "001-2017_MI": []}
    for f in sorted(Path(root).rglob("*.gdf")):
        m = re.match(r"motor(execution|imagination)_subject(\d+)_run(\d+)", f.stem)
        sess = "ME" if m.group(1) == "execution" else "MI"
        subj = f"S{int(m.group(2)):02d}"
        if subj == "S01" and sess == "ME":
            continue
        raw = mne.io.read_raw_gdf(f, preload=False)
        pick_by_name(raw, D17_CH)
        on, lab, code = [], [], []
        for o, d in zip(raw.annotations.onset, raw.annotations.description):
            if d in D17_CODES:
                on.append(o); lab.append(D17_CODES[d]); code.append(int(d))
        if len(on) != 42:
            raise RuntimeError(f"{f.name}: {len(on)} cues")
        raw = preprocess(raw)
        X = cut(raw, on, code)
        out[f"001-2017_{sess}"].append((X, lab, subj, f.stem, np.asarray(on)))
    return out


D19_OFFLINE_RUNS = {3, 4, 5, 6, 7, 10, 11, 12, 13}


def run_2019(root):
    rows = []
    for f in sorted(Path(root).rglob("*.gdf")):
        m = re.match(r"(P\d\d) Run (\d+)$", f.stem)
        if not m or int(m.group(2)) not in D19_OFFLINE_RUNS:
            continue
        if "MACOSX" in str(f) or f.name.startswith("._"):
            continue
        subj = m.group(1)
        if subj == "P01" and "zenodo" not in str(f).lower():
            continue
        raw = mne.io.read_raw_gdf(f, preload=False)
        eeg = [c for c, t in zip(raw.ch_names, raw.get_channel_types()) if t == "eeg"
               and not canon(c).startswith("EOG")]
        on, lab, code = [], [], []
        for o, d in zip(raw.annotations.onset, raw.annotations.description):
            if d in D19_CODES:
                on.append(o); lab.append(D19_CODES[d]); code.append(int(d))
        if len(on) != 40:
            raise RuntimeError(f"{f}: {len(on)} cues")
        pick_by_name(raw, eeg)
        raw = preprocess(raw)
        X = cut(raw, on, code)
        rows.append((X, lab, subj, f.stem, np.asarray(on), raw.ch_names))
    chans = {tuple(r[5]) for r in rows}
    if len(chans) != 1:
        raise RuntimeError(f"channel sets differ across runs: {len(chans)}")
    return {"001-2019_offline": [r[:5] for r in rows]}, list(rows[0][5])


def run_2020(root):
    from pymatreader import read_mat
    out = {"001-2020_gel": [], "001-2020_water": [], "001-2020_dry": []}
    sysmap = {"G": "gel", "V": "water", "H": "dry"}
    audit = []
    for f in sorted(Path(root).glob("*.mat")):
        system = sysmap[f.stem[0]]
        m = read_mat(str(f))
        hdr = m["header"]
        sig = np.asarray(m["signal"], dtype=np.float64)
        # signal is (channels, samples); channels_labels also lists sensor channels that are
        # not stored in signal (e.g. PTH-*), so labels are matched by position up to n rows.
        labels = [str(x).strip() for x in hdr["channels_labels"]]
        if sig.shape[0] > sig.shape[1]:
            sig = sig.T
        labels = labels[:sig.shape[0]]
        info = mne.create_info(labels, float(hdr["sample_rate"]), "misc")
        raw = mne.io.RawArray(sig, info)
        pick_by_name(raw, D20_CH)
        codes = np.asarray(m["events"]["codes"]).astype(int).ravel()
        pos = np.asarray(m["events"]["positions"]).astype(int).ravel() - 1  # MAT is 1-based
        sel = np.isin(codes, list(D20_CODES))
        c, p = codes[sel], pos[sel]
        t = p / float(hdr["sample_rate"])
        dt = np.diff(t, prepend=-np.inf)
        keep = dt >= 4.0
        for i in np.where(~keep)[0]:
            audit.append({"file": f.name, "event_order": int(i + 1), "delta_s": float(dt[i])})
        raw = preprocess(raw)
        X = cut(raw, t[keep], c[keep])
        lab = [D20_CODES[k] for k in c[keep]]
        out[f"001-2020_{system}"].append((X, lab, f.stem, f.stem, t[keep]))
    return out, audit


def save(out, out_dir, channels):
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {}
    for stratum, parts in out.items():
        X = np.concatenate([p[0] for p in parts])
        y = np.concatenate([np.asarray(p[1]) for p in parts])
        s = np.concatenate([[p[2]] * len(p[1]) for p in parts])
        r = np.concatenate([[p[3]] * len(p[1]) for p in parts])
        on = np.concatenate([p[4] for p in parts])
        bad = ~np.isfinite(X).all(axis=(1, 2))
        np.savez(out_dir / f"{stratum}.npz", X=X, y=y, subject=s, run=r, onset_s=on,
                 channels=np.asarray(channels[stratum]), fs=FS, tmin=TMIN, finite=~bad)
        pd.DataFrame({"subject": s, "run": r, "onset_s": on, "label": y, "finite": ~bad}).to_csv(
            out_dir / f"{stratum}_manifest.csv", index=False)
        summary[stratum] = {"n_epochs": int(len(y)), "n_subjects": int(len(set(s))),
                            "n_channels": int(X.shape[1]), "n_times": int(X.shape[2]),
                            "non_finite_epochs": int(bad.sum()),
                            "per_class": {k: int(v) for k, v in zip(*np.unique(y, return_counts=True))}}
    return summary


if __name__ == "__main__":
    ds, root, out_dir = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
    if ds == "001-2017":
        out = run_2017(root); ch = {k: D17_CH for k in out}; extra = {}
    elif ds == "001-2019":
        out, c19 = run_2019(root); ch = {k: c19 for k in out}; extra = {}
    else:
        out, audit = run_2020(root); ch = {k: D20_CH for k in out}; extra = {"excluded_under_4s": audit}
    summ = save(out, out_dir, ch)
    summ.update(extra)
    summ["preprocessing"] = {"highpass_hz": HP, "phase": "zero", "applied_to": "continuous",
                             "resample_hz": FS, "window_s": [TMIN, TMAX], "baseline": None,
                             "amplitude_rejection": None, "mne": mne.__version__}
    (out_dir / f"{ds}_extraction_summary.json").write_text(json.dumps(summ, indent=2))
    print(json.dumps(summ, indent=2))
