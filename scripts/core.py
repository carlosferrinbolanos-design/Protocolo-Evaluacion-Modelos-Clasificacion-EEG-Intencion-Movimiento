# Shared core: David's frozen v3 protocol, repeated participant-disjoint splits.
import json, time, gc, os
import numpy as np, pandas as pd
SEED0 = 20261113
NSEEDS = 10
def seed_of(k): return SEED0 + k          # k=0 reproduces David's frozen split exactly
def split(name, s, k):
    u = np.array(sorted(set(s)))
    rng = np.random.default_rng(seed_of(k) + sum(map(ord, name)))
    tst = set(rng.choice(u, size=max(1, int(np.ceil(.2 * len(u)))), replace=False))
    te = np.array([q in tst for q in s]); return ~te, te, sorted(tst)
def permute_train(y, s, tr, seed):
    z = y.copy(); rng = np.random.default_rng(seed)
    for q in sorted(set(s[tr])):
        ii = np.where((s == q) & tr)[0]; z[ii] = rng.permutation(z[ii])
    return z
def save_preds(outdir, name, model, control, k, s, te, ytrue, ypred, classes, runtime, testsubs):
    os.makedirs(outdir, exist_ok=True)
    df = pd.DataFrame({'stratum': name, 'model': model, 'control': control, 'seed_idx': k, 'seed': seed_of(k),
                       'subject': s[te], 'y_true': np.asarray(classes)[ytrue], 'y_pred': np.asarray(classes)[ypred]})
    f = os.path.join(outdir, f'{name}__{model}__{control}__s{k}.csv.gz')
    df.to_csv(f + '.tmp.gz', index=False, compression='gzip'); os.replace(f + '.tmp.gz', f)
    with open(os.path.join(outdir, f'{name}__{model}__{control}__s{k}.json'), 'w') as fh:
        json.dump({'stratum': name, 'model': model, 'control': control, 'seed_idx': k, 'seed': seed_of(k),
                   'runtime_s': runtime, 'test_subjects': testsubs, 'n_test': int(te.sum())}, fh)
def done(outdir, name, model, control, k):
    return os.path.exists(os.path.join(outdir, f'{name}__{model}__{control}__s{k}.csv.gz'))
