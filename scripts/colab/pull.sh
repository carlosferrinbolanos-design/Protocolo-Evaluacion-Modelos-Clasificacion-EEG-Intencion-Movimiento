#!/bin/bash
# usage: pull.sh SESSION  -> tar /content/out on VM, download, extract into preds/
S=$1; D=$(cd "$(dirname "$0")/../.." && pwd)
cat > /tmp/tar_$S.py <<PY
import subprocess;print(subprocess.run('cd /content/out && tar czf /content/out_$S.tgz --exclude=ALLDONE* . 2>&1; ls /content/out | wc -l; ls /content/out | grep ALLDONE',shell=True,capture_output=True,text=True).stdout)
PY
timeout 120 colab exec -s $S -f /tmp/tar_$S.py --timeout 100 2>&1 | tail -n 3
timeout 300 colab download -s $S /content/out_$S.tgz /tmp/out_$S.tgz 2>&1 | tail -n 1
mkdir -p $D/preds && tar xzf /tmp/out_$S.tgz -C $D/preds && echo "pulled $S: $(tar tzf /tmp/out_$S.tgz | wc -l) files"
