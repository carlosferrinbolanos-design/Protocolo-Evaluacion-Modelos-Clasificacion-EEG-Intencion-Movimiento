#!/bin/bash
# Resume training-label permutations for 2017 after the 2026-11-15 power loss (MI resumes at 40/200).
cd "$(dirname "$0")/../code"   # original: D:/ICTQ/TESIS_pipeline_v5/code
P=${PYTHON:-python}
for job in "001-2017_MI EEGNet" "001-2017_ME EEGNet"; do
  set -- $job
  $P evaluate.py permtrain ../epochs ../results $1 $2 200 >> ../results/permtrain.log 2>&1
  echo "EXIT $1 $2 $?" >> ../results/permtrain_status.txt
done
