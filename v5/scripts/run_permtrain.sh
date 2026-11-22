#!/bin/bash
# Training-label permutation (200) for the best pipeline of each stratum, cheapest first.
cd "$(dirname "$0")/../code"   # original: D:/ICTQ/TESIS_pipeline_v5/code
P=${PYTHON:-python}
for job in "001-2020_dry DeepConvNet" "001-2020_water DeepConvNet" "001-2020_gel ShallowConvNet" \
           "001-2019_offline EEGNet" "001-2017_MI EEGNet" "001-2017_ME EEGNet"; do
  set -- $job
  $P evaluate.py permtrain ../epochs ../results $1 $2 200 >> ../results/permtrain.log 2>&1
  echo "EXIT $1 $2 $?" >> ../results/permtrain_status.txt
done
