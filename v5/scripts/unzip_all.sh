#!/bin/bash
# Extract each downloaded zip into its own folder after a path-safety check.
R=${TESIS_DATA_ROOT:?set TESIS_DATA_ROOT}
# CHK: helper that prints "bad [...]" with unsafe member paths of a zip (not versioned; temporary helper in the original run)
CHK=${ZIPCHECK:?set ZIPCHECK to a zip path-safety checker}
for z in $R/raw/$1/*.zip; do
  stem=$(basename "$z" .zip); dest="$R/extracted/$1/$stem"
  [ -f "$dest/.done" ] && continue
  out=$(${PYTHON:-python} -I $CHK "$z") || { echo "BADZIP $z"; continue; }
  echo "$out" | grep -q "bad \[\]" || { echo "UNSAFE $z: $out"; continue; }
  mkdir -p "$dest" && unzip -q -o "$z" -d "$dest" && touch "$dest/.done" && echo "OK $stem"
done
