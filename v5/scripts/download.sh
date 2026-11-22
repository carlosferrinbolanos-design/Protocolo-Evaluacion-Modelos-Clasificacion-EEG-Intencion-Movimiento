#!/bin/bash
# Downloads public BNCI datasets into $TESIS_DATA_ROOT/raw (resumable).
# download_list.txt (one "URL filename" pair per line) is NOT versioned: build it from the
# BNCI Horizon 2020 data-set pages (001-2017, 001-2019, 001-2020) and Zenodo; see v5/README.md.
R=${TESIS_DATA_ROOT:?set TESIS_DATA_ROOT}/raw
get() { url="$1"; out="$R/$2"; curl -sS -fL --ssl-no-revoke --retry 5 --retry-delay 10 -C - -o "$out" "$url" && echo "OK $2" || echo "FAIL $2"; }
export -f get; export R
cat download_list.txt | xargs -P 4 -L 1 bash -c 'get "$0" "$1"'
