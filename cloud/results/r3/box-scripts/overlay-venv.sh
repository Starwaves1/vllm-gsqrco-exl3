#!/bin/bash
# shellcheck disable=SC2013  # patch paths have no spaces
# Make a patched copy of /workspace/venv-main without touching it (CPU only, seconds, no GPU):
# hardlink copy (no extra disk), then each file the patch touches is replaced by a real copy
# before patching, so venv-main's bytes never change. Idempotent: rebuilds the copy each time.
#   bash overlay-venv.sh DEST PATCH [PATCH...]      e.g. /workspace/venv-r3 ../patches/x.patch
set -euo pipefail
DEST=$1; shift
SRC=/workspace/venv-main
SP=lib/python3.12/site-packages
[ -d "$SRC/$SP/vllm" ] || { echo "no $SRC"; exit 2; }
case $DEST in /workspace/venv-r3*) ;; *) echo "DEST must be /workspace/venv-r3*"; exit 2 ;; esac
rm -rf "$DEST"
cp -al "$SRC" "$DEST"
for p in "$@"; do
  for f in $(grep -E '^\+\+\+ b/' "$p" | sed -E 's#^\+\+\+ b/([^[:space:]]+).*#\1#'); do
    t=$DEST/$SP/$f
    if [ "$(stat -c %i "$SRC/$SP/$f")" = "$(stat -c %i "$t")" ]; then  # still venv-main's inode
      cp --remove-destination "$SRC/$SP/$f" "$t.r3tmp" && mv -f "$t.r3tmp" "$t"
    fi
    rm -f "$(dirname "$t")/__pycache__/$(basename "${t%.py}")".*.pyc
  done
  patch -p1 -d "$DEST/$SP" --no-backup-if-mismatch < "$p"
done
# venv-main untouched: same inode count check on one patched file
for p in "$@"; do
  for f in $(grep -E '^\+\+\+ b/' "$p" | sed -E 's#^\+\+\+ b/([^[:space:]]+).*#\1#'); do
    [ "$(stat -c %i "$SRC/$SP/$f")" != "$(stat -c %i "$DEST/$SP/$f")" ] || { echo "hardlink not broken: $f"; exit 3; }
    cmp -s "$SRC/$SP/$f" "$DEST/$SP/$f" && { echo "patch did not change $f"; exit 3; }
  done
done
"$DEST/bin/python" -c "import vllm, sys; print('overlay ok', vllm.__version__, vllm.__file__)"
