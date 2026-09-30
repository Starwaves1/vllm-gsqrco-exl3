#!/bin/bash
# Apply production's vLLM overlay to a fresh wheel install: the minimal steps of production's
# scripts/deploy-vllm.sh (private deploy repo) for one deploy from the wheel's own commit.
#   1. the wheel must be vLLM $VLLM_EXPECT_VERSION and match $VLLM_BASE on every file git
#      tracks under vllm/ (deploy-vllm.sh --init <base>);
#   2. copy the files under vllm/ that differ between $VLLM_BASE and $VLLM_OVERLAY
#      (A/M: write the blob, D: delete) (deploy-vllm.sh <overlay>);
#   3. every tracked file must now match $VLLM_OVERLAY (deploy-vllm.sh --verify).
# Idempotent: a wheel that already matches the overlay passes straight to step 3.
#   usage: overlay-vllm.sh <vllm-src clone> <site-packages dir>
set -euo pipefail
export LC_ALL=C
source "$(dirname "${BASH_SOURCE[0]}")/pins.sh"
SRC=$1 SP=$2
g() { git -C "$SRC" "$@"; }
die() { echo "overlay: $*" >&2; exit 2; }

grep -qF "__version__ = version = '$VLLM_EXPECT_VERSION'" "$SP/vllm/_version.py" \
  || die "$SP/vllm/_version.py is not vLLM $VLLM_EXPECT_VERSION"
g cat-file -e "$VLLM_BASE^{commit}" && g cat-file -e "$VLLM_OVERLAY^{commit}" || die "commits missing in $SRC"
g merge-base --is-ancestor "$VLLM_BASE" "$VLLM_OVERLAY" || die "$VLLM_OVERLAY does not descend from $VLLM_BASE"

TMP=$(mktemp -d /tmp/overlay-vllm.XXXXXX); trap 'rm -rf "$TMP"' EXIT
# mismatches <commit>: tracked files under vllm/ whose wheel copy differs (or is missing)
mismatches() {
  g ls-tree -r --full-tree "$1" -- vllm | awk '$2 == "blob" {print $4 "\t" $3}' | sort > "$TMP/want"
  cut -f1 "$TMP/want" | while IFS= read -r p; do
    if [ -f "$SP/$p" ]; then printf '%s\t%s\n' "$p" "$(git hash-object --no-filters "$SP/$p")"
    else printf '%s\tABSENT\n' "$p"; fi
  done | sort > "$TMP/have"
  echo "$1: $(wc -l < "$TMP/want") tracked files under vllm/" >&2
  join -t$'\t' "$TMP/want" "$TMP/have" | awk -F'\t' '$2 != $3 {print $1}'
}

bad=$(mismatches "$VLLM_OVERLAY")
if [ -z "$bad" ]; then echo "wheel already matches overlay $VLLM_OVERLAY"; exit 0; fi
bad=$(mismatches "$VLLM_BASE")
[ -z "$bad" ] || { echo "$bad" | head -20 >&2; die "wheel differs from base $VLLM_BASE on $(echo "$bad" | wc -l) files (not a fresh wheel?)"; }
echo "wheel matches base $VLLM_BASE"

g diff --no-renames --name-status "$VLLM_BASE" "$VLLM_OVERLAY" -- vllm > "$TMP/changes"
echo "overlay plan: $(wc -l < "$TMP/changes") files (A=add M=modify D=delete)"
while IFS=$'\t' read -r st p; do
  echo "  $st $p"
  dst=$SP/$p
  if [ "$st" = D ]; then rm -f "$dst"; continue; fi
  mkdir -p "$(dirname "$dst")"
  g cat-file blob "$VLLM_OVERLAY:$p" > "$dst.overlay.$$"
  mode=$(g ls-tree "$VLLM_OVERLAY" -- "$p" | awk '{print $1}')
  if [ "$mode" = 100755 ]; then chmod 755 "$dst.overlay.$$"; else chmod 644 "$dst.overlay.$$"; fi
  mv -f "$dst.overlay.$$" "$dst"
done < "$TMP/changes"

bad=$(mismatches "$VLLM_OVERLAY")
[ -z "$bad" ] || { echo "$bad" >&2; die "post-check failed"; }
echo "OK: every tracked file matches overlay $VLLM_OVERLAY"
