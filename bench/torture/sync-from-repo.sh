#!/bin/bash
# Refresh this copy of the torture harness from the gsq-vllm repo's `torture` branch (bench/torture/).
# Results under torture-runs/ are kept.
#   ./sync-from-repo.sh        REPO=~/gsq-vllm REF=origin/torture by default
set -euo pipefail
REPO=${REPO:-$HOME/gsq-vllm} REF=${REF:-origin/torture}
DEST=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
git -C "$REPO" fetch -q origin torture || echo "fetch failed; using the local $REF" >&2
git -C "$REPO" archive "$REF" bench/torture | tar -x --strip-components=2 -C "$DEST"
echo "synced $DEST from $REPO $(git -C "$REPO" rev-parse --short "$REF")"
