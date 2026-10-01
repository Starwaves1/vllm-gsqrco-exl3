#!/bin/bash
# Refresh this copy of prod-recorder from the gsq-vllm repo's `prod-recorder` branch (bench/prod-recorder/).
# Recordings under data/ are kept.
#   ./sync-from-repo.sh        REPO=~/gsq-vllm REF=origin/prod-recorder by default
set -euo pipefail
REPO=${REPO:-$HOME/gsq-vllm} REF=${REF:-origin/prod-recorder}
DEST=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
git -C "$REPO" fetch -q origin prod-recorder || echo "fetch failed; using the local $REF" >&2
git -C "$REPO" archive "$REF" bench/prod-recorder | tar -x --strip-components=2 -C "$DEST"
echo "synced $DEST from $REPO $(git -C "$REPO" rev-parse --short "$REF")"
