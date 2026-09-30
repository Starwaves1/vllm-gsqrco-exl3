#!/bin/bash
# (requested-workloads/01 lib/dl.sh + DL_REV=<commit or branch>, default main)
# Hugging Face downloads as parallel HTTP Range requests, resumable, every file checked
# (the pattern of gsq-vllm cloud/results/phase1/box-scripts/dl-gguf-par.sh and
# dl-baseline-curl.sh; the hf/xet client stalled on rented boxes).
#   dl.sh file <repo> <path> <dest-file> <sha256>   one file, sha256 must match
#   dl.sh repo <repo> <dest-dir>                     every file: LFS files against the Hub's
#                                                    sha256, the rest against their git blob id
# A finished file gets <file>.ok (its sha256); re-running skips it. HF_TOKEN is used if set.
# DL_CHUNKS parallel ranges per large file (default 16).
set -euo pipefail
N=${DL_CHUNKS:-16}
AUTH=(); [ -n "${HF_TOKEN:-}" ] && AUTH=(-H "Authorization: Bearer $HF_TOKEN")
die() { echo "dl: $*" >&2; exit 2; }

# tree <repo>: "path<TAB>size<TAB>lfs-sha256-or-empty<TAB>git-oid" per file
tree() {
  curl -sSfL "${AUTH[@]}" "https://huggingface.co/api/models/$1/tree/${DL_REV:-main}?recursive=true" | python3 -c '
import json, sys
for f in json.load(sys.stdin):
    if f["type"] == "file":
        lfs = f.get("lfs") or {}
        print(f["path"], lfs.get("size", f["size"]), lfs.get("oid", ""), f["oid"], sep="\t")'
}

chunk() {  # i url size chunk_size dir
  local i=$1 url=$2 total=$3 sz=$4 dir=$5
  local a=$((i * sz)) b=$(((i + 1) * sz - 1)) f have
  [ $b -ge "$total" ] && b=$((total - 1))
  local want=$((b - a + 1)); f=$dir/$(printf %03d "$i")
  for _ in $(seq 60); do
    have=$(stat -c %s "$f" 2>/dev/null || echo 0)
    [ "$have" = "$want" ] && return 0
    [ "$have" -gt "$want" ] && { rm -f "$f"; have=0; }
    curl -sSfL "${AUTH[@]}" --connect-timeout 20 --speed-limit 100000 --speed-time 60 \
      -r $((a + have))-$b "$url" >> "$f" || sleep 5
  done
  echo "chunk $i failed" >&2; return 1
}
export -f chunk; export AUTH_HDR=${HF_TOKEN:-}

get() {  # url size dest   (writes dest.part)
  local url=$1 size=$2 dest=$3
  if [ "$size" -lt "${DL_MIN_CHUNKED:-104857600}" ]; then
    curl -sSfL "${AUTH[@]}" --retry 5 -o "$dest.part" "$url"; return
  fi
  local dir=$dest.chunks sz=$(((size + N - 1) / N))
  mkdir -p "$dir"
  # the exported function re-reads AUTH from AUTH_HDR (arrays do not export)
  seq 0 $((N - 1)) | xargs -P "$N" -I{} bash -c \
    'AUTH=(); [ -n "$AUTH_HDR" ] && AUTH=(-H "Authorization: Bearer $AUTH_HDR"); chunk "$@"' _ {} "$url" "$size" "$sz" "$dir"
  : > "$dest.part"
  for i in $(seq 0 $((N - 1))); do cat "$dir/$(printf %03d "$i")" >> "$dest.part"; done
  [ "$(stat -c %s "$dest.part")" = "$size" ] || die "$dest: size $(stat -c %s "$dest.part") != $size"
  rm -rf "$dir"
}

fetch() {  # repo path dest size sha256 [git-oid]
  local repo=$1 path=$2 dest=$3 size=$4 sha=$5 oid=${6:-} have
  local url="https://huggingface.co/$repo/resolve/${DL_REV:-main}/$path"
  if [ -f "$dest" ] && [ -f "$dest.ok" ] && [ "$(cat "$dest.ok")" = "${sha:-$oid}" ]; then echo "ok (cached) $path"; return; fi
  mkdir -p "$(dirname "$dest")"
  get "$url" "$size" "$dest"
  if [ -n "$sha" ]; then have=$(sha256sum "$dest.part" | cut -d' ' -f1)
  else have=$(git hash-object --no-filters "$dest.part"); fi
  [ "$have" = "${sha:-$oid}" ] || { rm -f "$dest.part"; die "$repo/$path: checksum $have != ${sha:-$oid}"; }
  mv -f "$dest.part" "$dest"; echo "$have" > "$dest.ok"
  echo "ok $path ($size bytes, ${sha:+sha256 }$have)"
}

case ${1:-} in
  file)
    repo=$2 path=$3 dest=$4 want=$5
    line=$(tree "$repo" | awk -F'\t' -v p="$path" '$1 == p') || true
    [ -n "$line" ] || die "$repo has no file $path"
    size=$(cut -f2 <<<"$line"); hub=$(cut -f3 <<<"$line")
    [ "$hub" = "$want" ] || die "$repo/$path: the Hub's sha256 is $hub, pinned $want"
    fetch "$repo" "$path" "$dest" "$size" "$want" ;;
  repo)
    repo=$2 dir=$3
    tree "$repo" > "/tmp/dl-tree.$$"; trap 'rm -f /tmp/dl-tree.$$' EXIT
    [ -s "/tmp/dl-tree.$$" ] || die "empty file list for $repo"
    while IFS=$'\t' read -r path size sha oid; do
      fetch "$repo" "$path" "$dir/$path" "$size" "$sha" "$oid"
    done < "/tmp/dl-tree.$$" ;;
  *) sed -n '2,10p' "$0"; exit 2 ;;
esac
