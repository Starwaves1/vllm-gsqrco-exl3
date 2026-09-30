#!/bin/bash
# one gpuq job: start the production-argv server from $WT, sample host RSS / RssAnon of its process
# tree every 0.5 s until /health, record load times, stop. WT=/workspace/wt-int2 = before.
source /workspace/wt-final/cloud/results/final/box-scripts/lib.sh
TAG=$1; R=$L/loadrss-$TAG; mkdir -p $R; date -u +"start %FT%TZ"; git log --oneline -1
( max=0 maxa=0; while :; do
    s=$(pgrep -f "bin/vllm serve" | head -1)
    if [ -n "$s" ]; then sid=$(ps -o sid= -p $s | tr -d ' ')
      read r a < <(for p in $(ps -o pid= -s $sid); do awk '/^VmRSS/{r=$2} /^RssAnon/{a=$2} END{print r+0, a+0}' /proc/$p/status 2>/dev/null; done | awk '{r+=$1; a+=$2} END {print r+0, a+0}')
      [ "$r" -gt "$max" ] && max=$r; [ "$a" -gt "$maxa" ] && maxa=$a
      echo "$(date +%s.%N | cut -c1-14) $r $a" >> $R/rss.txt; echo "$max $maxa" > $R/max.txt; fi
    sleep 0.5; done ) &
SMP=$!
serve
kill $SMP; read m ma < $R/max.txt
echo "peak RSS $((m / 1024)) MiB, peak RssAnon $((ma / 1024)) MiB (sum over the server's session, until healthy)"
stopall; date -u +"end %FT%TZ"
