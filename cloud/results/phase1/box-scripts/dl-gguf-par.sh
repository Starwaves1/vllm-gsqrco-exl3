#!/bin/bash
# parallel Range download of the GGUF tail onto the existing partial
U=https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-27B-GSQ-RCO-GGUF/resolve/main/Swift-1.5-Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf
P=/workspace/models/gguf.part C=/workspace/models/gguf.chunks TOTAL=12120016896 N=32
mkdir -p $C
S=$(stat -c %s $P); REM=$((TOTAL-S)); SZ=$(( (REM + N - 1) / N ))
echo "resume from $S, $N chunks of $SZ"
chunk() { local i=$1 a=$(( S + $1*SZ )) b=$(( S + ($1+1)*SZ - 1 )); [ $b -ge $TOTAL ] && b=$((TOTAL-1)); local want=$((b-a+1)) f=$C/$(printf %03d $1)
  for t in $(seq 1 60); do
    have=$( [ -f $f ] && stat -c %s $f || echo 0 ); [ "$have" = "$want" ] && return 0
    [ "$have" -gt "$want" ] && { echo "chunk $i OVERSIZE"; return 1; }
    curl -sSL --connect-timeout 20 --speed-limit 50000 --speed-time 60 -r $((a+have))-$b "$U" >> $f
  done; echo "chunk $i FAILED"; }
export -f chunk; export U C S SZ TOTAL
seq 0 $((N-1)) | xargs -P $N -I{} bash -c "chunk {}"
for i in $(seq 0 $((N-1))); do cat $C/$(printf %03d $i) >> $P; done
echo "size $(stat -c %s $P)"; date -u +"gguf done %FT%TZ"
H=$(sha256sum $P | cut -d" " -f1); echo "sha256 $H"
[ "$H" = 9aecf1cd41b2cb2f32a74e0d889e33855ebef43b26f43b43feb5720239e677e5 ] && mv $P /workspace/models/Swift-1.5-Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf && echo GGUF_VERIFIED || echo GGUF_BAD
