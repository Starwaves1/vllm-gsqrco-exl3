#!/bin/bash
# Download ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF IQ3_S-mtp with 32 parallel Range requests
# (pattern of /workspace/dl-gguf-par.sh) and verify sha256 against the HF LFS oid.
set -u
REV=d562806dbafae37109975e970aae91b43e73b440   # repo commit the tree listing was read at
F=Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf
U=https://huggingface.co/ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF/resolve/$REV/$F
WANT=58fd826723939933dc86f45b7fe04545cbc2de1c70f6fe2cdd3858c87a98c12f TOTAL=12120016960 N=32
D=/workspace/models C=$D/base.chunks P=$D/base.part
mkdir -p $C; SZ=$(( (TOTAL + N - 1) / N ))
date -u +"start %FT%TZ"
chunk() { local i=$1 a=$(( $1*SZ )) b=$(( ($1+1)*SZ - 1 )); [ $b -ge $TOTAL ] && b=$((TOTAL-1)); local want=$((b-a+1)) f=$C/$(printf %03d $1)
  for t in $(seq 1 60); do
    have=$( [ -f $f ] && stat -c %s $f || echo 0 ); [ "$have" = "$want" ] && return 0
    [ "$have" -gt "$want" ] && { echo "chunk $i OVERSIZE"; return 1; }
    curl -sSL --connect-timeout 20 --speed-limit 50000 --speed-time 60 -r $((a+have))-$b "$U" >> $f
  done; echo "chunk $i FAILED"; }
export -f chunk; export U C SZ TOTAL
seq 0 $((N-1)) | xargs -P $N -I{} bash -c "chunk {}"
rm -f $P; for i in $(seq 0 $((N-1))); do cat $C/$(printf %03d $i) >> $P; done
echo "size $(stat -c %s $P)"; date -u +"download done %FT%TZ"
H=$(sha256sum $P | cut -d" " -f1); echo "sha256 $H"
if [ "$H" = $WANT ]; then mv $P $D/$F && rm -rf $C && echo GGUF_VERIFIED; else echo GGUF_BAD; fi
date -u +"end %FT%TZ"
