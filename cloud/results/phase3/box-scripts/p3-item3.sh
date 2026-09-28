#!/bin/bash
# item 3: (b) draft lm_head placeholder dropped, alone (MTP_DRAFT_VOCAB=0: full-vocab draft head,
# shared from the target); (a+b) pruned 40,960-row Q4_K draft head: VRAM, decode speed, profile
source /workspace/p3/p3-lib.sh
date -u +"start %FT%TZ"
MTP_DRAFT_VOCAB=0 /workspace/p3/p3-vram.sh item3b > $L/item3b-vram.log 2>&1; cat $L/item3b-vram.log
/workspace/p3/p3-vram.sh item3 > $L/item3-vram.log 2>&1; cat $L/item3-vram.log
speed item3
/workspace/p3/p3-profile.sh item3 gsq > $L/item3-profile.log 2>&1; tail -1 $L/item3-profile.log
date -u +"end %FT%TZ"; echo ITEM3_DONE
