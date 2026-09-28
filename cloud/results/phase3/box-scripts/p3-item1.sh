#!/bin/bash
# item 1 (routing: MMQ from 8 rows, lm_head from 4): parity, decode speed, profile; then the
# W4A16 baseline profile for item 2 (host-gap comparison)
source /workspace/p3/p3-lib.sh
date -u +"start %FT%TZ"
parity item1
speed item1
/workspace/p3/p3-profile.sh item1 gsq > $L/item1-profile.log 2>&1; tail -1 $L/item1-profile.log
/workspace/p3/p3-profile.sh base baseline > $L/base-profile.log 2>&1; tail -1 $L/base-profile.log
date -u +"end %FT%TZ"; echo ITEM1_DONE
