#!/bin/bash
/workspace/p1-ab.sh gsq ladderclock > /workspace/logs/p1-ab-gsq.log 2>&1
/workspace/p1-ab.sh baseline full > /workspace/logs/p1-ab-baseline.log 2>&1
echo AFTER_DONE
