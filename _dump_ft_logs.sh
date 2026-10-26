#!/bin/bash
cd /mnt/d/workspace/Swallow/wandb
for d in run-20261024_170152-2gc7tn9m run-20261024_173124-91vn8j05 run-20261023_112652-annw55il; do
  echo "===== $d"
  grep -a "Testing accuracy" "$d/files/output.log" | tr -d '\r' | head -30
done