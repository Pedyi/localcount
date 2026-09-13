#!/usr/bin/env bash
# Every experiment that needs no network access.
set -e
for e in exp1_space_bound exp7_mixture exp8_separation exp2_monte_carlo; do
  echo; echo "############ $e ############"; echo
  python experiments/$e.py | tee "results/$e.txt"
done
echo; echo "results written to results/"
