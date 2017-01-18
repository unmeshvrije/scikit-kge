#!/bin/sh

python -m pdb skge/run_transe.py --fin data/twitter.bin \
       --test-all 5 --nb 100 --me 10 \
              --margin 2.0 --lr 0.1 --ncomp 50  --incr 10 --ftax data/taxonomy
