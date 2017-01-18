#!/bin/sh

python skge/run_transe.py --fin data/lubm1_uri.bin \
       --test-all 50 --nb 100 --me 500 \
              --margin 2.0 --lr 0.1 --ncomp 50  --incr 10 --ftax data/taxonomy
