#!/bin/sh

python skge/run_transe.py --fin data/wn18.bin \
       --test-all 50 --nb 100 --me 50 \
       --margin 2.0 --lr 0.1 --ncomp 50 --incr 100 --fembed embeds.pickle
