#!/bin/sh

python skge/run_hole.py --fin data/wn18.bin \
       --test-all 50 --nb 100 --me 500 \
       --margin 0.2 --lr 0.1 --ncomp 150
