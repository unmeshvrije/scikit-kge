#!/bin/sh
EPOCH=10
MS=5

if [ $# -ge 2 ]; then
    EPOCH=$1;
fi

if [ $# -ge 3 ]; then
    MS=$2;
fi

#python run_hole.py --fin /var/scratch/uji300/trident/lubm1  --test-all 55 --nb 1000 --me $EPOCH --margin 0.2 --lr 0.1 --ncomp 50 --minsubsize $MS --fe ./delme-lubm-hole --fout /var/scratch/uji300/trident/lubm1_hole_model


#python run_hole.py --fin /var/scratch/uji300/trident/lubm1  --test-all 55 --nb 1000 --me $EPOCH --margin 0.2 --lr 0.1 --ncomp 50 --subcreate --minsubsize $MS --subalgo avg --fout /var/scratch/uji300/trident/lubm1_hole_model

python run_hole.py --fin /var/scratch/uji300/trident/lubm1  --test-all 55 --nb 1000 --me $EPOCH --margin 0.2 --lr 0.1 --ncomp 50 --subtest --minsubsize $MS --subalgo avg --fout "./lubm1-HolE-epochs-10-sub_algo-avg-tau-5.mod" --fsub "./lubm1-HolE-epochs-10-sub_algo-avg-tau-5.sub"
