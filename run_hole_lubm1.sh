#!/bin/sh
EPOCH=50
MS=10

if [ $# -eq 1 ]; then
    EPOCH=$1;
fi

if [ $# -ge 2 ]; then
    MS=$2;
fi

echo "# args = $#"
echo "EPOCH = $EPOCH"
echo "MS = $MS"
#python run_hole.py --fin /var/scratch/uji300/trident/lubm1  --test-all 55 --nb 1000 --me $EPOCH --margin 0.2 --lr 0.1 --ncomp 50 --minsubsize $MS --fe ./delme-lubm-hole --fout "/var/scratch/uji300/hole/lubm1_hole_model_epochs_$EPOCH"


python run_hole.py --fin /var/scratch/uji300/trident/lubm1  --test-all 55 --nb 1000 --me $EPOCH --margin 0.2 --lr 0.1 --ncomp 50 --subcreate --minsubsize $MS --subalgo avg --fout "/var/scratch/uji300/hole/lubm1_hole_model_epochs_$EPOCH"

python run_hole.py --fin /var/scratch/uji300/trident/lubm1  --test-all 55 --nb 1000 --me $EPOCH --margin 0.2 --lr 0.1 --ncomp 50 --subtest --minsubsize $MS --subalgo avg --fout "./lubm1-HolE-epochs-$EPOCH-sub_algo-avg-tau-$MS"".mod" --fsub "./lubm1-HolE-epochs-$EPOCH-sub_algo-avg-tau-$MS"".sub"
