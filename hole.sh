#!/bin/sh
EPOCH=50
MS=10
DB="lubm1"

if [ $# -ge 1 ]; then
    EPOCH=$1
fi

if [ $# -ge 2 ]; then
    MS=$2
fi

if [ $# -ge 3 ]; then
    DB=$3
fi

echo "# args = $#"
echo "EPOCH = $EPOCH"
echo "MS = $MS"
echo "DB = $DB"


python run_hole.py --fin /var/scratch/uji300/trident/$DB  --test-all 100 --nb 1000 --me $EPOCH --margin 0.2 --lr 0.1 --ncomp 50 --minsubsize $MS --fout "/var/scratch/uji300/hole/$DB""_hole_model_epochs_$EPOCH"


#python run_hole.py --fin /var/scratch/uji300/trident/lubm1  --test-all 55 --nb 1000 --me $EPOCH --margin 0.2 --lr 0.1 --ncomp 50 --subcreate --minsubsize $MS --subalgo avg --fout "/var/scratch/uji300/hole/lubm1_hole_model_epochs_$EPOCH"

#python run_hole.py --fin /var/scratch/uji300/trident/lubm1  --test-all 55 --nb 1000 --me $EPOCH --margin 0.2 --lr 0.1 --ncomp 50 --subtest --minsubsize $MS --subalgo avg --fout "./lubm1-HolE-epochs-$EPOCH-sub_algo-avg-tau-$MS"".mod" --fsub "./lubm1-HolE-epochs-$EPOCH-sub_algo-avg-tau-$MS"".sub"
