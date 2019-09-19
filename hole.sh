#!/bin/sh
DB="lubm1"
EPOCH=50
MS=10
TOPK=5
SUBALGO="hole"
SUBTYPE="avg"

if [ $# -ge 1 ]; then
    DB=$1
fi

if [ $# -ge 2 ]; then
    EPOCH=$2
fi

if [ $# -ge 3 ]; then
    MS=$3
fi

if [ $# -ge 4 ]; then
    TOPK=$4
fi

if [ $# -ge 5 ]; then
    SUBALGO=$5
fi

if [ $# -ge 6 ]; then
    SUBTYPE=$6
fi

echo "# args = $#"
echo "DB = $DB"
echo "EPOCH = $EPOCH"
echo "MS = $MS"
echo "TOPK = $TOPK"
echo "SUBALGO = $SUBALGO"
echo "SUBTYPE = $SUBTYPE"

hole_embeddings_file="/var/scratch/uji300/hole/$DB""_hole_model_epochs_$EPOCH"
subgraph_embeddings_home="/var/scratch/uji300/hole/"

if [ ! -f $hole_embeddings_file ]; then
    echo "Embeddings not found training with HolE..."
    python run_hole.py --fin /var/scratch/uji300/trident/$DB  --test-all 100 --nb 1000 --me $EPOCH --margin 0.2 --lr 0.1 --ncomp 50 --fout $hole_embeddings_file
fi

subfile_name=$subgraph_embeddings_home"$DB-HolE-epochs-$EPOCH-$SUBTYPE-tau-$MS"".sub"
model_file_name=$subgraph_embeddings_home"$DB-HolE-epochs-$EPOCH-$SUBTYPE-tau-$MS"".mod"
if [ ! -f $subfile_name ] || [ ! -f $model_file_name ]; then
    echo "Generating subgraphs for $DB with mincard $MS and algo $SUBALGO..."
    python run_hole.py --fin /var/scratch/uji300/trident/$DB  --nb 1000 --me $EPOCH --margin 0.2 --lr 0.1 --ncomp 50 --subcreate --minsubsize $MS --subalgo $SUBALGO --subdistance $SUBTYPE --fout $hole_embeddings_file
fi

python run_hole.py --fin /var/scratch/uji300/trident/$DB  --nb 1000 --me $EPOCH --margin 0.2 --lr 0.1 --ncomp 50 --subtest --minsubsize $MS --subalgo $SUBALGO --subdistance $SUBTYPE --fout $model_file_name  --fsub $subfile_name --topk $TOPK
