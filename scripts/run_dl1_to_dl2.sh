#!/bin/bash
FILES=/fefs/aswg/data/real/DL1/20200115/v0.4.3_v00/dl1_LST-1.1.Run01795.*.fits.h5
PATH_MODELS=/fefs/aswg/data-tmp/models/20190415/south_pointing/20191109_v0.1_v00/trained_models/
PATH_OUTPUT=/fefs/aswg/workspace/maria.bernardos/LSTanalysis/real/DL2/20200115/
SCRIPT=/home/maria.bernardos/GitHub/cta-lstchain/lstchain/scripts/lstchain_mc_dl1_to_dl2.py

for f in $FILES
do
    srun --mem=50g python $SCRIPT -f $f -p $PATH_MODELS -o $PATH_OUTPUT &
done
