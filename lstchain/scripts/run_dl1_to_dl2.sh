#!/bin/bash

#SBATCH --exclude=cp50,cp51,cp52,cp03,cp07

RUN=1836
FILES=/fefs/aswg/data/real/DL1/20200118/v0.4.3_v00/dl1_LST-1.1.Run0${RUN}.*.fits.h5


`mkdir /fefs/aswg/workspace/maria.bernardos/LSTanalysis/real/DL2/20200118/Run${RUN}`

for FILE in $FILES
do
    srun -n 1 -o out${RUN}.txt --exclude=cp50,cp51,cp52,cp03,cp07 --mem=10g python lstchain_mc_dl1_to_dl2.py -f $FILE -p /fefs/aswg/data-tmp/models/20190415/south_pointing/20191109_v0.1_v00/trained_models/ -o /fefs/aswg/workspace/maria.bernardos/LSTanalysis/real/DL2/20200118/ &

done
