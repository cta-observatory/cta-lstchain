#!/bin/bash

#DATES=( 20191123 20191124 20191126 20191129 ) #First Crab Campaign
#DATES=(20200114 20200115 20200117 20200118 20200125 20200127 20200128 20200131 20200201 20200202) #Second Crab Campaign
<<<<<<< Updated upstream
#DATES=(20200213 20200215 20200217 20200218 20200227 20200228) #Third Crab Campaign
DATES=(20200227 20200228)
#Create outputs if they dont exist

#CAMPAIGN="1stCrabCampaign"
#CAMPAIGN="2ndCrabCampaign"
CAMPAIGN="3rdCrabCampaign"

`mkdir -p /fefs/aswg/workspace/maria.bernardos/LSTanalysis/real/DL2/$CAMPAIGN/v0.4.4_v00/`

sizeref=999999999
=======
DATES=(20200213 20200215 20200217 20200218 20200227 20200228) #Third Crab Campaign

#Create outputs if they dont exist

`mkdir -p /fefs/aswg/workspace/maria.bernardos/LSTanalysis/real/DL2/1stCrabCampaign/v0.4.4_v00/`
`mkdir -p /fefs/aswg/workspace/maria.bernardos/LSTanalysis/real/DL2/2ndCrabCampaign/v0.4.4_v00/`
`mkdir -p /fefs/aswg/workspace/maria.bernardos/LSTanalysis/real/DL2/3rdCrabCampaign/v0.4.4_v00/`

>>>>>>> Stashed changes

for date in "${DATES[@]}"
do
    FILES=/fefs/aswg/data/real/DL1/$date/v0.4.4_v00/*.fits.h5
    PATH_MODELS=/fefs/aswg/data/models/20190415/south_pointing/20200229_v0.4.4_TV01/
    #PATH_OUTPUT=/fefs/aswg/workspace/maria.bernardos/LSTanalysis/real/DL2/1stCrabCampaign/v0.4.4_v00/$date
    #PATH_OUTPUT=/fefs/aswg/workspace/maria.bernardos/LSTanalysis/real/DL2/2ndCrabCampaign/v0.4.4_v00/$date
<<<<<<< Updated upstream
    PATH_OUTPUT=/fefs/aswg/workspace/maria.bernardos/LSTanalysis/real/DL2/$CAMPAIGN/v0.4.4_v00/$date
=======
    PATH_OUTPUT=/fefs/aswg/workspace/maria.bernardos/LSTanalysis/real/DL2/3ndCrabCampaign/v0.4.4_v00/$date
>>>>>>> Stashed changes
    SCRIPT=/home/maria.bernardos/GitHub/cta-lstchain/lstchain/scripts/lstchain_mc_dl1_to_dl2.py

    for f in $FILES
    do
<<<<<<< Updated upstream
        b=$(basename $f)
        outfile=$PATH_OUTPUT"/dl2_"$b
        if [ ! -f $outfile ]
        then
            #echo $outfile" NOT EXISTS"
            srun -o out.txt --mem=50g python $SCRIPT -f $f -p $PATH_MODELS -o $PATH_OUTPUT &
        else
            size=$(wc -c $outfile | awk '{print $1}')
            if [ $((size)) -lt $((sizeref)) ]
            then
                #echo $size
                #echo $outfile
                srun -o out.txt --mem=50g python $SCRIPT -f $f -p $PATH_MODELS -o $PATH_OUTPUT &
            fi
        fi
        #srun -o out.txt --mem=50g python $SCRIPT -f $f -p $PATH_MODELS -o $PATH_OUTPUT &
=======
        srun -o out.txt --mem=50g python $SCRIPT -f $f -p $PATH_MODELS -o $PATH_OUTPUT &
>>>>>>> Stashed changes
    done
done
