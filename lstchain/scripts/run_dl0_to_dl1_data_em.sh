#!/bin/bash

#DATES=( 20191123 20191124 20191126 20191129 ) #First Crab Campaign

DATES=( 20191126 ) #First Crab Campaign
#Create outputs if they dont exist

`mkdir -p /fefs/aswg/workspace/maria.bernardos/LSTanalysis/EManalysis/real/DL1/1stCrabCampaign/`
`mkdir -p /fefs/aswg/workspace/maria.bernardos/LSTanalysis/EManalysis/real/DL1/2ndCrabCampaign/`
`mkdir -p /fefs/aswg/workspace/maria.bernardos/LSTanalysis/EManalysis/real/DL1/3rdCrabCampaign/`

CAMPAIGN="1stCrabCampaign"
for date in "${DATES[@]}"
do
    FILES=/fefs/aswg/data/real/R0/$date/LST-1.1.Run01649*.fits.fz
    PATH_OUTPUT=/fefs/aswg/workspace/maria.bernardos/LSTanalysis/EManalysis/real/DL1/$CAMPAIGN/$date
    SCRIPT=/home/maria.bernardos/GitHub/cta-lstchain/lstchain/scripts/lstchain_data_r0_to_dl1_em.py
    CALIBRATION=/fefs/aswg/data/real/calibration/$date/v00/calibration*
    PEDESTAL=/fefs/aswg/data/real/calibration/$date/v00/drs4*
    TIME_CAL=/fefs/aswg/data/real/calibration/$date/v00/time*

    POINTING=/fefs/aswg/workspace/maria.bernardos/LSTanalysis/DrivePositioning/drive_log_${date:2:2}_${date:4:2}_${date:6:2}.txt

    for f in $FILES
    do
        b=$(basename $f)
        srun -o dl0_to_dl1.out python $SCRIPT -f $f -o $PATH_OUTPUT -conf /home/maria.bernardos/GitHub/cta-lstchain/lstchain/data/lstchain_standard_config.json -pedestal $PEDESTAL -calib $CALIBRATION -time_calib $TIME_CAL -pointing $POINTING &
    done
done
