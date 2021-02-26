#!/bin/bash
#SBATCH --output=array_%A_%a.out
#SBATCH --error=array_%A_%a.err
#SBATCH --array=1-1000
#SBATCH --ntasks=1

#directory='/fefs/aswg/data/mc/DL0/20190415/gamma-diffuse/south_pointing/'
directory='/fefs/aswg/data/mc/DL0/20190415/proton/south_pointing/'

nfiles=`ls $directory | wc -l`
echo "Running analysis on $nfiles files"

#OUT_FOLDER='/fefs/aswg/workspace/maria.bernardos/LSTanalysis/EManalysis/mc/DL1/20190415/gamma-diffuse/south_pointing/files/'
#OUT_LOGS='/fefs/aswg/workspace/maria.bernardos/LSTanalysis/EManalysis/mc/DL1/20190415/gamma-diffuse/south_pointing/job_logs/'

OUT_FOLDER='/fefs/aswg/workspace/maria.bernardos/LSTanalysis/EManalysis/mc/DL1/20190415/proton/south_pointing/files/'
OUT_LOGS='/fefs/aswg/workspace/maria.bernardos/LSTanalysis/EManalysis/mc/DL1/20190415/proton/south_pointing/job_logs/'

SCRIPT='/home/maria.bernardos/GitHub/cta-lstchain/lstchain/scripts/lstchain_mc_r0_to_dl1_em.py'

#PREF='gamma-diffuse_20deg_180deg_run'
#SUF='___cta-prod3-demo-2147m-LaPalma-baseline-mono.simtel.gz'

PREF='proton_20deg_180deg_run'
SUF='___cta-prod3-demo-2147m-LaPalma-baseline-mono.simtel.gz'

RUN=$(($SLURM_ARRAY_TASK_ID+4000))

python $SCRIPT -f ${directory}${PREF}${RUN}$SUF -o $OUT_FOLDER -conf '/fefs/aswg/workspace/maria.bernardos/LSTanalysis/EManalysis/lstchain_no_time_pars_config.json' &

#for i in $(seq 1 $nfiles)
#do
#    srun -o ${OUT_LOGS}'run'${i}'.o' python $SCRIPT -f ${directory}${PREF}${i}$SUF -o $OUT_FOLDER -conf '/fefs/aswg/workspace/maria.bernardos/LSTanalysis/EManalysis/lstchain_no_time_pars_config.json' &
#done
