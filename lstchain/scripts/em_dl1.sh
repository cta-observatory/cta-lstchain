#!/bin/bash

#for i in {1..10}
#do
#    srun python lstchain_mc_r0_to_dl1_em.py -f /fefs/aswg/data/mc/DL0/20190415/gamma-diffuse/south_pointing/gamma-diffuse_20deg_180deg_run${i}___cta-prod3-demo-2147m-LaPalma-baseline-mono.simtel.gz -o /fefs/aswg/workspace/maria.bernardos/LSTanalysis/DL1/20190415/gamma-diffuse/south_pointing/20200122_em_hillas_test/ &

#done


for i in {11..100}
do
    srun python lstchain_mc_r0_to_dl1_em.py -f /fefs/aswg/data/mc/DL0/20190415/proton/south_pointing/proton_20deg_180deg_run${i}___cta-prod3-demo-2147m-LaPalma-baseline-mono.simtel.gz -o /fefs/aswg/workspace/maria.bernardos/LSTanalysis/DL1/20190415/proton/south_pointing/20200122_em_hillas_test/ &
done

#for i in {1..10}
#do
#    srun python lstchain_mc_r0_to_dl1_em.py -f /fefs/aswg/data/mc/DL0/20190415/gamma/south_pointing/gamma_20deg_180deg_run${i}___cta-prod3-demo-2147m-LaPalma-baseline-mono_off0.4.simtel.gz -o /fefs/aswg/workspace/maria.bernardos/LSTanalysis/DL1/20190415/gamma/south_pointing/20200122_em_hillas_test/ &

#done
