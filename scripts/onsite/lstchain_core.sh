#!/bin/sh

### core job to run a pipeline step
# outdir and file should be passed to the script
# example: ./core_job.sh /path_to/gamma_20deg_0deg.simtel.gz /path_to/results


conda activate cta


filelist=$1
outdir=$2

echo "filelist $filelist"
echo "outdir $outdir"


for file in `cat $filelist`;
do
    echo "processing $file";
    python /fefs/home/thomas.vuillaume/software/cta-observatory/cta-lstchain/scripts/lst-r0_to_dl1.py -f $file -o $outdir
done
