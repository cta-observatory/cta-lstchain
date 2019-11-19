#!/bin/sh
# slurm core job - send a job with the command passed as arg1 on every line in the file passed as arg2


source /local/home/lstanalyzer/.bashrc
conda activate cta


CMD=$1
filelist=$2

for file in `cat $filelist`;
do
    echo "processing $file";
    $CMD -f $file
done
