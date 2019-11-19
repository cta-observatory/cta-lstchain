#!/bin/sh

# core job to run cmd $1 inside conda environement

source /local/home/lstanalyzer/.bashrc
conda activate cta


$1