#!/bin/bash
# Create test calibration files (used for unit tests in ctapipe_io_lst / lstchain)
# Needs to be run on the IT cluster, e.g. ctan-cp01 or a slurm job
set -eux -o pipefail

TIME_CALIB_DATE="20191124"
TIME_CALIB_PEDESTAL_RUN="01623"
TIME_CALIB_PEDCAL_RUN="01625"

R0_PATH="/fefs/aswg/data/real/R0/"
DATE="20200218"
PEDESTAL_RUN="02005"
PEDCAL_RUN="02006"

version=v$(python -c 'import lstchain; print(lstchain.__version__, end="")')
OUTPUT_PATH="/fefs/aswg/workspace/$USER/lstchain-dev/$version"
echo $OUTPUT_PATH
mkdir -p $OUTPUT_PATH
mkdir -p $OUTPUT_PATH/monitoring/RunSummary

lstchain_create_run_summary \
	-d "${DATE}" \
	--r0-path "$R0_PATH" \
	-o $OUTPUT_PATH/monitoring/RunSummary

lstchain_create_run_summary \
	-d "${TIME_CALIB_DATE}" \
	--r0-path "$R0_PATH" \
	-o $OUTPUT_PATH/monitoring/RunSummary

onsite_create_drs4_pedestal_file \
 	-r $TIME_CALIB_PEDESTAL_RUN \
 	-v $version \
 	--r0-dir=$R0_PATH \
 	--base_dir=$OUTPUT_PATH \
 	--yes \
 	-m 20000

onsite_create_drs4_pedestal_file \
 	-r $PEDESTAL_RUN \
 	-v $version \
 	--r0-dir=$R0_PATH \
 	--base_dir=$OUTPUT_PATH \
 	--yes \
 	-m 20000

onsite_create_drs4_time_file \
 	-r $TIME_CALIB_PEDCAL_RUN \
 	-p $TIME_CALIB_PEDESTAL_RUN \
 	-v $version \
 	--r0-dir=$R0_PATH \
 	--base_dir=$OUTPUT_PATH

onsite_create_calibration_file \
 	-r $PEDCAL_RUN \
 	-p $PEDESTAL_RUN \
 	-v $version \
 	--time_run=$TIME_CALIB_PEDCAL_RUN \
 	--r0-dir=$R0_PATH \
 	--base_dir=$OUTPUT_PATH \
	--yes \
 	-f 52 \
	--no_sys_correction
