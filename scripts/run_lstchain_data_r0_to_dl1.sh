#!/bin/bash
#SBATCH --export=all
#SBATCH --tasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2000




###########################################################
################   Extract input options   ################
###########################################################

RAWOPTS=$@
OPTS=`getopt -o d:r:p:th --long date:,run_number:,pointing:,pedestal:,calibration:,config:,prod_id:,test,help -n 'parse-options' -- "$@"`

if [ $? != 0 ] ; then echo "Failed parsing options." >&2 ; exit 1 ; fi

TEST=false

usage() { echo "Script to process r0 files up to dl1a+b h5 files. To be used along with sbatch.

IMPORTANT! cta environment has to be activate before running this script:
    conda activate cta
    conda activate /home/daniel.morcuende/.local/miniconda3/envs/lstchain-v2 (bug fixed)

Usage: $0 [-d|--date YYYY_MM_DD] [-r|--run_number XXXX]
          [-p|--pointing YY_MM_DD]
          [--pedestal <path to pedestal file>]
          [--calibration <path to calibration file>]
          [--config <path to config file>]
          [--prod_id <prod ID>]
          [-t|--test]
          [-h|--help]


Input arguments:
    -d YYYY_MM_DD, --date YYYY_MM_DD      Date of the run
    -r XXXX, --run_number XXXX            Number of the run
    -p, --pointing  YY_MM_DD              Date of the drive report to add pointing information
                                          (not added by default)
    --pedestal <path_to_pedestal>         Use a given pedestal file
    --calibration <path_to_calibration>   Use a given calibration file
    --config <path_to_config>             Use a given config file
    --prod_id <prod ID>                   Specify prod ID
    -t, --test                            Use a test config file with 100 max_events
    -h, --help                            Show this message

EXAMPLES:

  (1) sbatch run_lstchain_data_r0_to_dl1.sh -d 2019_10_28 -r 1555

  (2) sbatch run_lstchain_data_r0_to_dl1.sh -d 2019_10_28 -r 1555 -p 19_10_24

  (2) sbatch run_lstchain_data_r0_to_dl1.sh -d 2019_10_28 -r 1555 --pedestal <pedestal_file_path> --calibration <calib_file_path>
"
    1>&2; exit 1;
}

while true; do
  case "$1" in
    -d | --date )       DATE=$(echo $2 | sed 's/-//g' | sed 's/_//g'); shift 2 ;;
    -r | --run_number ) RUN_NUMBER=$2; shift 2 ;;
    -p | --pointing )   POINTING=$2; shift 2 ;;
    --pedestal )        PEDESTAL=$2; shift 2 ;;
    --calibration )     CALIBRATION=$2; shift 2 ;;
    --config )          CONFIG=$2; shift 2 ;;
    --prod_id )         PROD_ID=$2; shift 2 ;;
    -t | --test )       TEST=true; shift ;;
    -h | --help )       usage ;;
    -- )                shift; break ;;
    * )                 HELP=true; break ;;
  esac
done

if [ -z $DATE ] || [ -z $RUN_NUMBER ] || [ -z $PROD_ID ]; then
    echo "Date/run number/prod_id not specified. Please see the help."
    exit 1
fi

###########################################################
################      Define variables     ################
###########################################################

MAIN_PATH=/fefs/aswg
DATA_PATH=/fefs/onsite/data
CTASOFT=/home/maria.bernardos/GitHub/cta-lstchain/lstchain
SCRIPTS=$CTASOFT/scripts
CALIBRATION_PATH=$MAIN_PATH/data/real/calibration/$DATE/v00
input_path=$DATA_PATH/$DATE
output_directory=$MAIN_PATH/workspace/maria.bernardos/LSTanalysis/EManalysis/real/dl1/$DATE/$PROD_ID

echo "DATE OF OBSERVATIONS    = ${DATE}"
echo "RUN NUMBER              = ${RUN_NUMBER}"
echo "PROD_ID                 = ${PROD_ID}"

if [ -d "$output_directory" ]; then
    echo "prod_ID directory already exists. Please try a new one."
    exit 1
fi

###########################################################
################      Get config file      ################
###########################################################

if [ -z $CONFIG ]; then
    if [ "$TEST" = false ]; then
        echo "Using a standard config file."
        config_file=$MAIN_PATH/workspace/maria.bernardos/LSTanalysis/EManalysis/lstchain_no_time_pars_config.json
        echo " CONFIG FILE    : $config_file"
    else
        echo "Using a test config file to process less events."
        config_file=$MAIN_PATH/scripts-osa/config/config-test.json
        echo " CONFIG FILE    : $config_file"
    fi
else
    echo " Custom config file will be used:"
    config_file=$CONFIG
    echo " CONFIG FILE    : $config_file"
fi

###########################################################
#############       Get pedestal file        ##############
###########################################################

if [ -z $PEDESTAL ]; then
    # Look for required pedestal file
    for file in $CALIBRATION_PATH/*pedestal*fits; do
        if [[ -f "$file" ]]; then
            echo " Using the pedestal file from the sabe observation date."
            pedestal_drs4_file=$file
            echo " PEDESTAL FILE    : $pedestal_drs4_file"
        else
            echo " No pedestal file found for this observation date."
            echo " A default file will be used instead."
            pedestal_drs4_file=$MAIN_PATH/data/real/calibration/20191122/v00/drs4_pedestal.Run1599.0000.fits
            echo " PEDESTAL FILE    : $pedestal_drs4_file"
        fi
    done
else
    echo " Custom pedestal file will be used:"
    pedestal_drs4_file=$PEDESTAL
    echo " PEDESTAL FILE    : $pedestal_drs4_file"
fi

###########################################################
#############      Get calibration file      ##############
###########################################################

if [ -z $CALIBRATION ]; then
    # Look for required calibration file
    for file in $CALIBRATION_PATH/*calibration*hdf5; do
        if [[ -f "$file" ]]; then
            echo " Using the calibration file from the sabe observation date."
            calibration_coeff_file=$file
            echo " CALIBRATION FILE : $calibration_coeff_file"
        else
            echo " No file of calibration coefficients found for this date"
            echo " A default file will be used instead."
            calibration_coeff_file=$MAIN_PATH/data/real/calibration/20191122/v00/calibration.Run1600.0000.hdf5
            echo " CALIBRATION FILE : $calibration_coeff_file"
        fi
    done
else
    echo " Custom calibration file will be used:"
    calibration_coeff_file=$CALIBRATION
    echo " CALIBRATION FILE : $calibration_coeff_file"
fi

###########################################################
#####  Check whether add pointing or not & run script  ####
###########################################################

if [ -z $POINTING ]; then
    echo "Not adding pointing information."
    for file in $input_path/*1.1*${RUN_NUMBER}*fits.fz; do
        if [[ -f "$file" ]]; then
            srun -N 1 -n 1 python $SCRIPTS/lstchain_data_r0_to_dl1_em.py \
                -f $file \
                -o $output_directory \
                -pedestal $pedestal_drs4_file \
                -calib $calibration_coeff_file \
                -conf $config_file &
        else
            echo "ERROR: Runs not found for this date."
        fi

    done
else
    DRIVE_LOG=/fefs/home/lapp/DrivePositioning
    POINTING_PATH=$DRIVE_LOG/drive_log_$POINTING.txt

    echo "Adding pointing information from Drive log."
    echo " DRIVE REPORT  = $POINTING_PATH"
    for file in $input_path/*1.1*${RUN_NUMBER}*fits.fz; do
        if [[ -f "$file" ]]; then
            srun -N 1 -n 1 python $SCRIPTS/lstchain_data_r0_to_dl1_em.py \
                -f $file \
                -o $output_directory \
                -pedestal $pedestal_drs4_file \
                -calib $calibration_coeff_file \
                -conf $config_file \
                -pointing $POINTING_PATH &
        else
            echo "ERROR: Runs not found for this date."
        fi

    done
fi

wait
