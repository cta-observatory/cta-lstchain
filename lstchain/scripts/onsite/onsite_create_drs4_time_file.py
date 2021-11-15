#!/usr//bin/env python

"""

 Onsite script for creating a drs4 time sampling correction file to be run as a command line:

 --> onsite_create_drs4_time_file

"""

import argparse
import os
from pathlib import Path
import lstchain
import subprocess

# parse arguments
parser = argparse.ArgumentParser(description='Create time calibration files',
                                 formatter_class = argparse.ArgumentDefaultsHelpFormatter)
required = parser.add_argument_group('required arguments')
optional = parser.add_argument_group('optional arguments')

required.add_argument('-r', '--run_number', help="Run number if the flat-field data",
                      type=int, required=True)
version=lstchain.__version__.rsplit('.post',1)[0]
optional.add_argument('-v', '--prod_version', help="Version of the production",
                      default=f"v{version}")
optional.add_argument('-p', '--pedestal_run', help="Pedestal run to be used. If None, it looks for the pedestal run of the date of the FF data.",type=int)

optional.add_argument('-s', '--statistics', help="Number of events for the flat-field and pedestal statistics",
                      type=int, default=20000)
optional.add_argument('-b','--base_dir', help="Root dir for the output directory tree",type=str, default='/fefs/aswg/data/real')
optional.add_argument('--sub_run', help="sub-run to be processed.", type=int, default=0)
default_config=os.path.join(os.path.dirname(__file__), "../../data/onsite_camera_calibration_param.json")
optional.add_argument('--config', help="Config file", default=default_config)


args = parser.parse_args()
run = args.run_number
ped_run = args.pedestal_run
prod_id = args.prod_version
stat_events = args.statistics
base_dir = args.base_dir
sub_run = args.sub_run
config_file = args.config
if config_file is None:
    config_file = os.path.join(os.path.dirname(__file__), "../../data/onsite_camera_calibration_param.json")

max_events = 1000000

def main():

    print(f"\n--> Start calculating drs4 time corrections from run {run}")

    try:
        # verify config file
        if not os.path.exists(config_file):
            raise IOError(f"Config file {config_file} does not exists. \n")

        print(f"\n--> Config file {config_file}")

        # verify input file
        file_list=sorted(Path(f"{base_dir}/R0").rglob(f'*{run}.{sub_run:04d}*'))
        if len(file_list) == 0:
            raise IOError(f"Run {run} not found\n")
        else:
            input_file = file_list[0]
        print(f"\n--> Input file: {input_file}")

        # find date
        input_dir, name = os.path.split(os.path.abspath(input_file))
        path, date = input_dir.rsplit('/', 1)

        # verify output dir
        output_dir = f"{base_dir}/monitoring/PixelCalibration/drs4_time_sampling_from_FF/{date}/{prod_id}"
        if not os.path.exists(output_dir):
            if not os.path.exists(output_dir):
                print(f"--> Create directory {output_dir}")
                os.makedirs(output_dir, exist_ok=True)

        # search the summary file info
        run_summary_path = f"{base_dir}/monitoring/RunSummary/RunSummary_{date}.ecsv"
        if not os.path.exists(run_summary_path):
            raise IOError(f"Night summary file {run_summary_path} does not exist\n")

        # pedestal base dir
        ped_dir = f"{base_dir}/monitoring/PixelCalibration/drs4_baseline/"

        # search the pedestal file of the same date
        if ped_run is None:
            # else search the pedestal file of the same date
            file_list = sorted(Path(f"{ped_dir}/{date}/{prod_id}/").rglob('drs4_pedestal*.0000.fits'))
            if len(file_list) == 0:
                raise IOError(f"No pedestal file found for date {date}\n")
            if len(file_list) > 1:
                raise IOError(f"Too many pedestal files found for date {date}: {file_list}, choose one run\n")
            else:
                pedestal_file = file_list[0]

        # else, if given, search a specific pedestal run
        else:
            file_list = sorted(Path(f"{ped_dir}").rglob(f'*/{prod_id}/drs4_pedestal.Run{ped_run}.0000.fits'))
            if len(file_list) == 0:
                raise IOError(f"Pedestal file from run {ped_run} not found\n")
            else:
                pedestal_file = file_list[0]

        print(f"\n--> Pedestal file: {pedestal_file}")

        #
        # produce drs4 time calibration file
        #
        time_file = f"{output_dir}/time_calibration.Run{run:05d}.0000.h5"

        print(f"\n--> PRODUCING TIME CALIBRATION in {time_file} ...")
        cmd = f"lstchain_data_create_time_calibration_file --input-file {input_file} " \
              f"--output-file {time_file} --config {config_file} " \
              f"--run-summary-path={run_summary_path} " \
              f"--pedestal-file {pedestal_file} --max-events {stat_events}"
        print("\n--> RUNNING...")
        subprocess.run(cmd.split())

        print("\n--> END")

    except Exception as e:
        print(f"\n >>> Exception: {e}")


if __name__ == '__main__':
    main()
