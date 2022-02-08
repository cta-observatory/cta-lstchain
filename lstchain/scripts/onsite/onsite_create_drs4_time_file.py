#!/usr//bin/env python

"""

 Onsite script for creating a drs4 time sampling correction file to be run as a command line:

 --> onsite_create_drs4_time_file

"""

import argparse
import os
import subprocess
from pathlib import Path

import lstchain
from lstchain.onsite import create_pro_symlink, find_r0_subrun, find_pedestal_file, find_run_summary

# parse arguments
parser = argparse.ArgumentParser(description='Create time calibration files',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
required = parser.add_argument_group('required arguments')
optional = parser.add_argument_group('optional arguments')

required.add_argument('-r', '--run_number', help="Run number if the flat-field data",
                      type=int, required=True)
version = lstchain.__version__
optional.add_argument('-v', '--prod_version', help="Version of the production",
                      default=f"v{version}")
optional.add_argument('-p', '--pedestal_run',
                      help="Pedestal run to be used. If None, it looks for the pedestal run of the date of the FF data.",
                      type=int)

optional.add_argument('-s', '--statistics', help="Number of events for the flat-field and pedestal statistics",
                      type=int, default=20000)
optional.add_argument('-b', '--base_dir', help="Root dir for the output directory tree", type=str,
                      default='/fefs/aswg/data/real')
optional.add_argument('--r0-dir', help="Root dir for the input r0 tree. By default, <base_dir>/R0 will be used",
                      type=Path)
optional.add_argument('--sub_run', help="sub-run to be processed.", type=int, default=0)
default_config = os.path.join(os.path.dirname(__file__), "../../data/onsite_camera_calibration_param.json")
optional.add_argument('--config', help="Config file", default=default_config)
optional.add_argument('--no_pro_symlink', action="store_true",
                      help='Do not update the pro dir symbolic link, assume true')
parser.add_argument(
    '--no-progress',
    action='store_true',
    help='Do not display a progress bar during event processing'
)

args = parser.parse_args()
run = args.run_number
ped_run = args.pedestal_run
prod_id = args.prod_version
stat_events = args.statistics
base_dir = args.base_dir
sub_run = args.sub_run
config_file = args.config
pro_symlink = not args.no_pro_symlink

if config_file is None:
    config_file = os.path.join(os.path.dirname(__file__), "../../data/onsite_camera_calibration_param.json")

max_events = 1000000
calib_dir = f"{base_dir}/monitoring/PixelCalibration/LevelA"


def main():
    print(f"\n--> Start calculating drs4 time corrections from run {run}")

    # verify config file
    if not os.path.exists(config_file):
        raise IOError(f"Config file {config_file} does not exists. \n")

    print(f"\n--> Config file {config_file}")

    # verify input file
    r0_dir = args.r0_dir or Path(args.base_dir) / 'R0'
    input_file = find_r0_subrun(run, sub_run, r0_dir)
    date = input_file.parent.name
    print(f"\n--> Input file: {input_file}")

    # verify output dir
    output_dir = f"{calib_dir}/drs4_time_sampling_from_FF/{date}/{prod_id}"

    if not os.path.exists(output_dir):
        print(f"--> Create directory {output_dir}")
        os.makedirs(output_dir, exist_ok=True)

    # update the default production directory
    if pro_symlink:
        pro = "pro"
        create_pro_symlink(output_dir)
    else:
        pro = prod_id

    run_summary_path = find_run_summary(date, args.base_dir)
    print(f"\n--> Use run summary {run_summary_path}")

    pedestal_file = find_pedestal_file(pro, args.ped_run, date=date, base_dir=args.base_dir)
    print(f"\n--> Pedestal file: {pedestal_file}")

    
    time_file = f"{output_dir}/time_calibration.Run{run:05d}.0000.h5"
    print(f"\n--> PRODUCING TIME CALIBRATION in {time_file} ...")
    cmd = [
        "lstchain_data_create_time_calibration_file",
        f"--input-file={input_file}",
        f"--output-file={time_file}",
        f"--config={config_file}",
        f"--run-summary-path={run_summary_path}",
        f"--pedestal-file={pedestal_file}",
        f"--max-events={stat_events}",
    ]

    if args.no_progress:
        cmd.append("--no-progress")

    print("\n--> RUNNING...")
    subprocess.run(cmd, check=True)

    print("\n--> END")


if __name__ == '__main__':
    main()
