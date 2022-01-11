#!/usr//bin/env python

"""
 Onsite script for creating drs4 pedestal file to be run as a command line:

 --> onsite_create_calibration_file -h

"""

import argparse
import subprocess
from pathlib import Path
import lstchain
from lstchain.io.data_management import query_yes_no
import lstchain.visualization.plot_drs4 as drs4
import os


# parse arguments
parser = argparse.ArgumentParser(description='Create DRS4 pedestal file',
                                 formatter_class = argparse.ArgumentDefaultsHelpFormatter)
required = parser.add_argument_group('required arguments')
optional = parser.add_argument_group('optional arguments')

required.add_argument('-r', '--run_number', help="Run number with drs4 pedestals",
                      type=int, required=True)
version=lstchain.__version__
optional.add_argument('-v', '--prod_version', help="Version of the production",
                      default=f"v{version}")
optional.add_argument('-m', '--max_events', help="Number of events to be processed",
                      type=int, default=20000)
optional.add_argument('-b','--base_dir', help="Base dir for the output directory tree",
                      type=str, default='/fefs/aswg/data/real')
optional.add_argument('--r0-dir', help="Root dir for the input r0 tree. By default, <base_dir>/R0 will be used", type=Path)
optional.add_argument('--tel_id', help="telescope id. Default = 1",
                      type=int, default=1)
optional.add_argument('-y', '--yes', action="store_true", help='Do not ask interactively for permissions, assume true')
optional.add_argument('--no_pro_symlink', action="store_true", help='Do not update the pro dir symbolic link, assume true')
parser.add_argument(
    '--no-progress',
    action='store_true',
    help='Do not display a progress bar during event processing'
)


args = parser.parse_args()
run = args.run_number
prod_id = args.prod_version
max_events = args.max_events
base_dir = args.base_dir
tel_id = args.tel_id
yes = args.yes
pro_symlink = not args.no_pro_symlink
calib_dir=f"{base_dir}/monitoring/PixelCalibration/LevelA"


def main():
    print(f"\n--> Start calculating DRS4 pedestals from run {run}\n")

    # verify input file
    r0_dir = args.r0_dir or Path(args.base_dir) / 'R0'
    file_list = sorted(r0_dir.rglob(f'*{run:05d}.0000*'))
    if len(file_list) == 0:
        print(f">>> Error: Run {run} not found under {base_dir}/R0 \n")
        raise NameError()
    else:
        input_file = f"{file_list[0]}"

    # find date
    input_dir, name = os.path.split(os.path.abspath(input_file))
    path, date = input_dir.rsplit('/', 1)

    # verify and make output dir
    output_dir = f"{calib_dir}/drs4_baseline/{date}/{prod_id}"
    if not os.path.exists(output_dir):
        print(f"--> Create directory {output_dir}")
        os.makedirs(output_dir, exist_ok=True)

    # update the default production directory
    if pro_symlink:
        pro="pro"
        pro_dir = f"{output_dir}/../{pro}"
        if os.path.exists(pro_dir):
            os.remove(pro_dir)
        os.symlink(prod_id,pro_dir)
        print("\n--> Use symbolic link pro")
    else:
        pro=prod_id

    # make log dir
    log_dir = f"{output_dir}/log"
    if not os.path.exists(log_dir):
        print(f"--> Create directory {log_dir}")
        os.makedirs(log_dir, exist_ok=True)

    # define output file
    output_file = f"{output_dir}/drs4_pedestal.Run{run:05d}.0000.h5"

    if os.path.exists(output_file):
        remove = False

        if not yes and os.getenv('SLURM_JOB_ID') is None:
            remove = query_yes_no(">>> Output file exists already. Do you want to remove it?")

        if yes or remove:
            os.remove(output_file)
        else:
            print("\n--> Output file exists already. Stop")
            exit(1)

    # run lstchain script
    cmd = [
        "lstchain_create_drs4_pedestal_file",
        f"--input={input_file}",
        f"--output={output_file}",
        f"--max-events={max_events}",
    ]

    if args.no_progress:
        cmd.append("--no-progress")

    subprocess.run(cmd, check=True)

    # plot and save some results
    plot_file=f"{output_dir}/log/drs4_pedestal.Run{run:05d}.0000.pdf"
    print(f"\n--> PRODUCING PLOTS in {plot_file} ...")
    drs4.plot_pedestals(input_file, output_file, run, plot_file, tel_id=tel_id, offset_value=400)

    print("\n--> END")


if __name__ == '__main__':
    main()
