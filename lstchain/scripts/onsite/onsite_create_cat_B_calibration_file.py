#!/usr//bin/env python

"""

 Onsite script for creating a Cat-B flat-field calibration file to be run as a command line:

 --> onsite_create_cat_B_calibration_file

"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

import lstchain
import lstchain.visualization.plot_calib as calib
from lstchain.io.data_management import query_yes_no
from lstchain.io import read_calibration_file
from lstchain.onsite import (
    DEFAULT_BASE_PATH,
    DEFAULT_CONFIG_CAT_B_CALIB,
    CAT_B_PIXEL_DIR,
    create_pro_symlink,
    find_interleaved_subruns,
    find_r0_subrun,
    find_systematics_correction_file,
    find_calibration_file,
    find_filter_wheels,
)

MAX_SUBRUNS = 100000

# parse arguments
parser = argparse.ArgumentParser(description='Create flat-field calibration Cat-B files',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
required = parser.add_argument_group('required arguments')
optional = parser.add_argument_group('optional arguments')

required.add_argument('-r', '--run_number', help="Run number of interleaved data",
                      type=int, required=True)

optional.add_argument('-c', '--catA_calibration_run',
                      help="Cat-A calibration run to be used. If None, it looks for the calibration run of the date of the interleaved data.",
                      type=int)

optional.add_argument('-s', '--statistics', help="Number of events for the flat-field and pedestal statistics",
                      type=int, default=2500)
optional.add_argument('-b', '--base_dir', help="Root dir for the output directory tree", type=Path,
                      default=DEFAULT_BASE_PATH)
optional.add_argument('--interleaved-dir', help="Root dir for the input interleaved files. By default, <base_dir>/DL1/date/version/interleaved will be used",
                      type=Path)
optional.add_argument('--r0-dir', help="Root dir for the input r0 tree. By default, <base_dir>/R0 will be used",
                      type=Path)

optional.add_argument(
    '--sys_date',
    help=(
        "Date of systematic correction file (format YYYYMMDD). \n"
        "Default: automatically search the best date \n"
    ),
)
optional.add_argument('--no_sys_correction',
                      help="Systematic corrections are not applied. \n",
                      action='store_true',
                      default=False)
optional.add_argument('--output_base_name', help="Base of output file name (change only for debugging)",
                      default="calibration")

optional.add_argument('--n_subruns', help="Number of subruns to be processed",
                      type=int, default=MAX_SUBRUNS)

optional.add_argument('-f', '--filters', help="Calibox filters")
optional.add_argument('--tel_id', help="telescope id. Default = 1", type=int, default=1)

optional.add_argument('--config', help="Config file", default=DEFAULT_CONFIG_CAT_B_CALIB, type=Path)
optional.add_argument('--mongodb', help="Mongo data-base (CACO DB) connection.", default="mongodb://10.200.10.161:27018/")

optional.add_argument('-y', '--yes', action="store_true", help='Do not ask interactively for permissions, assume true')
optional.add_argument('--no_pro_symlink', action="store_true",
                      help='Do not update the pro dir symbolic link, assume true')


def main():
    args, remaining_args = parser.parse_known_args()
    run = args.run_number
    n_subruns = args.n_subruns
    prod_id = f"v{lstchain.__version__}"
    stat_events = args.statistics
    
    sys_date = args.sys_date
    no_sys_correction = args.no_sys_correction
    tel_id = args.tel_id
    config_file = args.config
    yes = args.yes
    pro_symlink = not args.no_pro_symlink
    r0_dir = args.r0_dir or args.base_dir / 'R0'

    # looks for the filter values in the database if not given
    if args.filters is None:
        filters = find_filter_wheels(run, args.mongodb)
    else:
        filters = args.filters

    if filters is None:
        sys.exit(f"Missing filter value for run {run}. \n")

    print(f"\n--> Start calculating Cat-B calibration from run {run}, filters {filters}")

    # verify config file
    if not config_file.exists():
        raise IOError(f"Config file {config_file} does not exists. \n")

    print(f"\n--> Config file {config_file}")

    # look in R0 to find the date
    r0_list = find_r0_subrun(run,0,r0_dir)
    date = r0_list.parent.name

    # find input path
    ver = prod_id.rsplit(".")
    input_path = args.interleaved_dir or args.base_dir / 'DL1'/ f"{date}/{ver[0]}.{ver[1]}/interleaved" 

    # verify input file
    input_files = find_interleaved_subruns(run, input_path)

    print(f"\n--> Found {len(input_files)} interleaved subruns in {input_path}")
    if n_subruns < MAX_SUBRUNS:
        print(f"--> Process {n_subruns} subruns")

    # verify output dir
    calib_dir = args.base_dir / CAT_B_PIXEL_DIR
    output_dir = calib_dir / "calibration" / date / prod_id
    if not output_dir.exists():
        print(f"\n--> Create directory {output_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)

    if pro_symlink:
        pro = "pro"
        create_pro_symlink(output_dir)
    else:
        pro = prod_id

    # make log dir
    log_dir = output_dir / "log"
    if not log_dir.exists():
        print(f"--> Create directory {log_dir}")
        log_dir.mkdir(parents=True, exist_ok=True)

    cat_A_calib_file = find_calibration_file(pro, args.catA_calibration_run, date=date, base_dir=args.base_dir)
    print(f"\n--> Cat-A calibration file: {cat_A_calib_file}")

    # define systematic correction file
    if no_sys_correction:
        systematics_file = None
    else:
        systematics_file = find_systematics_correction_file(pro, date, sys_date, args.base_dir)

    print(f"\n--> F-factor systematics correction file: {systematics_file}")

    # define charge file names
    print("\n***** PRODUCE CAT_B CALIBRATION FILE ***** ")

    if filters is not None:
        filter_info = f"_filters_{filters}"
    else:
        filter_info = ""
    
    input_file_pattern=f"interleaved_LST-1.Run{run:05d}.*.h5"
    output_name = f"cat_B_calibration{filter_info}.Run{run:05d}"

    output_file = output_dir / f'{output_name}.h5'
    print(f"\n--> Output file {output_file}")

    log_file = log_dir / f"{output_name}.log"
    print(f"\n--> Log file {log_file}")

    if output_file.exists():
        remove = False

        if not yes and os.getenv('SLURM_JOB_ID') is None:
            remove = query_yes_no(">>> Output file exists already. Do you want to remove it?")

        if yes or remove:
            os.remove(output_file)
            os.remove(log_file)
        else:
            print("\n--> Output file exists already. Stop")
            exit(1)

    #
    # produce ff calibration file
    #

    cmd = [
        "lstchain_create_cat_B_calibration_file",
        f"--input_path={input_path}",
        f"--output_file={output_file}",
        f"--input_file_pattern={input_file_pattern}",
        f"--n_subruns={n_subruns}",
        f"--cat_A_calibration_file={cat_A_calib_file}",
        f"--LSTCalibrationCalculator.systematic_correction_path={systematics_file}",
        f"--FlasherFlatFieldCalculator.sample_size={stat_events}",
        f"--PedestalIntegrator.sample_size={stat_events}",
        f"--config={config_file}",
        f"--log-file={log_file}",
        "--log-file-level=INFO",
        *remaining_args,
    ]
    
    print("\n--> RUNNING...")
    subprocess.run(cmd, check=True)

    # plot and save some results
    plot_file = f"{output_dir}/log/{output_name}.pdf"

    print(f"\n--> PRODUCING PLOTS in {plot_file} ...")
    mon = read_calibration_file(output_file, tel_id)
    calib.plot_calibration_results(mon.pedestal, mon.flatfield, mon.calibration, run, plot_file,"Cat-B")

    print("\n--> END")

if __name__ == '__main__':
    main()

