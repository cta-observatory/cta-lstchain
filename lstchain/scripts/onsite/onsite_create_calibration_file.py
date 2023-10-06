#!/usr//bin/env python

"""

 Onsite script for creating a flat-field calibration file file to be run as a command line:

 --> onsite_create_calibration_file

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
    DEFAULT_CONFIG,
    CAT_A_PIXEL_DIR,
    create_pro_symlink,
    find_r0_subrun,
    find_pedestal_file,
    find_run_summary,
    find_systematics_correction_file,
    find_time_calibration_file,
    find_filter_wheels,
)

# parse arguments
parser = argparse.ArgumentParser(description='Create flat-field calibration files',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
required = parser.add_argument_group('required arguments')
optional = parser.add_argument_group('optional arguments')

required.add_argument('-r', '--run_number', help="Run number if the flat-field data",
                      type=int, required=True)
optional.add_argument('-p', '--pedestal_run',
                      help="Pedestal run to be used. If None, it looks for the pedestal run of the date of the FF data.",
                      type=int)

version = lstchain.__version__

optional.add_argument('-v', '--prod_version', help="Version of the production",
                      default=f"v{version}")
optional.add_argument('-s', '--statistics', help="Number of events for the flat-field and pedestal statistics",
                      type=int, default=10000)
optional.add_argument('-b', '--base_dir', help="Root dir for the output directory tree", type=Path,
                      default=DEFAULT_BASE_PATH)
optional.add_argument('--r0-dir', help="Root dir for the input r0 tree. By default, <base_dir>/R0 will be used",
                      type=Path)

optional.add_argument('--time_run',
                      help="Run for time calibration. If None, search the last time run before or equal the FF run",
                      type=int)
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

optional.add_argument('--sub_run', help="sub-run to be processed.", type=int, default=0)
optional.add_argument('--min_ff', help="Min FF intensity cut in ADC.", type=float)
optional.add_argument('--max_ff', help="Max FF intensity cut in ADC.", type=float)
optional.add_argument('-f', '--filters', help="Calibox filters")
optional.add_argument('--tel_id', help="telescope id. Default = 1", type=int, default=1)

optional.add_argument('--config', help="Config file", default=DEFAULT_CONFIG, type=Path)

optional.add_argument('--mongodb', help="Mongo data-base (CACO DB) connection.", default="mongodb://10.200.10.161:27018/")
optional.add_argument('-y', '--yes', action="store_true", help='Do not ask interactively for permissions, assume true')
optional.add_argument('--no_pro_symlink', action="store_true",
                      help='Do not update the pro dir symbolic link, assume true')

optional.add_argument(
    '--flatfield-heuristic', action='store_const', const=True, dest="use_flatfield_heuristic",
    help=(
        "If given, try to identify flatfield events from the raw data."
        " Should be used only for data from before 2022"
    )
)
optional.add_argument(
    '--no-flatfield-heuristic', action='store_const', const=False, dest="use_flatfield_heuristic",
    help=(
        "If given, do not to identify flatfield events from the raw data."
        " Should be used only for data from before 2022"
    )
)


def main():
    args, remaining_args = parser.parse_known_args()
    run = args.run_number
    prod_id = args.prod_version
    stat_events = args.statistics
    time_run = args.time_run
    sys_date = args.sys_date
    no_sys_correction = args.no_sys_correction
    output_base_name = args.output_base_name
    sub_run = args.sub_run
    tel_id = args.tel_id
    config_file = args.config
    yes = args.yes
    pro_symlink = not args.no_pro_symlink

    # looks for the filter values in the database if not given
    if args.filters is None:
        filters = find_filter_wheels(run, args.mongodb)
    else:
        filters = args.filters

    if filters is None:
        sys.exit(f"Missing filter value for run {run}. \n")

    # define the FF selection cuts
    if args.min_ff is None or args.max_ff is None:
        min_ff, max_ff = define_FF_selection_range(filters)
    else:
        min_ff, max_ff = args.min_ff, args.max_ff

    print(f"\n--> Start calculating calibration from run {run}, filters {filters}")

    # verify config file
    if not config_file.exists():
        raise IOError(f"Config file {config_file} does not exists. \n")

    print(f"\n--> Config file {config_file}")

    # verify input file
    r0_dir = args.r0_dir or args.base_dir / 'R0'
    input_file = find_r0_subrun(run, sub_run, r0_dir)
    date = input_file.parent.name
    print(f"\n--> Input file: {input_file}")

    # verify output dir
    calib_dir = args.base_dir / CAT_A_PIXEL_DIR
    output_dir = calib_dir / "calibration" / date / prod_id
    if not output_dir.exists():
        print(f"--> Create directory {output_dir}")
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

    # search the summary file info
    run_summary_path = find_run_summary(date, args.base_dir)
    print(f"\n--> Use run summary {run_summary_path}")

    pedestal_file = find_pedestal_file(pro, args.pedestal_run, date=date, base_dir=args.base_dir)
    print(f"\n--> Pedestal file: {pedestal_file}")

    # search for time calibration file
    time_file = find_time_calibration_file(pro, run, time_run, args.base_dir)
    print(f"\n--> Time calibration file: {time_file}")


    # define systematic correction file
    if no_sys_correction:
        systematics_file = None
    else:
        systematics_file = find_systematics_correction_file(pro, date, sys_date, args.base_dir)

    print(f"\n--> F-factor systematics correction file: {systematics_file}")

    # define charge file names
    print("\n***** PRODUCE CHARGE CALIBRATION FILE ***** ")

    if filters is not None:
        filter_info = f"_filters_{filters}"
    else:
        filter_info = ""

    # remember there are no systematic corrections
    prefix = "no_sys_corrected_" if no_sys_correction else ""

    output_name = f"{prefix}{output_base_name}{filter_info}.Run{run:05d}.{sub_run:04d}"

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
        "lstchain_create_calibration_file",
        f"--input_file={input_file}",
        f"--output_file={output_file}",
        "--LSTEventSource.default_trigger_type=tib",
        f"--EventSource.min_flatfield_adc={min_ff}",
        f"--EventSource.max_flatfield_adc={max_ff}",
        f"--LSTCalibrationCalculator.systematic_correction_path={systematics_file}",
        f"--LSTEventSource.EventTimeCalculator.run_summary_path={run_summary_path}",
        f"--LSTEventSource.LSTR0Corrections.drs4_time_calibration_path={time_file}",
        f"--LSTEventSource.LSTR0Corrections.drs4_pedestal_path={pedestal_file}",
        f"--LSTEventSource.use_flatfield_heuristic={args.use_flatfield_heuristic}",
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
    calib.plot_calibration_results(mon.pedestal, mon.flatfield, mon.calibration, run, plot_file,"Cat-A")

    print("\n--> END")


def define_FF_selection_range(filters):
    """ return the range of charges to select the FF events """

    try:
        if filters is None:
            raise ValueError("Filters are not defined")
        # give standard values if standard filters
        if filters == '52':
            min_ff = 3000
            max_ff = 12000

        else:

            # ... recuperate transmission value of all the filters
            transm_file = os.path.join(os.path.dirname(__file__), "../../data/filters_transmission.dat")

            f = open(transm_file, 'r')
            # skip header
            f.readline()
            trasm = {}
            for line in f:
                columns = line.split()
                trasm[columns[0]] = float(columns[1])

            if trasm[filters] > 0.001:
                min_ff = 4000
                max_ff = 1000000

            elif trasm[filters] <= 0.001 and trasm[filters] > 0.0005:
                min_ff = 1200
                max_ff = 12000
            else:
                min_ff = 200
                max_ff = 5000

    except Exception as e:
        print(f"\n >>> Exception: {e}")
        raise IOError("--> No FF selection range information")

    return min_ff, max_ff


if __name__ == '__main__':
    main()
