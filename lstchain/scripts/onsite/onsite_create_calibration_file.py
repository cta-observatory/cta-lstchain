#!/usr//bin/env python

"""

 Onsite script for creating a flat-field calibration file file to be run as a command line:

 --> onsite_create_calibration_file

"""

import argparse
import os
import sys
from pathlib import Path
from lstchain.io.data_management import query_yes_no
import lstchain.visualization.plot_calib as calib
import lstchain
import subprocess
import pymongo

def none_or_str(value):
    if value == "None":
        return None
    return value

# parse arguments
parser = argparse.ArgumentParser(description='Create flat-field calibration files',
                                 formatter_class = argparse.ArgumentDefaultsHelpFormatter)
required = parser.add_argument_group('required arguments')
optional = parser.add_argument_group('optional arguments')

required.add_argument('-r', '--run_number', help="Run number if the flat-field data",
                      type=int, required=True)
optional.add_argument('-p', '--pedestal_run', help="Pedestal run to be used. If None, it looks for the pedestal run of the date of the FF data.",
                      type=none_or_str)

version=lstchain.__version__.rsplit('.post',1)[0]

optional.add_argument('-v', '--prod_version', help="Version of the production",
                      default=f"v{version}")
optional.add_argument('-s', '--statistics', help="Number of events for the flat-field and pedestal statistics",
                      type=int, default=10000)
optional.add_argument('-b','--base_dir', help="Root dir for the output directory tree", type=str, default='/fefs/aswg/data/real')

optional.add_argument('--time_run', help="Run for time calibration. If None, search the last time run before or equal the FF run", type=none_or_str)
optional.add_argument('--sys_date',
                      help="Date of systematic correction file (format YYYYMMDD). \n"
                           "Default: automatically search the best date \n",
                      type = none_or_str)
optional.add_argument('--no_sys_correction',
                      help="Systematic corrections are not applied. \n",
                      action='store_true',
                      default=False)
optional.add_argument('--output_base_name', help="Base of output file name (change only for debugging)", default="calibration")

optional.add_argument('--sub_run', help="sub-run to be processed.", type=int, default=0)
optional.add_argument('--min_ff', help="Min FF intensity cut in ADC.", type=float)
optional.add_argument('--max_ff', help="Max FF intensity cut in ADC.", type=float)
optional.add_argument('-f','--filters', help="Calibox filters", type=none_or_str)
optional.add_argument('--tel_id', help="telescope id. Default = 1", type=int, default=1)
default_config=os.path.join(os.path.dirname(__file__), "../../data/onsite_camera_calibration_param.json")
optional.add_argument('--config', help="Config file", default=default_config)
optional.add_argument('--mongodb', help="Mongo data-base connection", default="mongodb://10.200.10.101:27017/")
optional.add_argument('-y', '--yes', action="store_true", help='Do not ask interactively for permissions, assume true')

args = parser.parse_args()
run = args.run_number
ped_run = args.pedestal_run
prod_id = args.prod_version
stat_events = args.statistics
base_dir = args.base_dir
time_run = args.time_run
sys_date = args.sys_date
no_sys_correction = args.no_sys_correction
output_base_name = args.output_base_name
sub_run = args.sub_run
tel_id = args.tel_id
config_file = args.config
mongodb = args.mongodb
yes = args.yes

def main():

    # looks for the filter values in the database if not given
    if args.filters is None:
        filters = search_filter(run)
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
    if not os.path.exists(config_file):
        print(f"Config file {config_file} does not exists. \n")
        sys.exit(1)

    print(f"\n--> Config file {config_file}")

    # verify input file
    file_list=sorted(Path(f"{base_dir}/R0").rglob(f'*{run}.{sub_run:04d}*'))
    if len(file_list) == 0:
        print(f"Run {run} not found")
        sys.exit(2)
    else:
        input_file = file_list[0]
    print(f"\n--> Input file: {input_file}")

    # find date
    input_dir, name = os.path.split(os.path.abspath(input_file))
    path, date = input_dir.rsplit('/', 1)

    # verify output dir
    output_dir = f"{base_dir}/monitoring/PixelCalibration/calibration/{date}/{prod_id}"
    if not os.path.exists(output_dir):
        print(f"--> Create directory {output_dir}")
        os.makedirs(output_dir, exist_ok=True)

    # make log dir
    log_dir = f"{output_dir}/log"
    if not os.path.exists(log_dir):
        print(f"--> Create directory {log_dir}")
        os.makedirs(log_dir, exist_ok=True)

    # search the summary file info
    run_summary_path = f"{base_dir}/monitoring/RunSummary/RunSummary_{date}.ecsv"
    if not os.path.exists(run_summary_path):
        print(f"Night summary file {run_summary_path} does not exist")
        sys.exit(3)

    print(f"\n--> Use run summary {run_summary_path}")
    # pedestal base dir
    ped_dir = f"{base_dir}/monitoring/PixelCalibration/drs4_baseline/"

    # search the pedestal file of the same date
    if ped_run is None:
        # else search the pedestal file of the same date
        file_list = sorted(Path(f"{ped_dir}/{date}/{prod_id}/").rglob('drs4_pedestal*.0000.fits'))
        if len(file_list) == 0:
            print(f"No pedestal file found for date {date}")
            sys.exit(4)
        if len(file_list) > 1:
            print(f"Too many pedestal files found for date {date}: {file_list}, choose one run")
            sys.exit(5)
        else:
            pedestal_file = file_list[0]

    # else, if given, search a specific pedestal run
    else:
        file_list = sorted(Path(f"{ped_dir}").rglob(f'*/{prod_id}/drs4_pedestal.Run{ped_run}.0000.fits'))
        if len(file_list) == 0:
            print(f"Pedestal file from run {ped_run} not found")
            sys.exit(6)
        else:
            pedestal_file = file_list[0]

    print(f"\n--> Pedestal file: {pedestal_file}")

    # search for time calibration file
    time_file = None
    time_dir = f"{base_dir}/monitoring/PixelCalibration/drs4_time_sampling_from_FF"

    # search the last time run before or equal to the calibration run
    if time_run is None:
        file_list = sorted(Path(f"{time_dir}").rglob(f'*/{prod_id}/time_calibration.Run*.0000.h5'))

        if len(file_list) == 0:
            print(f"No time calibration file found in the data tree for prod {prod_id}")
            sys.exit(7)
        else:
            for file in file_list:
                run_in_list = file.stem.rsplit("Run")[1].rsplit('.')[0]
                if int(run_in_list) <= run:
                    time_file = file
                else:
                    break

        if time_file is None:
            print(f"No time calibration file found before run {run} for prod {prod_id}")
            sys.exit(8)

    # if given, search a specific time file
    else:
        file_list = sorted(Path(f"{time_dir}").rglob(f'*/{prod_id}/time_calibration.Run{time_run:05d}.0000.h5'))
        if len(file_list) == 0:
            print(f"Time calibration file from run {time_run} not found")
            sys.exit(9)
        else:
            time_file = file_list[0]

    if not os.path.exists(time_file):
        print(f"Time calibration file {time_file} does not exist")
        sys.exit(10)

    print(f"\n--> Time calibration file: {time_file}")

    sys_dir = f"{base_dir}/monitoring/PixelCalibration/ffactor_systematics/"

    # define systematic correction file
    if no_sys_correction:
        systematics_file = None

    else:
        # use specific sys corrections
        if sys_date is not None:
            systematics_file = f"{sys_dir}/{sys_date}/{prod_id}/ffactor_systematics_{sys_date}.h5"

        # search the first sys correction file before the run,
        # if nothing before, use the first found
        else:
            dir_list = sorted(Path(sys_dir).rglob(f"*/{prod_id}/ffactor_systematics*"))
            if len(dir_list) == 0:
                print(f"No systematic correction file found for production {prod_id} in {sys_dir}")
                sys.exit(11)
            else:
                sys_date_list = sorted([file.parts[-3] for file in dir_list],reverse=True)
                selected_date = next((day for day in sys_date_list if day <= date), sys_date_list[-1])
                systematics_file = f"{sys_dir}/{selected_date}/{prod_id}/ffactor_systematics_{selected_date}.h5"

        if not os.path.exists(systematics_file):
            print(f"F-factor systematics correction file {systematics_file} does not exist\n")
            sys.exit(12)

    print(f"\n--> F-factor systematics correction file: {systematics_file}")

# define charge file names
    print("\n***** PRODUCE CHARGE CALIBRATION FILE ***** ")

    if filters is not None:
        filter_info=f"_filters_{filters}"
    else:
        filter_info = ""

    # remember there are no systematic corrections
    prefix = ""
    if no_sys_correction:
        prefix = "no_sys_corrected_"

    output_name = f"{prefix}{output_base_name}{filter_info}.Run{run:05d}.{sub_run:04d}"

    output_file = f"{output_dir}/{output_name}.h5"
    print(f"\n--> Output file {output_file}")

    log_file = f"{output_dir}/log/{output_name}.log"
    print(f"\n--> Log file {log_file}")

    if os.path.exists(output_file):
        remove = False

        if not yes and os.getenv('SLURM_JOB_ID') is None:
            remove = query_yes_no(">>> Output file exists already. Do you want to remove it?")

        if yes or remove:
            os.remove(output_file)
            os.remove(log_file)
        else:
            print("\n--> Output file exists already. Stop")
            sys.exit(1)

    #
    # produce ff calibration file
    #

    cmd = [
        "lstchain_create_calibration_file",
        f"--input_file={input_file}",
        f"--output_file={output_file}",
        "--EventSource.default_trigger_type=tib",
        f"--EventSource.min_flatfield_adc={min_ff}",
        f"--EventSource.max_flatfield_adc={max_ff}",
        f"--LSTCalibrationCalculator.systematic_correction_path={systematics_file}",
        f"--LSTEventSource.EventTimeCalculator.run_summary_path={run_summary_path}",
        f"--LSTEventSource.LSTR0Corrections.drs4_time_calibration_path={time_file}",
        f"--LSTEventSource.LSTR0Corrections.drs4_pedestal_path={pedestal_file}",
        f"--FlatFieldCalculator.sample_size={stat_events}",
        f"--PedestalCalculator.sample_size={stat_events}",
        f"--config={config_file}",
        f"--log-file={log_file}",
        "--log-file-level=DEBUG",
    ]

    print("\n--> RUNNING...")
    subprocess.run(cmd, check=True)

    # plot and save some results
    plot_file=f"{output_dir}/log/{output_name}.pdf"

    print(f"\n--> PRODUCING PLOTS in {plot_file} ...")
    calib.read_file(output_file,tel_id)
    calib.plot_all(calib.ped_data, calib.ff_data, calib.calib_data, run, plot_file)

    print("\n--> END")

def search_filter(run):
    """read the employed filters form mongodb"""
    filters = None
    try:

        myclient = pymongo.MongoClient(mongodb)

        mydb = myclient["CACO"]
        mycol = mydb["RUN_INFORMATION"]
        mydoc = mycol.find({"run_number": {"$eq": run}})
        for x in mydoc:
            w1 = int(x["cbox"]["wheel1 position"])
            w2 = int(x["cbox"]["wheel2 position"])
            filters=f"{w1:1d}{w2:1d}"

    except Exception as e:
        raise IOError(
            "--> No mongo DB filter information."
            " You must pass the filters by argument: -f [filters]"
        ) from e

    return filters

def define_FF_selection_range(filters):
    """ return the range of charges to select the FF events """

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


    return min_ff, max_ff

if __name__ == '__main__':
    main()
