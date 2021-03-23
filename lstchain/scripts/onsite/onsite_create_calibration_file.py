#!/usr//bin/env python

"""

 Onsite script for creating a flat-field calibration file file to be run as a command line:

 --> onsite_create_calibration_file

"""

import argparse
import os
from pathlib import Path
from lstchain.io.data_management import query_yes_no
import lstchain.visualization.plot_calib as calib

# parse arguments
parser = argparse.ArgumentParser(description='Create flat-field calibration files',
                                 formatter_class = argparse.ArgumentDefaultsHelpFormatter)
required = parser.add_argument_group('required arguments')
optional = parser.add_argument_group('optional arguments')

required.add_argument('-r', '--run_number', help="Run number if the flat-field data",
                      type=int, required=True)
required.add_argument('-p', '--pedestal_run', help="Run number of the drs4 pedestal run",
                      type=int, required=True)
optional.add_argument('-v', '--version', help="Version of the production",
                      type=int, default=0)
optional.add_argument('-s', '--statistics', help="Number of events for the flat-field and pedestal statistics",
                      type=int, default=10000)
optional.add_argument('-b','--base_dir', help="Root dir for the output directory tree",type=str, default='/fefs/aswg/data/real')
optional.add_argument('--default_time_run', help="If 0 time calibration is calculated otherwise create a link to the give run time calibration",type=int, default='1625')
optional.add_argument('--ff_calibration', help="Perform the charge calibration (yes/no)",type=str, default='yes')
optional.add_argument('--tel_id', help="telescope id. Default = 1", type=int, default=1)
optional.add_argument('--sub_run', help="sub-run to be processed. Default = 0", type=int, default=0)
optional.add_argument('--min_ff', help="Min FF intensity cut in ADC. Default = 4000", type=float, default=4000)
optional.add_argument('--max_ff', help="Max FF intensity cut in ADC. Default = 12000", type=float, default=12000)

args = parser.parse_args()
run = args.run_number
ped_run = args.pedestal_run
prod_id = 'v%02d'%args.version
stat_events = args.statistics
base_dir = args.base_dir
default_time_run = args.default_time_run
ff_calibration = args.ff_calibration
tel_id = args.tel_id
min_ff = args.min_ff
max_ff = args.max_ff
sub_run = args.sub_run

max_events = 1000000

def main():

    print(f"\n--> Start calculating calibration from run {run}")

    try:
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
        output_dir = f"{base_dir}/calibration/{date}/{prod_id}"
        if not os.path.exists(output_dir):
            raise IOError(f"Output directory {output_dir} does not exist\n")

        # search the pedestal calibration file
        pedestal_file = f"{output_dir}/drs4_pedestal.Run{ped_run:05d}.0000.fits"
        if not os.path.exists(pedestal_file):
            raise IOError(f"Pedestal file {pedestal_file} does not exist.\n ")

        # search the summary file info
        run_summary_path = f"{base_dir}/monitoring/RunSummary/RunSummary_{date}.ecsv"
        if not os.path.exists(run_summary_path):
            raise IOError(f"Night summary file {run_summary_path} does not exist\n")

        # define config file
        config_file = os.path.join(os.path.dirname(__file__), "../../data/onsite_camera_calibration_param.json")
        if not os.path.exists(config_file):
            raise IOError(f"Config file {config_file} does not exists. \n")

        print(f"\n--> Config file {config_file}")

        #
        # produce drs4 time calibration file
        #
        time_file = f"{output_dir}/time_calibration.Run{run:05d}.0000.h5"
        print(f"\n***** PRODUCE TIME CALIBRATION FILE ***** ")
        if default_time_run is 0:
            print(f"\n--> PRODUCING TIME CALIBRATION in {time_file} ...")
            cmd = f"lstchain_data_create_time_calibration_file  --input-file {input_file} " \
                  f"--output-file {time_file} --config {config_file} " \
                  f"--run-summary-path={run_summary_path} " \
                  f"--pedestal-file {pedestal_file} 2>&1"
            print("\n--> RUNNING...")
            os.system(cmd)
        else:
            # otherwise perform a link to the default time calibration file
            print(f"\n--> PRODUCING LINK TO DEFAULT TIME CALIBRATION (run {default_time_run})")
            file_list = sorted(
                Path(f"{base_dir}/calibration/").rglob(f'*/{prod_id}/time_calibration.Run{default_time_run}*'))

            if len(file_list) == 0:
                raise IOError(f"Time calibration file for run {default_time_run} not found\n")
            else:
                time_calibration_file = file_list[0]
                cmd = f"ln -sf {time_calibration_file} {time_file}"
                os.system(cmd)

        print(f"\n--> Time calibration file: {time_file}")

        # define charge file names
        print(f"\n***** PRODUCE CHARGE CALIBRATION FILE ***** ")
        output_file = f"{output_dir}/calibration.Run{run:05d}.{sub_run:04d}.h5"
        log_file = f"{output_dir}/log/calibration.Run{run:05d}.{sub_run:04d}.log"
        print(f"\n--> Output file {output_file}")
        if os.path.exists(output_file) and ff_calibration == 'yes':
            if query_yes_no(">>> Output file exists already. Do you want to remove it?"):
                os.remove(output_file)
            else:
                print(f"\n--> Stop")
                exit(1)

        print(f"\n--> Log file {log_file}")

        #
        # produce ff calibration file
        #
        if ff_calibration == 'yes':
            # run lstchain script
            cmd = f"lstchain_create_calibration_file " \
                  f"--input_file={input_file} --output_file={output_file} "\
                  f"--EventSource.max_events={max_events} " \
                  f"--EventSource.default_trigger_type=tib " \
                  f"--EventSource.min_flatfield_adc={min_ff} " \
                  f"--EventSource.max_flatfield_adc={max_ff} " \
                  f"--LSTEventSource.EventTimeCalculator.run_summary_path={run_summary_path} " \
                  f"--LSTEventSource.LSTR0Corrections.drs4_time_calibration_path={time_file} " \
                  f"--LSTEventSource.LSTR0Corrections.drs4_pedestal_path={pedestal_file} " \
                  f"--FlatFieldCalculator.sample_size={stat_events} --PedestalCalculator.sample_size={stat_events} " \
                  f"--config={config_file}  >  {log_file} 2>&1"

            print("\n--> RUNNING...")
            os.system(cmd)

            # plot and save some results
            plot_file=f"{output_dir}/log/calibration.Run{run:05d}.{sub_run:04d}.pedestal.Run{ped_run:05d}.0000.pdf"
            print(f"\n--> PRODUCING PLOTS in {plot_file} ...")
            calib.read_file(output_file,tel_id)
            calib.plot_all(calib.ped_data, calib.ff_data, calib.calib_data, run, plot_file)

        print("\n--> END")

    except Exception as e:
        print(f"\n >>> Exception: {e}")


if __name__ == '__main__':
    main()
