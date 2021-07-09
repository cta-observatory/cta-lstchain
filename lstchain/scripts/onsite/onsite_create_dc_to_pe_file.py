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
import lstchain

# parse arguments
parser = argparse.ArgumentParser(description='Create flat-field calibration files',
                                 formatter_class = argparse.ArgumentDefaultsHelpFormatter)
required = parser.add_argument_group('required arguments')
optional = parser.add_argument_group('optional arguments')

required.add_argument('-r', '--run_number', help="Run number if the flat-field data",
                      type=int, required=True)
optional.add_argument('-p', '--pedestal_run', help="Pedestal run to be used. If None, it looks for the pedestal run of the date of the FF data.",
                      type=int)

version,subversion=lstchain.__version__.rsplit('.post',1)
optional.add_argument('-v', '--prod_version', help="Version of the production",
                      default=f"v{version}")
optional.add_argument('-s', '--statistics', help="Number of events for the flat-field and pedestal statistics",
                      type=int, default=10000)
optional.add_argument('-b','--base_dir', help="Root dir for the output directory tree", type=str, default='/fefs/aswg/data/real')
optional.add_argument('--time_run', help="run time calibration. If None, search the last time run before the FF run", type=int)
optional.add_argument('--sub_run', help="sub-run to be processed.", type=int, default=0)
optional.add_argument('--min_ff', help="Min FF intensity cut in ADC.", type=float, default=4000)
optional.add_argument('--max_ff', help="Max FF intensity cut in ADC.", type=float, default=12000)
optional.add_argument('--tel_id', help="telescope id. Default = 1", type=int, default=1)
default_config=os.path.join(os.path.dirname(__file__), "../../data/onsite_camera_calibration_param.json")
optional.add_argument('--config', help="Config file", default=default_config)


args = parser.parse_args()
run = args.run_number
ped_run = args.pedestal_run
prod_id = args.prod_version
stat_events = args.statistics
base_dir = args.base_dir
time_run = args.time_run
min_ff = args.min_ff
max_ff = args.max_ff
sub_run = args.sub_run
tel_id = args.tel_id
config_file = args.config

max_events = 1000000

def main():

    print(f"\n--> Start calculating calibration from run {run}")

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
        output_dir = f"{base_dir}/monitoring/CameraCalibration/dc_to_pe/{date}/{prod_id}"
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
            raise IOError(f"Night summary file {run_summary_path} does not exist\n")

        # pedestal base dir
        ped_dir = f"{base_dir}/monitoring/CameraCalibration/drs4_baseline/"

        # search the pedestal file of the same date
        if ped_run is None:
            # else search the pedestal file of the same date
            file_list = sorted(Path(f"{ped_dir}/{date}/{prod_id}/").rglob(f'drs4_pedestal*.0000.fits'))
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

        # search for time calibration file
        time_dir = f"{base_dir}/monitoring/CameraCalibration/drs4_time_sampling_from_FF"

        # search the last time run before the calibration run
        if time_run is None:
            file_list = sorted(Path(f"{time_dir}").rglob(f'*/{prod_id}/time_calibration.Run*.0000.h5'))
            if len(file_list) == 0:
                raise IOError(f"No time calibration file found in the data tree for prod {prod_id}\n")
            if len(file_list) >1:
                for file in file_list:
                    run_in_list = file.stem.rsplit("Run")[1].rsplit('.')[0]
                    if int(run_in_list) > run:
                        break
                    else:
                        time_file = file
            if time_file is None:
                raise IOError(f"No time calibration file found before run {run} for prod {prod_id}\n")

        # if given, search a specific time file
        else:
            file_list = sorted(Path(f"{time_dir}").rglob(f'*/{prod_id}/time_calibration.Run{time_run:05d}.0000.h5'))
            if len(file_list) == 0:
                raise IOError(f"Time calibration file from run {time_run} not found\n")
            else:
                time_file = file_list[0]

        print(f"\n--> Time calibration file: {time_file}")

        # define charge file names
        print(f"\n***** PRODUCE CHARGE CALIBRATION FILE ***** ")
        output_file = f"{output_dir}/calibration.Run{run:05d}.{sub_run:04d}.h5"
        log_file = f"{output_dir}/log/calibration.Run{run:05d}.{sub_run:04d}.log"
        print(f"\n--> Output file {output_file}")
        #"""
        if os.path.exists(output_file):
            if query_yes_no(">>> Output file exists already. Do you want to remove it?"):
                os.remove(output_file)
            else:
                print(f"\n--> Stop")
                exit(1)
        #"""
        print(f"\n--> Log file {log_file}")

        #
        # produce ff calibration file
        #

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
        plot_file=f"{output_dir}/log/dc_to_pe_calibration.Run{run:05d}.{sub_run:04d}.pdf"

        print(f"\n--> PRODUCING PLOTS in {plot_file} ...")
        calib.read_file(output_file,tel_id)
        calib.plot_all(calib.ped_data, calib.ff_data, calib.calib_data, run, plot_file)

        print("\n--> END")

    except Exception as e:
        print(f"\n >>> Exception: {e}")


if __name__ == '__main__':
    main()
