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
from ctapipe_io_lst.event_time import read_night_summary


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


args = parser.parse_args()
run = args.run_number
ped_run = '%05d'%args.pedestal_run
print(ped_run)
prod_id = 'v%02d'%args.version
stat_events = args.statistics
base_dir = args.base_dir
default_time_run = args.default_time_run
ff_calibration = args.ff_calibration
tel_id = args.tel_id
sub_run = '%04d'%args.sub_run

max_events = 1000000


def main():

    print(f"\n--> Start calculating calibration from run {run}")

    try:
        # verify input file
        file_list=sorted(Path(f"{base_dir}/R0").rglob(f'*{run}.{sub_run}*'))
        if len(file_list) == 0:
            print(f">>> Error: Run {run} not found\n")
            raise NameError()
        else:
            input_file = file_list[0]
        print(f"\n--> Input file: {input_file}")

        # find date
        input_dir, name = os.path.split(os.path.abspath(input_file))
        path, date = input_dir.rsplit('/', 1)

        # verify output dir
        output_dir = f"{base_dir}/calibration/{date}/{prod_id}"
        if not os.path.exists(output_dir):
            print(f">>> Error: The output directory {output_dir} do not exist")
            print(f">>>        You must create the drs4 pedestal file to create it.\n Exit")
            exit(0)

        # search the pedestal calibration file

        pedestal_file = f"{output_dir}/drs4_pedestal.Run{ped_run}.0000.fits"
        if not os.path.exists(pedestal_file):
            print(f">>> Error: The pedestal file {pedestal_file} do not exist.\n Exit")
            exit(0)

        # search the summary file info
        file_list = sorted(Path(f"{base_dir}/monitoring/NightSummary/").rglob(f'*Nig*{date}.txt'))
        night_log = str(file_list[0])
        summary = read_night_summary(night_log)
        print(f"\n--> Summary file {night_log}")
        run_info = summary.loc[run]

        # define config file
        config_file = os.path.join(os.path.dirname(__file__), "../../data/onsite_camera_calibration_param.json")
        if not os.path.exists(config_file):
            print(f">>> Config file {config_file} do not exists. \n Exit ")
            exit(1)
        print(f"\n--> Config file {config_file}")

        #
        # produce drs4 time calibration file
        #

        time_file = f"{output_dir}/time_calibration.Run{run}.0000.hdf5"
        print(f"\n***** PRODUCE TIME CALIBRATION FILE ***** ")
        if default_time_run is 0:
            print(f"\n--> PRODUCING TIME CALIBRATION in {time_file} ...")
            cmd = f"lstchain_data_create_time_calibration_file  --input-file {input_file} " \
                  f"--output-file {time_file} --config {config_file} " \
                  f"--dragon-reference-time={int(run_info['ucts_t0_dragon'])} " \
                  f"--dragon-reference-counter={int(run_info['dragon_counter0'])} " \
                  f"--pedestal-file {pedestal_file} 2>&1"
            print("\n--> RUNNING...")
            os.system(cmd)
        else:
            # otherwise perform a link to the default time calibration file
            print(f"\n--> PRODUCING LINK TO DEFAULT TIME CALIBRATION (run {default_time_run})")
            file_list = sorted(
                Path(f"{base_dir}/calibration/").rglob(f'*/{prod_id}/time_calibration.Run{default_time_run}*'))

            if len(file_list) == 0:
                print(f">>> Error: time calibration file for run {default_time_run} not found\n")
                raise NameError()
            else:
                time_calibration_file = file_list[0]
                cmd = f"ln -sf {time_calibration_file} {time_file}"
                os.system(cmd)

        print(f"\n--> Time calibration file: {time_file}")

        # define charge file names
        print(f"\n***** PRODUCE CHARGE CALIBRATION FILE ***** ")
        output_file = f"{output_dir}/calibration.Run{run}.{sub_run}.hdf5"
        log_file = f"{output_dir}/log/calibration.Run{run}.{sub_run}.log"
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
                  f"--LSTEventSource.EventTimeCalculator.dragon_reference_time={int(run_info['ucts_t0_dragon'])} " \
                  f"--LSTEventSource.EventTimeCalculator.dragon_reference_counter={int(run_info['dragon_counter0'])} " \
                  f"--LSTEventSource.LSTR0Corrections.drs4_time_calibration_path={time_file} " \
                  f"--LSTEventSource.LSTR0Corrections.drs4_pedestal_path={pedestal_file} " \
                  f"--FlatFieldCalculator.sample_size={stat_events} --PedestalCalculator.sample_size={stat_events} " \
                  f"--config={config_file}  >  {log_file} 2>&1"

            print("\n--> RUNNING...")
            os.system(cmd)

            # plot and save some results
            plot_file=f"{output_dir}/log/calibration.Run{run}.{sub_run}.pedestal.Run{ped_run}.0000.pdf"
            print(f"\n--> PRODUCING PLOTS in {plot_file} ...")
            calib.read_file(output_file,tel_id)
            calib.plot_all(calib.ped_data, calib.ff_data, calib.calib_data, run, plot_file)

        print("\n--> END")

    except Exception as e:
        print(f"\n >>> Exception: {e}")


if __name__ == '__main__':
    main()
