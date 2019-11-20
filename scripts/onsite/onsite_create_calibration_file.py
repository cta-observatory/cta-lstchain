#!/usr//bin/env python

"""

 Onsite script for creating a flat-field calibration file file to be run as a command line:

 --> onsite_create_calibration_file

"""

import argparse
from pathlib import Path
from lstchain.io.data_management import *
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
optional.add_argument('-b','--base_dir', help="Base dir for the output directory tree",type=str, default='/fefs/aswg/data/real')

args = parser.parse_args()
run = args.run_number
ped_run = args.pedestal_run
prod_id = 'v%02d'%args.version
stat_events = args.statistics
base_dir = args.base_dir
max_events = 1000000


def main():

    print(f"\n--> Start calculating calibration from run {run}")

    try:
        # verify input file
        file_list=sorted(Path(f"{base_dir}/R0").rglob(f'*{run}.0000*'))
        if len(file_list) == 0:
            print(f">>> Error: Run {run} not found\n")
            raise NameError()
        else:
            input_file = file_list[0]

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

        # define output and log file
        output_file = f"{output_dir}/calibration.Run{run}.0000.hdf5"
        log_file = f"{output_dir}/log/calibration.Run{run}.0000.log"
        print(f"\n--> Output file {output_file}")
        if os.path.exists(output_file):
            if query_yes_no(">>> Output file exists already. Do you want to remove it?"):
                os.remove(output_file)
            else:
                exit(1)

        print(f"\n--> Log file {log_file}")

        # define config file
        config_file = os.path.join(os.path.dirname(__file__), "../../lstchain/data/onsite_camera_calibration_param.json")
        if not os.path.exists(config_file):
            print(f">>> Config file {config_file} do not exists. \n Exit ")
            exit(1)
        print(f"\n--> Config file {config_file}")

        # run lstchain script
        cmd = f"lstchain_data_create_calibration_file " \
              f"--input_file={input_file} --output_file={output_file} --pedestal_file={pedestal_file} " \
              f"--FlatFieldCalculator.sample_size={stat_events} --PedestalCalculator.sample_size={stat_events}  " \
              f"--EventSource.max_events={max_events} --config={config_file}  >  {log_file} 2>&1"
        #print(cmd)
        print("\n--> RUNNING...")

        os.system(cmd)

        # plot and save some results
        plot_file=f"{output_dir}/log/calibration.Run{run}.0000.pdf"
        print(f"\n--> PRODUCING PLOTS in {plot_file} ...")
        calib.read_file(output_file)
        calib.plot_all(calib.ped_data, calib.ff_data, calib.calib_data, run, plot_file)
        print("\n--> END")

    except Exception as e:
        print(f"\n >>> Exception: {e}")



if __name__ == '__main__':
    main()