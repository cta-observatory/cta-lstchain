#!/usr//bin/env python

"""
 Onsite script for creating drs4 pedestal file to be run as a command line:

 --> onsite_create_calibration_file -h

"""

import argparse
from pathlib import Path
from lstchain.io.data_management import *
import lstchain.visualization.plot_drs4 as drs4

# parse arguments
parser = argparse.ArgumentParser(description='Create DRS4 pedestal file',
                                 formatter_class = argparse.ArgumentDefaultsHelpFormatter)
required = parser.add_argument_group('required arguments')
optional = parser.add_argument_group('optional arguments')

required.add_argument('-r', '--run_number', help="Run number with drs4 pedestals",
                      type=int, required=True)
optional.add_argument('-v', '--prod_version', help="Version of the production",
                      type=int, default=0)
optional.add_argument('-m', '--max_events', help="Number of events to be processed",
                      type=int, default=8000)
optional.add_argument('-b','--base_dir', help="Base dir for the output directory tree",
                      type=str, default='/fefs/aswg/data/real')

args = parser.parse_args()
run = args.run_number
prod_id = 'v%02d'%args.prod_version
max_events = args.max_events
base_dir = args.base_dir


def main():
    print(f"\n--> Start calculating DRS4 pedestals from run {run}\n")

    try:
        # verify input file
        file_list=sorted(Path(f"{base_dir}/R0").rglob(f'*{run}.0000*'))
        if len(file_list) == 0:
            print(f">>> Error: Run {run} not found\n")
            raise NameError()
        else:
            input_file = f"{file_list[0]}"

        # find date
        input_dir, name = os.path.split(os.path.abspath(input_file))
        path, date = input_dir.rsplit('/', 1)

        # verify and make output dir
        output_dir = f"{base_dir}/calibration/{date}/{prod_id}"
        if not os.path.exists(output_dir):
            print(f"--> Create directory {output_dir}")
            os.makedirs(dir, exist_ok=True)

        # make log dir
        log_dir = f"{output_dir}/log"
        if not os.path.exists(log_dir):
            print(f"--> Create directory {log_dir}")
            os.makedirs(dir, exist_ok=True)

        # define output file
        output_file = f"{output_dir}/drs4_pedestal.Run{run}.0000.fits"
        if os.path.exists(output_file):
            print(f">>> Output file {output_file} exists already. ")
            if query_yes_no("Do you want to remove it?"):
                os.remove(output_file)
            else:
                print(f">>> Exit")
                exit(1)

        # run lstchain script
        cmd = f"lstchain_data_create_drs4_pedestal_file --input_file {input_file} " \
              f"--output_file {output_file} --max_events {max_events}"

        os.system(cmd)

        # plot and save some results
        plot_file=f"{output_dir}/log/drs4_pedestal.Run{run}.0000.pdf"
        print(f"\n--> PRODUCING PLOTS in {plot_file} ...")
        drs4.plot_pedestals(input_file, output_file, run, plot_file, tel_id=1, offset_value=300)

        print("\n--> END")



    except Exception as e:
        print(f"\n >>> Exception: {e}")


if __name__ == '__main__':
    main()