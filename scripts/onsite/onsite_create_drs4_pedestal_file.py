#!/usr//bin/env python

"""
 F. Cassol, 12/11/2019

 Onsite script for creating drs4 pedestal file to be run as a command line:

 --> onsite_create_calibration_file -h

"""

import argparse

from lstchain.io.data_management import *
from pathlib import Path


# to be changed with the right one
base_dir = '/ctadata/franca/fefs/aswg/real'

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
                      type=int, default=5000)

args = parser.parse_args()
run = args.run_number
prod_id = 'v%02d'%args.version
max_events = args.max_events


def main():
    print(f"\n--> Start calculating DRS4 pedestals from run {run}\n")

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

        # verify and make output dir
        output_dir = f"{base_dir}/calibration/{date}/{prod_id}"
        check_and_make_dir(output_dir)

        # make log dir
        log_dir = f"{output_dir}/log"
        check_and_make_dir(log_dir)

        # define output file
        output_file = f"{output_dir}/drs4_pedestal.Run{run}.0000.fits"
        print(f"--> Output file {output_file}")

        # run lstchain script
        cmd = f"lstchain_data_create_pedestal_file --input_file {input_file} " \
              f"--output_file {output_file} --max_events {max_events}"
        os.system(cmd)

        print("\n--> END")

    except Exception as e:
        print(f"\n >>> Exception: {e}")


if __name__ == '__main__':
    main()