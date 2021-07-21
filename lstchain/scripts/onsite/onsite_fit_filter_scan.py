#!/usr//bin/env python

"""

 Onsite script to fit a filter scan production

 --> onsite_create_dc_to_pe_file

"""

import argparse
import os
from lstchain.io.data_management import query_yes_no
import lstchain


def none_or_str(value):
    if value == "None":
        return None
    return value

# parse arguments
parser = argparse.ArgumentParser(description='Create flat-field calibration files',
                                 formatter_class = argparse.ArgumentDefaultsHelpFormatter)
required = parser.add_argument_group('required arguments')
optional = parser.add_argument_group('optional arguments')

required.add_argument('-d', '--date', help="Date of the filter scan", required=True)

# config file is mandatory because it contains the list of input runs
required.add_argument('--config', help="Config file with run list", required=True)

version,subversion=lstchain.__version__.rsplit('.post',1)
optional.add_argument('-v', '--prod_version', help="Version of the production",
                      default=f"v{version}")
optional.add_argument('-b','--base_dir', help="Root dir for the output directory tree", type=str, default='/fefs/aswg/data/real')
optional.add_argument('--sub_run', help="sub-run to be processed.", type=int, default=0)

args = parser.parse_args()
date = args.date
prod_id = args.prod_version
base_dir = args.base_dir
sub_run = args.sub_run
config_file = args.config



def main():

    channel = ["HG", "LG"]

    try:
        # verify config file
        if not os.path.exists(config_file):
            raise IOError(f"Config file {config_file} does not exists. \n")

        print(f"\n--> Config file {config_file}")

        # verify input dir
        input_dir=f"{base_dir}/monitoring/CameraCalibration/dc_to_pe/{date}/{prod_id}"
        if not os.path.exists(input_dir):
            raise IOError(f"Input directory {input_dir} not found\n")

        print(f"\n--> Input directory {input_dir}")

        # verify output dir
        output_dir = f"{base_dir}/monitoring/CameraCalibration/filter_scan/{date}/{prod_id}"
        if not os.path.exists(output_dir):
            print(f"--> Create directory {output_dir}")
            os.makedirs(output_dir, exist_ok=True)

        # make log dir
        log_dir = f"{output_dir}/log"
        if not os.path.exists(log_dir):
            print(f"--> Create directory {log_dir}")
            os.makedirs(log_dir, exist_ok=True)

        for gain in [0,1]:

            print(f"\n-->>>>> Process {channel[gain]} gain <<<<<")
            # define charge file names
            output_file = f"{output_dir}/filter_scan_{channel[gain]}_{date}.{sub_run:04d}.h5"
            log_file = f"{output_dir}/log/filter_scan_{channel[gain]}_{date}.{sub_run:04d}.log"

            print(f"\n--> Output file {output_file}")

            if os.path.exists(output_file):
                if query_yes_no(">>> Output file exists already. Do you want to remove it?"):
                    os.remove(output_file)
                else:
                    print(f"\n--> Stop")
                    exit(1)

            print(f"\n--> Log file {log_file}")

            #
            # produce filter scan fit file
            #

            cmd = f"lstchain_fit_filter_scan " \
                  f"--config={config_file} --input_dir={input_dir} --output_path={output_file} "\
                  f"--sub_run={sub_run} --gain_channel={gain} "\
                  f" >  {log_file} 2>&1"

            print("\n--> RUNNING...")
            if os.system(cmd):
                raise Exception(f"Error in command execution: {cmd}")



        print("\n--> END")

    except Exception as e:
        print(f"\n >>> Exception: {e}")


if __name__ == '__main__':
    main()
