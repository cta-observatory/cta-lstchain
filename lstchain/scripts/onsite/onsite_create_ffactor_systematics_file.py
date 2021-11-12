#!/usr//bin/env python

"""

 Onsite script to create a F-factor systematic correction file by fitting an intensity scan


"""

import argparse
import os
import subprocess
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

required.add_argument('-d', '--date', help="Date of the scan (YYYYMMDD)", required=True)

# config file is mandatory because it contains the list of input runs
required.add_argument('-c','--config', help="Config file (json format) with the list of runs", required=True)

version=lstchain.__version__.rsplit('.post',1)[0]
optional.add_argument('-v', '--prod_version', help="Version of the production",
                      default=f"v{version}")
optional.add_argument('-b','--base_dir', help="Root dir for the output directory tree", type=str, default='/fefs/aswg/data/real')
optional.add_argument('--sub_run', help="sub-run to be processed.", type=int, default=0)
optional.add_argument('--input_prefix', help="Prefix of the input file names", default="calibration")
optional.add_argument('-y', '--yes', action="store_true", help='Do not ask interactively for permissions, assume true')

args = parser.parse_args()
date = args.date
prod_id = args.prod_version
base_dir = args.base_dir
sub_run = args.sub_run
config_file = args.config
prefix = args.input_prefix
yes = args.yes

def main():

    try:
        # verify config file
        if not os.path.exists(config_file):
            raise IOError(f"Config file {config_file} does not exists. \n")

        print(f"\n--> Config file {config_file}")

        # verify input dir
        input_dir=f"{base_dir}/monitoring/PixelCalibration/calibration/{date}/{prod_id}"
        if not os.path.exists(input_dir):
            raise IOError(f"Input directory {input_dir} not found\n")

        print(f"\n--> Input directory {input_dir}")

        # verify output dir
        output_dir = f"{base_dir}/monitoring/PixelCalibration/ffactor_systematics/{date}/{prod_id}"
        if not os.path.exists(output_dir):
            print(f"--> Create directory {output_dir}")
            os.makedirs(output_dir, exist_ok=True)

        # make log dir
        log_dir = f"{output_dir}/log"
        if not os.path.exists(log_dir):
            print(f"--> Create directory {log_dir}")
            os.makedirs(log_dir, exist_ok=True)

        # define output file names
        output_file = f"{output_dir}/{prefix}_scan_fit_{date}.{sub_run:04d}.h5"
        log_file = f"{output_dir}/log/{prefix}_scan_fit_{date}.{sub_run:04d}.log"
        plot_file = f"{output_dir}/log/{prefix}_scan_fit_{date}.{sub_run:04d}.pdf"

        if os.path.exists(output_file):
            remove = False

            if not yes and os.getenv('SLURM_JOB_ID') is None:
                remove = query_yes_no(">>> Output file exists already. Do you want to remove it?")

            if yes or remove:
                os.remove(output_file)
                os.remove(log_file)
            else:
                print("\n--> Output file exists already. Stop")
                exit(1)

        print(f"\n--> Plot file {plot_file}")
        print(f"\n--> Log file {log_file}")

        #
        # produce intensity scan fit file
        #

        cmd = f"lstchain_fit_intensity_scan " \
              f"--config={config_file} --input_dir={input_dir} --output_path={output_file} "\
              f"--plot_path={plot_file} --sub_run={sub_run} " \
              f"--input_prefix={prefix} --log-file={log_file} --log-file-level=DEBUG "

        print("\n--> RUNNING...")
        subprocess.run(cmd.split())


        print("\n--> END")

    except Exception as e:
        print(f"\n >>> Exception: {e}")


if __name__ == '__main__':
    main()
