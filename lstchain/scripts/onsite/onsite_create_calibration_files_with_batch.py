#!/usr//bin/env python

"""

 Onsite script for recontruct a run list from a filter scan

 --> onsite_reconstruct_filter_scan -r xxx yyy zzz

"""

import argparse
import os
from pathlib import Path
import lstchain
import subprocess
import shutil

# parse arguments
parser = argparse.ArgumentParser(description='Reconstruct filter scan, this must be run after the night calibration scripts',
                                 formatter_class = argparse.ArgumentDefaultsHelpFormatter)
required = parser.add_argument_group('required arguments')
optional = parser.add_argument_group('optional arguments')

required.add_argument('-r', '--run_list', help="Run number if the flat-field data",
                      type=int, nargs="+")

version,subversion=lstchain.__version__.rsplit('.post',1)
optional.add_argument('-f', '--filters_list', help="Filter list (same order than run list)",
                      type=int, nargs="+")

optional.add_argument('-v', '--prod_version',
                      help="Version of the production",
                      default=f"v{version}")
optional.add_argument('-s', '--statistics',
                      help="Number of events for the flat-field and pedestal statistics",
                      type=int,
                      default=10000)
optional.add_argument('-p', '--pedestal_run',
                      help="Pedestal run to be used. If None, it looks for the pedestal run of the date of the FF data.",
                      type=int)
optional.add_argument('--time_run',
                      help="Run for the time calibration. If None, search the last time run before or equal the first filter scan run",
                      type=int)
optional.add_argument('-b','--base_dir', help="Root dir for the output directory tree",type=str, default='/fefs/aswg/data/real')
optional.add_argument('--sub_run_list',
                      help="sub-run list to be processed.",
                      type=int,
                      nargs="+",default=[0])
optional.add_argument('--sys_date',
                      help="Date of systematics corrections (format YYYYMMDD). \n"
                           "If '0', no corrections are applied. Default: automatically search the best date \n")


default_config=os.path.join(os.path.dirname(__file__), "../../data/onsite_camera_calibration_param.json")
optional.add_argument('--config', help="Config file", default=default_config)


args = parser.parse_args()
run_list = args.run_list
filters_list = args.filters_list
ped_run = args.pedestal_run
prod_id = args.prod_version
stat_events = args.statistics
base_dir = args.base_dir
time_run = args.time_run

sub_run_list = args.sub_run_list
config_file = args.config
sys_date = args.sys_date
calib_dir=f"{base_dir}/monitoring/PixelCalibration"
max_events = 1000000

def main():

    if shutil.which('srun') is None:
        print(">>> This script needs a slurm batch system. Stop")
        return

    print(f"\n--> Start reconstruct runs {run_list} and sub-runs {sub_run_list}")

    # verify config file
    if not os.path.exists(config_file):
        raise IOError(f"Config file {config_file} does not exists. \n")

    print(f"\n--> Config file {config_file}")

    # for old runs or if the data-base is not available
    # it is possible to give the filter list
    if len(filter_list) > 0 and len(filter_list) != len(run_list):
            raise ValueError("Filter list length must be equal to run list length. Verify \n")

    # loops over runs and sub_runs and send jobs
    for i, run in enumerate(run_list):
        if filters_list is not None:
            filters = filters_list[i]

        for sub_run in sub_run_list:
            print(f"\n--> Runs {run} and sub-run {sub_run}")
            try:
                # verify input file
                file_list = sorted(Path(f"{base_dir}/R0").rglob(f'*{run}.{sub_run:04d}*'))
                if len(file_list) == 0:
                    raise IOError(f"Run {run} not found\n")
                else:
                    input_file = file_list[0]

                print(f"--> Input file: {input_file}")

                # find date
                input_dir, name = os.path.split(os.path.abspath(input_file))
                path, date = input_dir.rsplit('/', 1)

                # verify output dir
                output_dir = f"{calib_dir}/calibration/{date}/{prod_id}"
                if not os.path.exists(output_dir):
                    print(f"--> Create directory {output_dir}")
                    os.makedirs(output_dir, exist_ok=True)

                # verify log dir
                log_dir = f"{calib_dir}/calibration/{date}/{prod_id}/log"
                if not os.path.exists(log_dir):
                    print(f"--> Create directory {log_dir}\n")
                    os.makedirs(log_dir, exist_ok=True)

                # job file
                job_file = f"{log_dir}/{run}_{sub_run}.job"

                with open(job_file, "w") as fh:
                    fh.writelines("#!/bin/bash\n")
                    fh.writelines("#SBATCH --job-name=%s.job\n" % run)
                    fh.writelines("#SBATCH --output=log/%s_%d.out\n" % (run,sub_run))
                    fh.writelines("#SBATCH --error=log/%s_%s.err\n" % (run,sub_run))
                    fh.writelines("#SBATCH -A dpps\n")
                    fh.writelines("#SBATCH -p long\n")
                    fh.writelines("#SBATCH --array=0-0\n")
                    fh.writelines("#SBATCH --cpus-per-task=1\n")
                    fh.writelines("#SBATCH --mem-per-cpu=10G\n")
                    fh.writelines("#SBATCH -D %s \n" % output_dir)

                    fh.writelines(
                        f"srun onsite_create_calibration_file -r {run} "
                        f"-p {ped_run} -v {prod_id} --sub_run {sub_run} "
                        f"-b {base_dir} --filters={filters} --sys_date={sys_date} --config {config_file} --time_run "
                        f"{time_run}\n")

                subprocess.run(["sbatch",job_file])

            except Exception as e:
                print(f"\n >>> Exception: {e}. Run skipped")
                continue


if __name__ == '__main__':
    main()
