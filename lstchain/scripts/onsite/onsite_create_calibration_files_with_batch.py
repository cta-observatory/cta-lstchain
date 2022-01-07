#!/usr//bin/env python

"""

 Onsite script for recontruct a run list from a filter scan

 --> onsite_create_calibration_files_with_batch -r xxx yyy zzz

"""

import argparse
import os
from pathlib import Path
import lstchain
import subprocess
import shutil
import sys
from datetime import datetime

# parse arguments
parser = argparse.ArgumentParser(description='Reconstruct filter scan, this must be run after the night calibration scripts',
                                 formatter_class = argparse.ArgumentDefaultsHelpFormatter)
required = parser.add_argument_group('required arguments')
optional = parser.add_argument_group('optional arguments')

required.add_argument('-r', '--run_list', help="Run number if the flat-field data",
                      type=int, nargs="+")


optional.add_argument('-f', '--filters_list', help="Filter list (same order as run list)",
                      type=int, nargs="+")
version=lstchain.__version__
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
optional.add_argument('--r0-dir', help="Root dir for the input r0 tree. By default, <base_dir>/R0 will be used", type=Path)

optional.add_argument('--sub_run_list',
                      help="sub-run list to be processed.",
                      type=int,
                      nargs="+",default=[0])
optional.add_argument('--sys_date',
                      help="Date of systematic corrections (format YYYYMMDD). \n"
                           "Default: automatically search the best date \n")
optional.add_argument('--no_sys_correction',
                      help="Systematic corrections are not applied. \n",
                      action='store_true',
                      default=False)
optional.add_argument('--output_base_name', help="Output file base name (change only for debugging)", default="calibration")
optional.add_argument('-y', '--yes', action="store_true", help='Do not ask interactively for permissions, assume true')
optional.add_argument('--no_pro_symlink', action="store_true", help='Do not update the pro dir symbolic link, assume true')


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
no_sys_correction = args.no_sys_correction
yes = args.yes
pro_symlink = not args.no_pro_symlink

output_base_name = args.output_base_name

calib_dir=f"{base_dir}/monitoring/PixelCalibration/LevelA"

def main():

    if shutil.which('srun') is None:
        sys.exit(">>> This script needs a slurm batch system. Stop")

    print(f"\n--> Start reconstruct runs {run_list} and sub-runs {sub_run_list}")

    # verify config file
    if not os.path.exists(config_file):
        sys.exit(f"Config file {config_file} does not exists. \n")

    print(f"\n--> Config file {config_file}")

    # for old runs or if the data-base is not available
    # it is possible to give the filter list
    if filters_list is not None and len(filters_list) != len(run_list):
        sys.exit("Filter list length must be equal to run list length. Verify \n")

    # loops over runs and sub_runs and send jobs
    filters = None
    for i, run in enumerate(run_list):
        if filters_list is not None:
            filters = filters_list[i]

        for sub_run in sub_run_list:
            print(f"\n--> Runs {run} and sub-run {sub_run}")

            # verify input file
            r0_dir = args.r0_dir or Path(args.base_dir) / 'R0'
            file_list = sorted(r0_dir.rglob(f'*{run}.{sub_run:04d}*'))
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
            now = datetime.now().replace(microsecond=0).isoformat(sep='T')
            job_file = f"{log_dir}/run_{run}_subrun_{sub_run}_date_{now}.job"

            with open(job_file, "w") as fh:
                fh.write("#!/bin/bash\n")
                fh.write("#SBATCH --job-name=%s.job\n" % run)
                fh.write("#SBATCH --output=log/run_%s_subrun_%s_date_%s.out\n" % (run,sub_run,now))
                fh.write("#SBATCH --error=log/run_%s_subrun_%s_date_%s.err\n" % (run,sub_run,now))
                fh.write("#SBATCH -p short\n")
                fh.write("#SBATCH --cpus-per-task=1\n")
                fh.write("#SBATCH --mem-per-cpu=10G\n")
                fh.write("#SBATCH -D %s \n" % output_dir)

                cmd = [
                    "srun",
                    "onsite_create_calibration_file",
                    f"-r {run}",
                    f"-v {prod_id}",
                    f"--sub_run={sub_run}",
                    f"-b {base_dir}",
                    f"-s {stat_events}",
                    f"--output_base_name={output_base_name}",
                    f"--config={config_file}",
                ]

                if ped_run is not None:
                    cmd.append(f"--pedestal_run={ped_run}")

                if time_run is not None:
                    cmd.append(f"--time_run={time_run}")

                if filters is not None:
                    cmd.append(f"--filters={filters}")

                if sys_date is not None:
                    cmd.append(f"--sys_date={sys_date}")

                if yes:
                    cmd.append("--yes")

                if no_sys_correction:
                    cmd.append("--no_sys_correction")

                # join command together with newline, line continuation and indentation
                fh.write(" \\\n  ".join(cmd))
                fh.write('\n')

            subprocess.run(["sbatch", job_file], check=True)


if __name__ == '__main__':
    main()
