#!/usr/bin/env python

"""

 Onsite script to process (in batch) the interleaved events of several runs 
 and to create the Cat-B calibration files

 --> onsite_create_catB_calibration_files_with_batch -r xxx yyy zzz

"""

import argparse
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import lstchain
from lstchain.onsite import (
    DEFAULT_BASE_PATH,
    CAT_B_PIXEL_DIR,
    find_interleaved_subruns,
    find_r0_subrun,
    DEFAULT_CONFIG_CAT_B_CALIB,
)

# parse arguments
parser = argparse.ArgumentParser(
    description='Reconstruct filter scan, this must be run after the night calibration scripts',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
required = parser.add_argument_group('required arguments')
optional = parser.add_argument_group('optional arguments')

required.add_argument('-r', '--run_list', help="Run numbers of intereleaved data",
                      type=int, nargs="+")
optional.add_argument('-f', '--filters_list', help="Filter list (same order as run list)",
                      type=int, nargs="+")

optional.add_argument('-s', '--statistics',
                      help="Number of events for the flat-field and pedestal statistics",
                      type=int,
                      default=2500)
optional.add_argument('-b', '--base_dir', help="Root dir for the output directory tree", type=Path,
                      default=DEFAULT_BASE_PATH)
optional.add_argument('--interleaved-dir', help="Root dir for the input interleaved files. By default, <base_dir>/DL1/date/version/interleaved will be used",
                       type=Path)
optional.add_argument('--r0-dir', help="Root dir for the input r0 tree. By default, <base_dir>/R0 will be used",
                      type=Path)

optional.add_argument('--n_subruns', help="Number of subruns to be processed",
                      type=int)

optional.add_argument('--sys_date',
                      help="Date of systematic corrections (format YYYYMMDD). \n"
                           "Default: automatically search the best date \n")
optional.add_argument('--no_sys_correction',
                      help="Systematic corrections are not applied. \n",
                      action='store_true',
                      default=False)

optional.add_argument('-y', '--yes', action="store_true", help='Do not ask interactively for permissions, assume true')
optional.add_argument('--no_pro_symlink', action="store_true",
                      help='Do not update the pro dir symbolic link, assume true')

optional.add_argument('--config', help="Config file", default=DEFAULT_CONFIG_CAT_B_CALIB, type=Path)


optional.add_argument('--queue',
                      help="Slurm queue. Deafault: short ",
                      default="short")


def main():
    args, remaining_args = parser.parse_known_args()
    run_list = args.run_list
    n_subruns = args.n_subruns
    
    filters_list = args.filters_list
    
    prod_id = f"v{lstchain.__version__}"
    stat_events = args.statistics
    base_dir = args.base_dir

    config_file = args.config
    sys_date = args.sys_date
    no_sys_correction = args.no_sys_correction
    yes = args.yes
    queue = args.queue
    r0_dir = args.r0_dir or args.base_dir / 'R0'
    calib_dir = base_dir / CAT_B_PIXEL_DIR

    if shutil.which('srun') is None:
        sys.exit(">>> This script needs a slurm batch system. Stop")

    print(f"\n--> Start to reconstruct runs {run_list}")

    # verify config file
    if not config_file.exists():
        sys.exit(f"Config file {config_file} does not exists. \n")

    print(f"\n--> Config file {config_file}")

    # for old runs or if the data-base is not available
    # it is possible to give the filter list
    if filters_list is not None and len(filters_list) != len(run_list):
        sys.exit("Filter list length must be equal to run list length. Verify \n")

    # loops over runs and send jobs
    filters = None
    for i, run in enumerate(run_list):
        print(f"\n--> Run {run} ")
        if filters_list is not None:
            filters = filters_list[i]

        # look in R0 to find the date
        r0_list = find_r0_subrun(run,0,r0_dir)
        date = r0_list.parent.name

        # find input path
        ver = prod_id.rsplit(".")
        input_path = args.interleaved_dir or args.base_dir / 'DL1'/ f"{date}/{ver[0]}.{ver[1]}/interleaved" 

        input_files = find_interleaved_subruns(run, input_path)

        print(f"--> Found {len(input_files)} interleaved subruns in {input_path}")
        if n_subruns:
            print(f"--> Process {n_subruns} subruns")

        # verify output dir
        calib_dir = args.base_dir / CAT_B_PIXEL_DIR
        output_dir = calib_dir / "calibration" / date / prod_id
        if not output_dir.exists():
            print(f"--> Create directory {output_dir}")
            output_dir.mkdir(parents=True, exist_ok=True)

        # make log dir
        log_dir = output_dir / "log"
        if not log_dir.exists():
            print(f"--> Create directory {log_dir}")
            log_dir.mkdir(parents=True, exist_ok=True)
           
        # job file
        now = datetime.now().replace(microsecond=0).isoformat(sep='T')
        job_file = log_dir / f"run_{run}_date_{now}.job"

        with job_file.open(mode="w") as fh:
            fh.write("#!/bin/bash\n")
            fh.write("#SBATCH --job-name=%s.job\n" % run)
            fh.write("#SBATCH --output=log/run_%s_date_%s.out\n" % (run, now))
            fh.write("#SBATCH --error=log/run_%s_date_%s.err\n" % (run, now))
            fh.write("#SBATCH -p %s\n" % queue)
            fh.write("#SBATCH --cpus-per-task=1\n")
            fh.write("#SBATCH --mem-per-cpu=10G\n")
            fh.write("#SBATCH -D %s \n" % output_dir)

            cmd = [
                "srun",
                "onsite_create_cat_B_calibration_file",
                f"-r {run}",
                f"-v {prod_id}",
                f"--interleaved-dir {input_path}",
                f"--r0-dir {r0_dir}",
                f"-b {base_dir}",
                f"-s {stat_events}",
                f"--config={config_file}",
            ]


            if filters is not None:
                cmd.append(f"--filters={filters}")

            if sys_date is not None:
                cmd.append(f"--sys_date={sys_date}")

            if yes:
                cmd.append("--yes")

            if no_sys_correction:
                cmd.append("--no_sys_correction")
            
            if n_subruns:
                cmd.append(f"--n_subruns={n_subruns}")  
            
            if args.no_pro_symlink is True:
                cmd.append("--no_pro_symlink")

            cmd.extend(remaining_args)

            # join command together with newline, line continuation and indentation
            fh.write(" \\\n  ".join(cmd))
            fh.write('\n')

        subprocess.run(["sbatch", job_file], check=True)


if __name__ == '__main__':
    main()
