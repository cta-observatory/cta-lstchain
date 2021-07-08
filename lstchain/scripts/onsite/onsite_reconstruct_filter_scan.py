#!/usr//bin/env python

"""

 Onsite script for recontruct a run list from a filter scan

 --> onsite_reconstruct_filter_scan -r xxx,yyy

"""

import argparse
import os
from pathlib import Path
import lstchain
# parse arguments
parser = argparse.ArgumentParser(description='Reconstruct filter scan, this must be run after the night calibration scripts',
                                 formatter_class = argparse.ArgumentDefaultsHelpFormatter)
required = parser.add_argument_group('required arguments')
optional = parser.add_argument_group('optional arguments')

required.add_argument('-r', '--run_list', help="Run number if the flat-field data",
                      type=int, nargs="+")

version,subversion=lstchain.__version__.rsplit('.post',1)
optional.add_argument('-v', '--version',
                      help="Version of the production",
                      default=f"v{version}")
optional.add_argument('-s', '--statistics',
                      help="Number of events for the flat-field and pedestal statistics",
                      type=int,
                      default=10000)
optional.add_argument('-p', '--pedestal_run',
                      help="Run number of the drs4 pedestal run",
                      type=int)
optional.add_argument('--time_run',
                      help="run time calibration",
                      type=int)

optional.add_argument('-b','--base_dir', help="Root dir for the output directory tree",type=str, default='/fefs/aswg/data/real')
optional.add_argument('--sub_run_list',
                      help="sub-run list to be processed.",
                      type=int,
                      nargs="+",default=[0])

#optional.add_argument('--min_ff', help="Min FF intensity cut in ADC.", type=float, default=4000)
#optional.add_argument('--max_ff', help="Max FF intensity cut in ADC.", type=float, default=12000)
default_config=os.path.join(os.path.dirname(__file__), "../../data/onsite_camera_calibration_param.json")
optional.add_argument('--config', help="Config file", default=default_config)
optional.add_argument('--tel_id', help="telescope id. Default = 1", type=int, default=1)


args = parser.parse_args()
run_list = args.run_list
ped_run = args.pedestal_run
prod_id = args.version
stat_events = args.statistics
base_dir = args.base_dir
time_run = args.time_run
tel_id = args.tel_id
#min_ff = args.min_ff
#max_ff = args.max_ff
sub_run_list = args.sub_run_list
config_file = args.config

calib_dir=f"{base_dir}/monitoring/CameraCalibration"
max_events = 1000000

def main():

    print(f"\n--> Start reconstruct runs {run_list} and sub-runs {sub_run_list}")

    # reference run = first run in list
    runs=sorted(run_list)
    first_run=runs[0]

    try:
        # verify config file
        if not os.path.exists(config_file):
            raise IOError(f"Config file {config_file} does not exists. \n")

        print(f"\n--> Config file {config_file}")

        # loops over runs and sub_runs and send jobs
        for run in run_list:
            for sub_run in sub_run_list:
                # verify input file
                file_list = sorted(Path(f"{base_dir}/R0").rglob(f'*{run}.{sub_run:04d}*'))
                if len(file_list) == 0:
                    raise IOError(f"Run {run} not found\n")
                else:
                    input_file = file_list[0]
                print(f"\n--------------")
                print(f"--> Input file: {input_file}")

                # find date
                input_dir, name = os.path.split(os.path.abspath(input_file))
                path, date = input_dir.rsplit('/', 1)

                # verify output dir
                output_dir = f"{calib_dir}/filter_scan/{date}/{prod_id}"
                if not os.path.exists(output_dir):
                    print(f"--> Create directory {output_dir}")
                    os.makedirs(output_dir, exist_ok=True)

                # verify log dir
                log_dir = f"{calib_dir}/filter_scan/{date}/{prod_id}/log"
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
                            if int(run_in_list) > first_run:
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

                # job file
                job_file = f"{log_dir}/{run}_{sub_run}.job"

                with open(job_file, "w") as fh:
                    fh.writelines("#!/bin/bash\n")
                    fh.writelines("#SBATCH --job-name=%s.job\n" % run)
                    fh.writelines("#SBATCH --output=log/%s.out\n" % run)
                    fh.writelines("#SBATCH --error=log/%s.err\n" % run)
                    fh.writelines("#SBATCH -A dpps\n")
                    fh.writelines("#SBATCH -p long\n")
                    fh.writelines("#SBATCH --array=0-0\n")
                    fh.writelines("#SBATCH --cpus-per-task=1\n")
                    fh.writelines("#SBATCH --mem-per-cpu=10G\n")
                    fh.writelines("#SBATCH -D %s \n" % output_dir)

                    fh.writelines(
                        f"srun onsite_create_dc_to_pe_file -r {run} "
                        f"-p {ped_run} -v {prod_id} --sub_run {sub_run} "
                        f"-b {base_dir} --config {config_file} --time_run "
                        f"{time_run}\n")

                #os.system("sbatch %s" % job_file)


        print("\n--> END")

    except Exception as e:
        print(f"\n >>> Exception: {e}")


if __name__ == '__main__':
    main()
