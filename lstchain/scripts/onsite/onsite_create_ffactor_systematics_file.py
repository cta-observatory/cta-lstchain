#!/usr//bin/env python
"""

 Onsite script to create a F-factor systematic correction file by fitting an intensity scan


"""
from pathlib import Path
import argparse
import os
import subprocess

import lstchain
from lstchain.io.data_management import query_yes_no
from lstchain.onsite import CAT_A_PIXEL_DIR, create_pro_symlink, DEFAULT_BASE_PATH

def none_or_str(value):
    if value == "None":
        return None
    return value


# parse arguments
parser = argparse.ArgumentParser(description='Create flat-field calibration files',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
required = parser.add_argument_group('required arguments')
optional = parser.add_argument_group('optional arguments')

required.add_argument('-d', '--date', help="Date of the scan (YYYYMMDD)", required=True)

# config file is mandatory because it contains the list of input runs
required.add_argument('-c', '--config', type=Path, help="Config file (json format) with the list of runs", required=True)

version = lstchain.__version__
optional.add_argument('-v', '--prod_version', help="Version of the production",
                      default=f"v{version}")
optional.add_argument('-b', '--base_dir', help="Root dir for the output directory tree", type=Path,
                      default=DEFAULT_BASE_PATH)
optional.add_argument('--sub_run', help="sub-run to be processed.", type=int, default=0)
optional.add_argument('--input_prefix', help="Prefix of the input file names", default="calibration")
optional.add_argument('-y', '--yes', action="store_true", help='Do not ask interactively for permissions, assume true')
optional.add_argument('--no_pro_symlink', action="store_true",
                      help='Do not update the pro dir symbolic link, assume true')



def main():
    args, remaining_args = parser.parse_known_args()
    date = args.date
    prod_id = args.prod_version
    base_dir = args.base_dir
    sub_run = args.sub_run
    config_file = args.config
    prefix = args.input_prefix
    yes = args.yes
    pro_symlink = not args.no_pro_symlink
    calib_dir = base_dir / CAT_A_PIXEL_DIR

    # verify config file
    if not config_file.exists():
        raise IOError(f"Config file {config_file} does not exists.")

    print(f"\n--> Config file {config_file}")

    # verify output dir
    output_dir = calib_dir / "ffactor_systematics" / date / prod_id
    if not output_dir.exists():
        print(f"--> Create directory {output_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)

    if pro_symlink:
        pro = "pro"
        create_pro_symlink(output_dir)
    else:
        pro = prod_id

    # verify input dir
    input_dir = calib_dir / "calibration" / date / pro
    if not input_dir.exists():
        raise IOError(f"Input directory {input_dir} not found")

    print(f"\n--> Input directory {input_dir}")

    # make log dir
    log_dir = output_dir / "log"
    if not log_dir.exists():
        print(f"--> Create directory {log_dir}")
        log_dir.mkdir(parents=True, exist_ok=True)

    # define output file names
    output_file = output_dir / f"{prefix}_scan_fit_{date}.{sub_run:04d}.h5"
    log_file = log_dir / f"{prefix}_scan_fit_{date}.{sub_run:04d}.log"
    plot_file = log_dir / f"{prefix}_scan_fit_{date}.{sub_run:04d}.pdf"

    if output_file.exists():
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

    cmd = [
        "lstchain_fit_intensity_scan",
        f"--config={config_file}",
        f"--input_dir={input_dir}",
        f"--output_path={output_file}",
        f"--plot_path={plot_file}",
        f"--sub_run={sub_run}",
        f"--input_prefix={prefix}",
        f"--log-file={log_file}",
        "--log-file-level=DEBUG",
        *remaining_args,
    ]

    print("\n--> RUNNING...")
    subprocess.run(cmd, check=True)
    print("\n--> END")


if __name__ == '__main__':
    main()
