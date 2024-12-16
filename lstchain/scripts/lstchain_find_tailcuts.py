#!/usr/bin/env python

"""
This script uses DL1 files to determine tailcuts which are adequate for the
bulk of the pixels in a given run. It does so simply based on the median (for
the whole camera) of the average pixel charge for pedestal events.

For reasons of stability & simplicity of analysis, we cannot decide the
cleaning levels on a subrun-by-subrun basis. We select values which are ok
for the whole run.

The script also returns the suggested NSB adjustment needed in the "dark-sky" MC
to match the data.

The script will process a maximum of 10 subruns of those provided as input,
distributed uniformly through the run - just for speed reasons: the result
would hardly change if all subruns are used.

lstchain_find_tailcuts -f "/.../dl1_LST-1.Run12469.????.h5"

The program creates in the output directory a .json file containing the
cleaning configuration.

"""

import argparse
import logging
from pathlib import Path
import sys

from lstchain.io.config import get_standard_config, dump_config
from lstchain.image.cleaning import find_tailcuts

parser = argparse.ArgumentParser(description="Tailcut finder")

parser.add_argument('-d', '--input-dir', dest='input_dir',
                    type=Path, default='./',
                    help='Input DL1 directory')
parser.add_argument('-r', '--run', dest='run_number',
                    type=int, help='Run number')
parser.add_argument('-o', '--output-dir', dest='output_dir',
                    type=Path, default='./',
                    help='Path to the output directory (default: %(default)s)')
parser.add_argument('--log', dest='log_file',
                    type=str, default=None,
                    help='Log file name')

log = logging.getLogger(__name__)

def main():
    args = parser.parse_args()
    log.setLevel(logging.INFO)
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    log_file = args.log_file
    if log_file is not None:
        handler = logging.FileHandler(log_file, mode='w')
        logging.getLogger().addHandler(handler)

    output_dir = args.output_dir.absolute()
    output_dir.mkdir(exist_ok=True, parents=True)
    input_dir = args.input_dir.absolute()

    run_id = args.run_number
    additional_nsb_rate, newconfig = find_tailcuts(input_dir, run_id)

    json_filename = Path(output_dir, f'dl1ab_Run{run_id:05d}.json')
    dump_config({'tailcuts_clean_with_pedestal_threshold': newconfig,
                 'dynamic_cleaning': get_standard_config()['dynamic_cleaning']},
                json_filename, overwrite=True)
    log.info(json_filename)
    log.info(f'Additional NSB rate (over dark MC): {additional_nsb_rate:.3f} '
             f'p.e./s')

    log.info('lstchain_find_tailcuts finished successfully!')

    return additional_nsb_rate
