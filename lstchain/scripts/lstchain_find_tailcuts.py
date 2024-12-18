#!/usr/bin/env python

"""
This script uses DL1 files to determine tailcuts which are adequate for the
bulk of the pixels in a given run. It does so simply based on the median (for
the whole camera) of the median pixel charge for pedestal events.

For reasons of stability & simplicity of analysis, we cannot decide the
cleaning levels on a subrun-by-subrun basis. We select values which are ok
for the whole run.

The script writes out the cleaning settings to a json file,
e.g. dl1ab_Run13181.json
It also returns the suggested NSB adjustment needed in the "dark-sky" MC
to match the data, in units of p.e./ns.

lstchain_find_tailcuts -d "/..../DL1/YYYYMMDD/v0.10/tailcut84 -r 13181 --log
out.log "

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

    if not input_dir.exists():
        logging.error('Input directory does not exist')
        sys.exit(1)

    if not input_dir.is_dir():
        logging.error('Input directory is not a directory!')
        sys.exit(1)

    run_id = args.run_number
    median_qped, additional_nsb_rate, newconfig = find_tailcuts(input_dir,
                                                                run_id)

    json_filename = Path(output_dir, f'dl1ab_Run{run_id:05d}.json')
    dump_config({'tailcuts_clean_with_pedestal_threshold': newconfig,
                 'dynamic_cleaning': get_standard_config()['dynamic_cleaning']},
                json_filename, overwrite=True)
    log.info(f'\nMedian pedestal charge: {median_qped:.3f} p.e.')
    log.info('\nCleaning settings:')
    log.info(newconfig)
    log.info('\nWritten to:')
    log.info(json_filename)
    log.info(f'\nAdditional NSB rate (over dark MC): {additional_nsb_rate:.4f} '
             f'p.e./ns')

    log.info('lstchain_find_tailcuts finished successfully!')
