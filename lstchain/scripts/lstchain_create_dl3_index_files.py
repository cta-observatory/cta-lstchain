#!/usr/bin/env python3

"""
A separate script to index the HDU tables and Observation tables of the DL3 files
in a particular directory.
The number of DL3 fits files to be indexed from the directory can be selected as
per a sorted order.
Usage:
$> python lstchain_create_dl3_index_files.py
--input-dl3-dir ./DL3/
--num-files n
"""

import os
from pathlib import Path
import logging
import sys

import argparse
import numpy as np
from lstchain.hdu import create_obs_hdu_index

log = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description="Creating DL3 index files")

# Required arguments
parser.add_argument('--input-dl3-dir', '-d', type=Path,
                    dest='input_dl3_dir',
                    help='path to DL3 files',
                    default=None, required=True
                    )
parser.add_argument('--num-files', '-n', type=int,
                    dest='number_of_files',
                    help='Number of files in sorted order',
                    default=1, required=True
                    )

args = parser.parse_args()

def main():

    if not args.input_dl3_dir.is_dir():
        log.error('Input dir does not exist')
        sys.exit(1)

    log.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    logging.getLogger().addHandler(handler)

    filename_list = []
    for file in os.listdir(args.input_dl3_dir):
        # Assuming the nomenclature is 'dl3_LST-1_{#Run number}_merged.fits'
        if file.startswith('dl3_'):
            filename_list.append(file)

    if len(filename_list) < args.number_of_files:
        log.error('Number of files given exceeds the number of files')
        sys.exit(1)

    filename_list = np.sort(filename_list)[:args.number_of_files]

    create_obs_hdu_index(filename_list, args.input_dl3_dir)

if __name__ == '__main__':
    main()
