#!/usr/bin/env python3

"""
Merge all fits files resulting 
from parallel muon reconstructions present in a 
directory. 
- Input: several fits files.
- Output single fits file.
Usage: 
$> python lstchain_merge_muon_files.py
--input-dir ./
--run-number 1881
"""

import argparse
import os
from distutils.util import strtobool
# import tables
from lstchain.io import get_dataset_keys
from glob import glob

parser = argparse.ArgumentParser(description='Merge fits files')

# Required arguments
parser.add_argument('--input-dir', '-d', type=str,
                    dest='srcdir',
                    help='path to the source directory of files',
                    )
parser.add_argument('--run-number', '-r', action='store', type=int,
                    dest='run_number',
                    help='Merge files run-wise if a run number is passed, \
                          otherwise merge all files in the directory',
                    default=None)

args = parser.parse_args()


def main():
    if args.run_number:
        run = f'Run{args.run_number:05d}'
        file_list = sorted(filter(
            lambda f: run in f,
            glob(os.path.join(args.srcdir, args.pattern))
        ))
    else:
        file_list = sorted(glob(os.path.join(args.srcdir, args.pattern)))




if __name__ == '__main__':
    main()
