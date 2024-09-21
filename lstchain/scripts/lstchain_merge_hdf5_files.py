#!/usr/bin/env python3

"""
Merge all HDF5 files resulting
from parallel reconstructions present in a
directory. Every dataset in the files must be
readable with pandas.

- Input: several hdf5 files.
- Output single hdf5 file.

Usage:

$> python lstchain_merge_hdf5_files.py
--input-dir ./

"""

import argparse
import os
from glob import glob

from lstchain.io import auto_merge_h5files
from lstchain.io import get_dataset_keys
from lstchain.io.io import (
    dl1_mon_tel_catB_cal_key,
    dl1_mon_tel_catB_ped_key,
    dl1_mon_tel_catB_flat_key
)

default_keys_to_copy = [
    dl1_mon_tel_catB_cal_key,
    dl1_mon_tel_catB_ped_key,
    dl1_mon_tel_catB_flat_key
]

parser = argparse.ArgumentParser(description='Merge HDF5 files')

# Required arguments
parser.add_argument(
    '-d', '--input-dir',
    help='path to the source directory of files',
    required=True,
)

# Optional arguments
parser.add_argument(
    '-o', '--output-file',
    help='Path of the resulting merged file',
    default='merge.h5'
)

parser.add_argument(
    '--no-image',
    action='store_true',
    help='Do not include images in output file',
)

parser.add_argument(
    '-r', '--run-number',
    type=int,
    help=(
        'Merge files run-wise if a run number is passed'
        'otherwise merge all files in the directory'
    )
)

parser.add_argument(
    '-p', '--pattern',
    default='*.h5',
    help='Glob pattern to match files',
)

parser.add_argument(
    '--no-progress',
    action='store_true',
    help='Do not display the progress bar during event processing'
)

parser.add_argument(
    '--skip-checks',
    action='store_true',
    help='Skip checks when merging files'
)

parser.add_argument(
    '--keys-to-copy',
    nargs="*", default=[],
    help='List of duplicated keys to be copied and not to be merged'
)

def main():
    args = parser.parse_args()

    if args.run_number:
        run = f'Run{args.run_number:05d}'
        file_list = sorted(filter(
            lambda f: run in f,
            glob(os.path.join(args.input_dir, args.pattern), recursive=True)
        ))
    else:
        file_list = sorted(glob(os.path.join(args.input_dir, args.pattern), recursive=True))

    if args.no_image:
        keys = get_dataset_keys(file_list[0])
        keys = {k for k in keys if 'image' not in k}
    else:
        keys = None

    keys_to_copy = default_keys_to_copy + args.keys_to_copy
    
    auto_merge_h5files(
        file_list,
        args.output_file,
        nodes_keys=keys,
        keys_to_copy=keys_to_copy,
        progress_bar=not args.no_progress,
        run_checks=not args.skip_checks
    )


if __name__ == '__main__':
    main()
