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
from glob import glob
from astropy.table import Table, vstack


parser = argparse.ArgumentParser(description='Merge fits files')

# Required arguments
parser.add_argument('--input-dir', '-d', type=str,
                    dest='srcdir',
                    help='path to the source directory of files',
                    )

# Optional arguments
parser.add_argument('--output-file', '-o', action='store', type=str,
                    dest='outfile',
                    help='Path of the resulting merged file',
                    default='merge.fits')

parser.add_argument('--run-number', '-r', action='store', type=int,
                    dest='run_number',
                    help='Merge files run-wise if a run number is passed, \
                          otherwise merge all files in the directory',
                    default=None)

parser.add_argument('--pattern', '-p',
                    help='Glob pattern to match files',
                    default='*.fits',
)

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


    if not file_list:
        raise IOError(f'No muon files in {args.srcdir} with the parameters requested')

    tab = Table.read('{}'.format(file_list[0]), format='fits')
    for i in range(1,len(file_list)):
        tab2 = Table.read('{}'.format(file_list[i]), format='fits')
        tab = vstack([tab, tab2])


    if os.path.exists(args.outfile):
        raise IOError(args.outfile + ' exists, exiting.')

    tab.write(args.outfile)


if __name__ == '__main__':
    main()


    
