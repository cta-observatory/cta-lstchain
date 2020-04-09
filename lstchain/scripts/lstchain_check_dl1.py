#!/usr/bin/env python
"""
Script to check the contents of LST DL1 files and associated muon ring file
To run it, type:
python lstchain_check_dl1.py
--input_file dl1_LST-1.1.Run01881.0000.fits.h5
"""
import warnings
# I had enough of those annoying future warnings, hence:
warnings.simplefilter(action='ignore', category=FutureWarning)

import argparse
import glob
from lstchain.datachecks import check_dl1

parser = argparse.ArgumentParser()

# Required arguments
parser.add_argument('--input_file', type=str, required=True,
                    help='Path to DL1 data file (containing pixel-wise '
                         'charge information and image parameters).'
                    )

parser.add_argument('--output_path', default='.', type=str,
                    help='Path to the output files'
                    )

args = parser.parse_args()

def main():

    filenames = glob.glob(args.input_file)
    filenames.sort()
    print('input files: {}'.format(filenames))
    print('output path: {}'.format(args.output_path))

    try:
        check_dl1(filenames, args.output_path)
    except Exception as str:
        print(str)
        exit(-1)

if __name__ == '__main__':
    main()
