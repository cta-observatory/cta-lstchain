#!/usr/bin/env python
"""
Script to check the contents of LST DL1 files and associated muon ring file
To run it, type e.g.:
python lstchain_check_dl1.py
--input_file dl1_LST-1.1.Run01881.0000.h5
"""
import warnings
# I had enough of those annoying future warnings, hence:
warnings.simplefilter(action='ignore', category=FutureWarning)

import argparse
import glob
from lstchain.datachecks import check_dl1, plot_datacheck

parser = argparse.ArgumentParser()

# Required arguments
# input file(s). Wildcards can be used, but inside quotes e.g. "dl1*.h5"
parser.add_argument('--input_file', type=str, required=True,
                    help='Path to DL1 data file(s) (containing pixel-wise '
                         'charge information and image parameters).'
                    )

parser.add_argument('--output_path', default='.', type=str,
                    help='Path to the output files'
                    )

args = parser.parse_args()

def main():

    print('input files: {}'.format(args.input_file))
    print('output path: {}'.format(args.output_path))

    filenames = glob.glob(args.input_file)
    if len(filenames) == 0:
        print('Input files not found!')
        exit(-1)

    # if input file is an existing dl1 datacheck .h5 file, just create the
    # output pdf with the check plots:
    if len(filenames) == 1 and filenames[0].startswith("datacheck_dl1"):
        plot_datacheck(filenames[0], args.output_path)
        return

    # otherwise, do the full analysis to produce the dl1_datacheck h5 file
    # and the associated pdf:
    # order input files by name, i.e. by run index:
    filenames.sort()
    try:
        check_dl1(filenames, args.output_path)
    except Exception as str:
        print(str)
        exit(-1)

if __name__ == '__main__':
    main()
