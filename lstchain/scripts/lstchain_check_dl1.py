#!/usr/bin/env python
"""

Script to check the contents of LST DL1 files and associated muon ring files
To run it, type e.g.:
python lstchain_check_dl1.py
--input_file dl1_LST-1.1.Run01881.0000.h5

or, for a whole run:
python lstchain_check_dl1.py
--input_file "dl1_LST-1.1.Run01881.*.h5"

It produces as output a datacheck_dl1_*.h5 file and a datacheck_dl1_*.pdf
file containing data check information. If the input file is a single dl1
file, then the output file names contain the run and subrun index (otherwise,
only the run index)

The script can also be run over one file of type datacheck_dl1_*.h5, and then
only the plotting part ios executed.

The muons_*.fits files which are produced together with the DL1 event files
must be available in the same directory as the input files (of whatever
type).

"""
from warnings import simplefilter
import argparse
import glob
# I had enough of those annoying future warnings, hence:
simplefilter(action='ignore', category=FutureWarning)
from lstchain.datachecks import check_dl1, plot_datacheck


parser = argparse.ArgumentParser(formatter_class=argparse.
                                 ArgumentDefaultsHelpFormatter)

required = parser.add_argument_group('required arguments')
optional = parser.add_argument_group('optional arguments')

# Required arguments
# input file(s). Wildcards can be used, but inside quotes e.g. "dl1*.h5"
required.add_argument('--input_file', type=str, required=True,
                      help='Path to DL1 data file(s) (containing pixel-wise '
                           'charge information and image parameters)'
                      )

# Optional arguments
# path for output files
optional.add_argument('--output_path', default='.', type=str,
                      help='Path to the output files'
                      )
# maximum number of processes to be run in parallel
# This refers to the processes explicitly spawned by check_dl1, not to what
# e.g. numpy may do on its own!
optional.add_argument('--max_cores', default=4, type=int,
                      help='Maximum number of processes spawned'
                      )
optional.add_argument('--omit_pdf', action='store_true',
                      help='Do NOT create the data check pdf file'
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
    # output pdf with the check plots (since nothing else can be done with
    # that input, the create_pdf argument is ignored in that case:
    if len(filenames) == 1 and filenames[0].startswith("datacheck_dl1"):
        plot_datacheck(filenames[0], args.output_path)
        return

    # otherwise, do the full analysis to produce the dl1_datacheck h5 file
    # and the associated pdf:
    # order input files by name, i.e. by run index (assuming they are in the
    # same directory):
    filenames.sort()
    check_dl1(filenames, args.output_path, args.max_cores, not args.omit_pdf)


if __name__ == '__main__':
    main()
