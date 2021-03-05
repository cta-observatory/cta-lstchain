#!/usr/bin/env python
"""

Script to check the contents of LST DL1 files and associated muon ring files
To run it, type e.g.:
python lstchain_check_dl1.py
--input-file dl1_LST-1.1.Run01881.0000.h5

or, for a whole run:
python lstchain_check_dl1.py
--input-file "dl1_LST-1.1.Run01881.*.h5"

It produces as output a datacheck_dl1_*.h5 file and a datacheck_dl1_*.pdf
file containing data check information. If the input file is a single dl1
file, then the output file names contain the run and subrun index (otherwise,
only the run index)

The script can also be run over one file of type datacheck_dl1_*.h5, and then
only the plotting part is executed.

The muons_*.fits files which are produced together with the DL1 event files
must be available in the same directory as the input files (of whatever
type).

"""
from warnings import simplefilter
import argparse
import glob
import logging
import os
# I had enough of those annoying future warnings, hence:
simplefilter(action='ignore', category=FutureWarning)
from lstchain.datachecks import check_dl1, plot_datacheck


parser = argparse.ArgumentParser(formatter_class=argparse.
                                 ArgumentDefaultsHelpFormatter)

required = parser.add_argument_group('required arguments')
optional = parser.add_argument_group('optional arguments')

# Required arguments
# input file(s). Wildcards can be used, but inside quotes e.g. "dl1*.h5"
required.add_argument('--input-file', type=str, required=True,
                      help='Path to DL1 data file(s) (containing pixel-wise '
                           'charge information and image parameters) OR to '
                           'datacheck_dl1_*.h5 files (only plotting part is '
                           'executed in that case)'
                      )

# Optional arguments
# path for output files
optional.add_argument('--output-dir', default='.', type=str,
                      help='Directory where the output files will be written'
                      )

# path for muons .fits files. If not given, it is assumed that the files are
# in the same directory of the input files (either of the dl1 type
# or of the datacheck_dl1 type)
optional.add_argument('--muons-dir', default=None, type=str,
                      help='Directory where the muon .fits files are located '
                      )
# maximum number of processes to be run in parallel
# This refers to the processes explicitly spawned by check_dl1, not to what
# e.g. numpy may do on its own!
optional.add_argument('--max-cores', default=4, type=int,
                      help='Maximum number of processes spawned'
                      )
optional.add_argument('--omit-pdf', action='store_true',
                      help='Do NOT create the data check pdf file'
                      )
optional.add_argument('--batch', '-b', action='store_true',
                      help='Run the script without plotting output'
                      )

args, unknown = parser.parse_known_args()


def main():

    logging.basicConfig(level=logging.INFO, format='%(message)s')
    logger = logging.getLogger(__name__)

    if len(unknown) > 0:
        ukn = ''
        for s in unknown:
            ukn += s+' '
        logger.error('Unknown options: '+ukn)
        exit(-1)

    logger.info('input files: {}'.format(args.input_file))
    logger.info('output directory: {}'.format(args.output_dir))

    filenames = glob.glob(args.input_file)
    if len(filenames) == 0:
        logger.error('Input files not found!')
        exit(-1)

    # order input files by name, i.e. by run index (assuming they are in the
    # same directory):
    filenames.sort()

    # if input files are existing dl1 datacheck .h5 files, just create the
    # output pdf with the check plots (since nothing else can be done with
    # that input, the create_pdf argument is ignored in that case:
    if os.path.basename(filenames[0]).startswith("datacheck_dl1"):
        plot_datacheck(filenames, args.output_dir, args.batch, args.muons_dir)
        return

    # otherwise, do the full analysis to produce the dl1_datacheck h5 file
    # and the associated pdf:
    check_dl1(filenames, args.output_dir, args.max_cores, not args.omit_pdf, args.batch)


if __name__ == '__main__':
    main()
