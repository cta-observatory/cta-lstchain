import argparse
import os
from distutils.util import strtobool
# import tables
# from lstchain.io import get_dataset_keys
from lstchain.io import smart_merge_h5files, auto_merge_h5files

parser = argparse.ArgumentParser(description="Merge all HDF5 files resulting from parallel reconstructions \
 present in a directory. Every dataset in the files must be readable with pandas.")


parser.add_argument('--source-dir', '-d', type=str,
                    dest='srcdir',
                    help='path to the source directory of files',
                    )

parser.add_argument('--outfile', '-o', action='store', type=str,
                    dest='outfile',
                    help='Path of the resulting merged file',
                    default='merge.h5')

parser.add_argument('--smart', action='store', type=lambda x: bool(strtobool(x)),
                    dest='smart',
                    help='Boolean. True for smart merge, False for auto merge',
                    default=True)

args = parser.parse_args()



if __name__ == '__main__':
    file_list = [args.srcdir + '/' + f for f in os.listdir(args.srcdir) if f.endswith('.h5')]

    if args.smart:
        smart_merge_h5files(file_list, args.outfile)
    else:
        auto_merge_h5files(file_list, args.outfile)

