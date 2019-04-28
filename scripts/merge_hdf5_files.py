import argparse
import os
from lstchain.io import merge_events_tables

parser = argparse.ArgumentParser(description="Merge all HDF5 files resulting from parallel reconstructions \
 present in a directory.\n"
                                 "By default, every dataset in the files will be merged, in this case, "
                                             "all datasets must be readable with pandas.\n"
                                 "If you want to merge a single dataset, use the --table-name option.\n"
                                 ">>> python merged_hdf5_files.py source-directory -o merged_files.h5 -t events/LSTCam",
                                 formatter_class=argparse.RawTextHelpFormatter,
                                 )


parser.add_argument('--source-dir', '-s', action='store', type=str,
                    dest='srcdir',
                    help='path to the source directory of files',
                    )

parser.add_argument('--outfile', '-o', action='store', type=str,
                    dest='outfile',
                    help='Path of the resulting merged file',
                    default='merge.h5',
                    )

parser.add_argument('--events-table', '-e', action='store', type=str,
                    dest='table_name',
                    help='Name of a single table to be considered for the merging',
                    default='events',
                    )

args = parser.parse_args()



if __name__ == '__main__':

    file_list = [args.srcdir + '/' + f for f in os.listdir(args.srcdir) if f.endswith('.h5')]

    print("Merging events in table {} from files:\n {}".format(args.table_name, file_list))

    merge_events_tables(file_list, args.outfile, args.table_name)