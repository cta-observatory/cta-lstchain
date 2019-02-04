import pandas as pd
import argparse
import os

parser = argparse.ArgumentParser(description="Merge all HDF5 files resulting from parallel reconstructions \
 present in a directory.")


parser.add_argument('--source-dir', '-d', type=str,
                    dest='srcdir',
                    help='path to the source directory of files',
                    )

parser.add_argument('--outfile', '-o', action='store', type=str,
                    dest='outfile',
                    help='Path of the resulting merged file',
                    default='merge.h5')

args = parser.parse_args()


if __name__ == '__main__':

    file_list = [args.srcdir + '/' + f for f in os.listdir(args.srcdir) if f.endswith('.h5')]

    dataframes = {}

    for file in file_list:
        ff = pd.HDFStore(file_list[0])
        keys = ff.keys()
        ff.close()
        for k in keys:
            if k in dataframes:
                dataframes[k] = pd.concat([dataframes[k], pd.read_hdf(file, key=k)])
            else:
                dataframes[k] = pd.read_hdf(file, key=k)

    for k, df in dataframes.items():
        df.to_hdf(args.outfile, key=k)

