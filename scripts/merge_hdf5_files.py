import pandas as pd
import argparse
import os
import h5py

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

args = parser.parse_args()


def get_dataset_keys(filename):
    """
    Return a list of all dataset keys in a HDF5 file

    Parameters
    ----------
    filename: str - path to the HDF5 file

    Returns
    -------
    list of keys
    """
    dataset_keys = []
    def walk(name, obj):
        if type(obj) == h5py._hl.dataset.Dataset:
            dataset_keys.append(name)

    with h5py.File(filename,'r') as file:
        file.visititems(walk)

    return dataset_keys


if __name__ == '__main__':

    file_list = [args.srcdir + '/' + f for f in os.listdir(args.srcdir) if f.endswith('.h5')]

    dataframes = {}

    for file in file_list:
        keys = get_dataset_keys(file)
        for k in keys:
            if k in dataframes:
                dataframes[k] = pd.concat([dataframes[k], pd.read_hdf(file, key=k)])
            else:
                dataframes[k] = pd.read_hdf(file, key=k)

    for k, df in dataframes.items():
        df.to_hdf(args.outfile, key=k)

