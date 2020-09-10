"""
A sseparate script to index the HDU tables and Observation tables of the DL3 files
in a particular directory.
The number of DL3 fits files to be indexed from the directory can be selected as
per a sorted order.

There is a small issue of combining the Table data as BinTableHDUs to the
compressed index fits files as it prepends a PrimaryHDU while writing it.
For now, the tables are directly written to the compressed index fits file.

"""


import os

import argparse
import numpy as np
from lstchain.hdu import create_obs_hdu_index


parser = argparse.ArgumentParser(description="Creating DL3 index files")

# Required arguments
parser.add_argument('--input-dl3-dir', '-d', type=str,
                    dest='input_dl3_dir',
                    help='path to DL3 files',
                    default=None, required=True
                    )
parser.add_argument('--num-files', '-n', type=int,
                    dest='number_of_files',
                    help='Number of files in sorted order',
                    default=1, required=True
                    )

args = parser.parse_args()

def main():

    file_list = []
    Run = []
    for file in os.listdir(args.input_dl3_dir):
        # Assuming the nomenclature is 'dl3_LST-1_{#Run number}_merged.fits'
        if file.startswith('dl3_'):
            file_list.append(file)
            Run.append(int(file.split('_')[2]))
    file_list = (np.sort(file_list))
    Run = (np.sort(Run))

    list_obs_id = Run[:args.number_of_files]

    create_obs_hdu_index(list_obs_id, args.input_dl3_dir)

if __name__ == '__main__':
    main()
