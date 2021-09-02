#!/fefs/aswg/workspace/jakub.jurysek/software/anaconda3/envs/lst-dev/bin/python3.7

import os
import numpy as np
import astropy.units as u
from optparse import OptionParser
from argparse import ArgumentParser
import pandas as pd


def merge_pedestals(files):

    i = 0
    for file in files:

        print(file)
        data = pd.read_pickle(file)

        if i == 0:
            data_all = data.copy()
        else:
            data_all = data_all.append(data.copy(), ignore_index=True)
        i += 1
    data_all = data_all.sort_values(by=['time'], ignore_index=True)
    return data_all


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument("-o", "--outfile", dest="outfile", help="Output file", default="./merged.pkl", type=str)
    parser.add_argument("-p", "--pedestal_files", dest="pedestal_files", help="input pedestal files", nargs='*')

    #retrieve args
    options = parser.parse_args()
    out_file        = options.outfile
    pedestal_files  = options.pedestal_files

    # Pedetal events from data
    pedestals = merge_pedestals(pedestal_files)

    print(pedestals.shape)



    # saving outputs
    pd_data = pd.DataFrame(pedestals, columns=["obs_id", "event_id", "time", "azimuth", "altitude", "mask_flash", "px_mean", "px_var", "data"])
    pd_data.to_pickle(out_file)
    print("All pedestal events saved to:", out_file)
