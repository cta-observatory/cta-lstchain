#!/fefs/aswg/workspace/jakub.jurysek/software/anaconda3/envs/lst-dev/bin/python3.7

import os
from matplotlib import pyplot as plt
import numpy as np
import astropy.units as u
from optparse import OptionParser
from argparse import ArgumentParser
import pandas as pd
import random

from lstchain.io.config import read_configuration_file
from lstchain.reco import r0_to_dl1
from lstchain import version


def merge_pedestals(files, drop_duplicates=True):

    i = 0
    for file in files:

        data = pd.read_pickle(file)

        if i == 0:
            data_all = data.copy()
        else:
            data_all = data_all.append(data.copy(), ignore_index=True)
        i += 1
    data_all = data_all.sort_values(by=['time'], ignore_index=True)

    if drop_duplicates:
        data_all = data_all.drop_duplicates(subset = ["time"])

    return data_all

def extend_pedestals(r1_waveforms, px_mean, target_samples=40):

    mean_ped = px_mean
    mean_ped = mean_ped[:, :, np.newaxis]
    r1_waveforms = np.concatenate((mean_ped, mean_ped, r1_waveforms, mean_ped, mean_ped), axis=-1)

    return r1_waveforms


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument("-d", "--display", dest="display", action="store_true", help="Display written data", default=False)
    parser.add_argument("-r", "--randomize", dest="randomize", action="store_true",
                        help="Randomize input pedestals. For each MC event, random pedestal event based on real data is constructed.", default=False)
    parser.add_argument("-w", "--window", dest="window",
                        help="Time window for random pedestal construction", type=float, default=30.)
    parser.add_argument("-o", "--outpath", dest="outpath", help="Output path", default="./", type=str)
    parser.add_argument("-m", "--mc", dest="mc", help="input mc file", nargs='*')
    parser.add_argument("-p", "--pedestal", dest="pedestal", help="pedestal file", nargs='*')
    parser.add_argument("-f", "--pedestal_factor", dest="pedestal_factor", help="pedestal normalization factor", type=float, default=1.)
    parser.add_argument('--config', '-c', action='store', type=str,
                    dest='config_file',
                    default=None
                    )

    #retrieve args
    options = parser.parse_args()
    display         = options.display
    out_path        = options.outpath
    mc_files              = options.mc
    pedestal_files  = options.pedestal
    pedestal_shift  = options.pedestal_factor
    randomize       = options.randomize
    window          = options.window
    config = read_configuration_file(options.config_file)

    print('lstchain version used:', version.get_version())

    if len(pedestal_files) > 0:
        merge = True
        print('Pedestal files', pedestal_files)

        # Pedetal events from data
        pedestals = merge_pedestals(pedestal_files, drop_duplicates=True)
        print(pedestals.shape)

        r1_waveforms = np.stack(pedestals['data'].to_numpy())
        on_pixel_masks = np.stack(pedestals['mask_flash'].to_numpy())
        event_id = np.stack(pedestals['event_id'].to_numpy())
        times = np.stack(pedestals['time'].to_numpy())

        print(min(times), max(times), np.mean(times))

        # selection of short time interval around a single subrun for MC-data comparison
        # LST-1.1.Run02969.0060.fits.fz, mean time 1605921630.582161
        """
        time0 = 1605921630.582161
        time_min = time0 - 60
        time_max = time0 + 60
        mask_subrun = (times > time_min) & (times < time_max)
        r1_waveforms = r1_waveforms[mask_subrun]
        on_pixel_masks = on_pixel_masks[mask_subrun]
        times = times[mask_subrun]
        """

        r1_waveforms = r1_waveforms * pedestal_shift

        #r1_waveforms_old = r1_waveforms
        #print(event_id[0], px_mean[0, 2], np.mean(r1_waveforms_old[0,2,:]), r1_waveforms_old[0,2,:])
        #r1_waveforms = extend_pedestals(r1_waveforms, px_mean, target_samples=40)

        #for i in range(5):
        #    print((np.mean(r1_waveforms[i, 2, :])-np.mean(r1_waveforms_old[i, 2, :])) / np.mean(r1_waveforms_old[i, 2, :]), (np.var(r1_waveforms[i, 2, :]) - np.var(r1_waveforms_old[i, 2, :])) / np.var(r1_waveforms_old[i, 2, :]) )

        """
        plt.figure()
        plt.hist(times-min(times), bins=1000)
        plt.savefig('./times.png')
        """

        # Merging with MC files, DL1 calibration
        print('MC files:', mc_files)
        pedestals_used_total = 0
        pedestals_left = r1_waveforms.shape[0]

        for mc in mc_files:
            if pedestals_left > 0:

                outfile = 'dl1_' + mc.split('/')[-1].split('.')[0] + '.h5'
                print('Running lstchain on file:', mc)

                if not randomize:
                    pedestals_used = r0_to_dl1.r0_to_dl1_pedestals(
                        mc,
                        output_filename=os.path.join(out_path, outfile),
                        custom_config=config,
                        r1_pedestals=r1_waveforms[pedestals_used_total:],
                        on_pixel_masks=on_pixel_masks[pedestals_used_total:],
                        correction_factor=pedestal_shift
                    )
                    pedestals_used_total += pedestals_used
                    pedestals_left = r1_waveforms.shape[0] - pedestals_used_total
                    print('DONE. Events processed:', pedestals_used)
                    print('Total pedestals used:', pedestals_used_total, 'Pedestal events left:', pedestals_left)
                else:
                    print('Using randomized pedestal events...')
                    pedestals_used = r0_to_dl1.r0_to_dl1_pedestals(
                        mc,
                        output_filename=os.path.join(out_path, outfile),
                        custom_config=config,
                        r1_pedestals=r1_waveforms,
                        on_pixel_masks=on_pixel_masks,
                        correction_factor=pedestal_shift,
                        random_pedestals=randomize,
                        times=times,
                        window=window
                    )
                    print('DONE. Events processed:', pedestals_used)
                print('DL1 file saved as:', os.path.join(out_path, outfile))
            else:
                print('No pedestals left! Following file cannot be processed:', mc)
