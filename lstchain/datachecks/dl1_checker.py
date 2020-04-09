#!/usr/bin/env python
"""
Functions to check the contents of LST DL1 files and associated muon ring files
"""

__all__ = [
    'check_dl1'
    'plot_datacheck',
]

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import tables
from astropy import units as u
from astropy.table import Table
from ctapipe.instrument import CameraGeometry
from ctapipe.utils.fitshistogram import Histogram
from astropy.io import fits
from lstchain.io.io import dl1_params_lstcam_key
from lstchain.io.io import dl1_images_lstcam_key

def check_dl1(filenames, output_path):

    histo_ucts_trig_type = Histogram(nbins=100, ranges=(-2.5, 97.5),
                                     name='ucts_trig_type',
                                     axis_names='ucts_trig_type')

    run_number = -1
    filename_prefix = ''

    for filename in filenames:
        print('Opening file', filename)
        new_run_number = int(filename[filename.find('Run')+3:][:5])
        if run_number < 0:
            run_number = new_run_number
            filename_prefix = filename[:filename.find('Run')]
        if new_run_number != run_number:
            raise RuntimeError('Error: found different run numbers among input '
                               'files. Exiting')

        cam_description_table = \
            Table.read(filename, path='instrument/telescope/camera/LSTCam')
        geom = CameraGeometry.from_table(cam_description_table)

        with tables.open_file(filename) as file:

            # unfortunately pandas.read_hdf does not seem compatible with
            # 'with... as...' statements
            parameters = pd.read_hdf(filename, key = dl1_params_lstcam_key)
            telescope_description = \
                pd.read_hdf(filename, key='instrument/telescope/optics')

            group = file.root.dl1.event.telescope.image.LST_LSTCam
            images = [x['image'] for x in group.iterrows()]

            # fill dummy event times with NaNs in case they do not exist
            # (like in MC):
            if 'dragon_time' not in parameters.keys():
                dummy_times = np.empty(len(parameters['event_id']))
                dummy_times[:] = np.nan
                parameters['dragon_time'] = dummy_times

            histo_ucts_trig_type.fill(parameters['ucts_trigger_type'])

            for full_image, event_id, dragon_time in \
            zip(images, parameters['event_id'], parameters['dragon_time']):
                if event_id%10000 == 0:
                    print(event_id)

    fits_filename = output_path+'/datacheck_'+filename_prefix+\
                    'Run'+str(run_number)+'.fits'
    if os.path.exists(fits_filename):
        os.remove(fits_filename)
    fits_file = fits.open(fits_filename, mode='append')
    fits_file.append(histo_ucts_trig_type.to_fits())
    fits_file.close()

def plot_datacheck(filename=''):
    datacheck = fits.open(filename)
    histo_ucts_trig_type = Histogram.from_fits(datacheck[0])
    histo_ucts_trig_type.draw_1d()
    plt.show()
