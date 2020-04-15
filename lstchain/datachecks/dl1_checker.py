#!/usr/bin/env python
"""
Functions to check the contents of LST DL1 files and associated muon ring files
"""

__all__ = [
    'check_dl1'
    'plot_datacheck',
    'DL1DataCheckContainer',
]

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import tables
from astropy import units as u
from astropy.table import Table
from ctapipe.core import Container, Field
from ctapipe.instrument import CameraGeometry
from ctapipe.io import HDF5TableWriter
from lstchain.io.io import dl1_params_lstcam_key
from lstchain.io.io import dl1_images_lstcam_key
from matplotlib.backends.backend_pdf import PdfPages

def check_dl1(filenames, output_path):
    """

    Parameters
    ----------
    filenames: _sorted_ (by growing subrun index) list of input DL1 .h5 files
    output_path: directory where output will be written

    Returns
    -------
    None

    """

    dl1datacheck = DL1DataCheckContainer()

    # obtain run number, and first part of file name, from first file:
    # NOTE: this assumes the string RunXXXXX.YYYY
    filename = filenames[0]
    run_number = int(filename[filename.find('Run')+3:][:5])
    subrun_index = int(filename[filename.find('Run')+9:][:4])
    filename_prefix = filename[:filename.find('Run')]

    # define output filename (overwrite if already existing)
    out_filename = output_path + '/datacheck_' + filename_prefix + 'Run' + str(
            run_number) + '.h5'
    if os.path.exists(out_filename):
        os.remove(out_filename)

    with HDF5TableWriter(out_filename) as writer:

        for filename in filenames:
            print('Opening file', filename)
            new_run_number = int(filename[filename.find('Run')+3:][:5])
            if new_run_number != run_number:
                raise RuntimeError('Error: found different run numbers among '
                                   'input files. Exiting')
            subrun_index = int(filename[filename.find('Run') + 9:][:4])

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

                # fill quantities which depend on event-wise (not
                # pixel-wise) parameters:
                dl1datacheck.fill_event_wise_info(subrun_index, parameters)
                writer.write("dl1datacheck", dl1datacheck)

                for full_image, event_id, dragon_time in \
                zip(images, parameters['event_id'], parameters['dragon_time']):
                    if event_id%10000 == 0:
                        print(event_id)

                dl1datacheck.reset()

    plot_datacheck(out_filename)

def plot_datacheck(filename=''):

    pdf_filename = filename.replace('.h5', '.pdf')

    dl1datacheck = pd.read_hdf(filename, key='dl1datacheck')
    with PdfPages(pdf_filename) as pdf:
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=[12.,9.])
        dl1datacheck.plot('subrun_index', 'num_events', ax=axes[0,0])
        pdf.savefig()

class DL1DataCheckContainer(Container):
    """
    Container to store outcome of the DL1 data check
    """
    subrun_index = Field(-1, 'subrun_index')
    num_events = Field(-1, 'num_events')
    num_shower_events = Field(-1, 'num_shower_events')
    num_pedestal_events = Field(-1, 'num_pedestal_events')
    num_flatfield_events = Field(-1, 'num_flatfield_events')

    def fill_event_wise_info(self, subrun_index, table):
        """
        Fills the container fields that depend on event-wise DL1 info

        Parameters
        ----------
        table: DL1 parameters, event-wise pandas table DL1 files

        Returns
        -------
        None

        """
        self.subrun_index = subrun_index
        self.num_events = table['ucts_trigger_type'].count()
        self.num_pedestal_events = \
            np.sum(table['ucts_trigger_type'].between(32,32))
