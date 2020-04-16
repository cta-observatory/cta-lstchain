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
from ctapipe.coordinates import EngineeringCameraFrame
from ctapipe.core import Container, Field
from ctapipe.instrument import CameraGeometry
from ctapipe.io import HDF5TableWriter
from ctapipe.visualization import CameraDisplay
from lstchain.io.io import dl1_params_lstcam_key
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
    # patch for DL1 files which contain the "stream tag" in the name: LST-1.1:
    out_filename = out_filename.replace('LST-1.1', 'LST-1')
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
                parameters = pd.read_hdf(filename, key=dl1_params_lstcam_key)
                telescope_description = \
                pd.read_hdf(filename, key='instrument/telescope/optics')

                # in order to read in the images we have to use tables,
                # because pandas is not compatible with vector columns
                group = file.root.dl1.event.telescope.image.LST_LSTCam

                # fill dummy event times with NaNs in case they do not exist
                # (like in MC):
                if 'dragon_time' not in parameters.keys():
                    dummy_times = np.empty(len(parameters['event_id']))
                    dummy_times[:] = np.nan
                    parameters['dragon_time'] = dummy_times

                # fill quantities which depend on event-wise (not
                # pixel-wise) parameters:
                dl1datacheck.fill_event_wise_info(subrun_index, parameters)
                dl1datacheck.fill_pixel_wise_info(group)
                writer.write("dl1datacheck", dl1datacheck)

                # loop over calibrated images
                #images = [x['image'] for x in group.iterrows()]
                #for full_image, event_id, dragon_time in \
                #zip(images, parameters['event_id'], parameters['dragon_time']):
                #    if event_id%10000 == 0:
                #        print(event_id)

                dl1datacheck.reset()

    # we assume that camera geom is the same in all files, & write the last one:
    geom.to_table().write(out_filename,
                          path=f'/instrument/telescope/camera/LSTCam',
                          append=True, serialize_meta=True)

    plot_datacheck(out_filename)

def plot_datacheck(filename=''):

    pdf_filename = filename.replace('.h5', '.pdf')

    cam_description_table = \
        Table.read(filename, path='instrument/telescope/camera/LSTCam')
    geom = CameraGeometry.from_table(cam_description_table)
    engineering_geom = geom.transform_to(EngineeringCameraFrame())

    with PdfPages(pdf_filename) as pdf, tables.open_file(filename) as file:

        table = file.root.dl1datacheck

        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=[12.,9.])
        axes[0,0].plot(table.col('subrun_index'), table.col('num_events'))
        pdf.savefig()

        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=[12.,9.])
        cam = CameraDisplay(engineering_geom,
                            np.sum(table.col('num_pulses_above_10_pe'), axis=0),
                            ax = axes[0,0])
        cam.add_colorbar(ax = axes[0,0])
        cam.show()
        pdf.savefig()

class DL1DataCheckContainer(Container):
    """
    Container to store outcome of the DL1 data check
    """
    subrun_index = Field(-1, 'Subrun index')
    num_events = Field(-1, 'Total number of events')
    num_shower_events = Field(-1, 'Number of shower events')
    num_pedestal_events = Field(-1, 'Number of pedestal events')
    num_flatfield_events = Field(-1, 'Number of flatfield events')
    num_pulses_above_10_pe = Field(None, 'Number of >10 p.e. pulses',
                                   unit=1./u.s)

    def fill_event_wise_info(self, subrun_index, table):
        """
        Fills the container fields that depend on event-wise DL1 info

        Parameters
        ----------
        table: DL1 parameters, event-wise python table "image" from DL1 files

        Returns
        -------
        None

        """
        self.subrun_index = subrun_index
        self.num_events = table['ucts_trigger_type'].count()
        self.num_pedestal_events = \
            np.sum(table['ucts_trigger_type'].between(32,32))

    def fill_pixel_wise_info(self, table):
        charge = table.col('image')
        # count, for each pixel, the number of entries with charge>10pe:
        self.num_pulses_above_10_pe  = np.sum(charge > 10., axis=0)
