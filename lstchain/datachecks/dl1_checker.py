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

    dl1datacheck_pedestals = DL1DataCheckContainer()
    dl1datacheck_cosmics = DL1DataCheckContainer()

    # obtain run number, and first part of file name, from first file:
    # NOTE: this assumes the string RunXXXXX.YYYY
    filename = filenames[0]
    run_number = int(filename[filename.find('Run')+3:][:5])
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

                # in order to read in the images we have to use tables,
                # because pandas is not compatible with vector columns
                image_table = file.root.dl1.event.telescope.image.LST_LSTCam

                # fill dummy event times with NaNs in case they do not exist
                # (like in MC):
                if 'dragon_time' not in parameters.keys():
                    dummy_times = np.empty(len(parameters['event_id']))
                    dummy_times[:] = np.nan
                    parameters['dragon_time'] = dummy_times

                # create subsets of the parameters dataframes:
                # (is this too memory consuming?)
                pedestals = \
                    parameters.loc[parameters['ucts_trigger_type'] == 32]
                cosmics = parameters.loc[parameters['ucts_trigger_type'] != 32]

                # create masks for the images table:
                pedestal_mask = image_table.col('ucts_trigger_type') == 32
                cosmics_mask = ~pedestal_mask

                # fill quantities which depend on event-wise (not
                # pixel-wise) parameters:
                dl1datacheck_pedestals.fill_event_wise_info(subrun_index,
                                                            pedestals)
                dl1datacheck_cosmics.fill_event_wise_info(subrun_index,
                                                          cosmics)

                # now fill pixel-wise information:
                dl1datacheck_pedestals.fill_pixel_wise_info(image_table,
                                                            pedestal_mask)
                dl1datacheck_cosmics.fill_pixel_wise_info(image_table,
                                                          cosmics_mask)

                writer.write("dl1datacheck/pedestals", dl1datacheck_pedestals)
                writer.write("dl1datacheck/cosmics", dl1datacheck_cosmics)

                dl1datacheck_pedestals.reset()
                dl1datacheck_cosmics.reset()

    # we assume that camera geom is the same in all files, & write the last one:
    geom.to_table().write(out_filename,
                          path=f'/instrument/telescope/camera/LSTCam',
                          append=True, serialize_meta=True)

    plot_datacheck(out_filename)


def plot_datacheck(filename=''):
    """

    Parameters
    ----------
    filename: .h5 file produced by the method check_dl1

    Returns
    -------
    None

    """

    pdf_filename = filename.replace('.h5', '.pdf')

    cam_description_table = \
        Table.read(filename, path='instrument/telescope/camera/LSTCam')
    geom = CameraGeometry.from_table(cam_description_table)
    engineering_geom = geom.transform_to(EngineeringCameraFrame())

    with PdfPages(pdf_filename) as pdf, tables.open_file(filename) as file:

        # first plot some results for interleved pedestals:
        table = file.root.dl1datacheck.pedestals

        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=[12., 9.])
        axes[0, 0].plot(table.col('subrun_index'), table.col('num_events'))
        pdf.savefig()

        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=[12., 9.])
        cam = CameraDisplay(engineering_geom,
                            np.sum(table.col('num_pulses_above_10_pe'), axis=0),
                            ax=axes[0, 0], norm='log', title='Rate of >10 '
                                                             'p.e. pulses')
        cam.add_colorbar(ax=axes[0, 0])
        cam.show()
        pdf.savefig()

        # now results for the cosmic events:
        table = file.root.dl1datacheck.cosmics

        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=[12., 9.])
        axes[0, 0].plot(table.col('subrun_index'), table.col('num_events'))
        pdf.savefig()

        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=[12., 9.])
        cam = CameraDisplay(engineering_geom,
                            np.sum(table.col('num_pulses_above_10_pe'), axis=0),
                            ax=axes[0, 0], norm='log', title='Rate of >10 '
                                                             'p.e. pulses')
        cam.add_colorbar(ax=axes[0, 0])
        cam.show()
        pdf.savefig()


class DL1DataCheckContainer(Container):
    """
    Container to store outcome of the DL1 data check
    """
    subrun_index = Field(-1, 'Subrun index')
    num_events = Field(-1, 'Total number of events')
    num_pulses_above_10_pe = Field(None, 'Number of >10 p.e. pulses',
                                   unit=1./u.s)

    def fill_event_wise_info(self, subrun_index, table):
        """
        Fills the container fields that depend on event-wise DL1 info

        Parameters
        ----------
        subrun_index
        table: DL1 parameters, event-wise pandas dataframe, "parameters" from
        DL1 files

        Returns
        -------
        None

        """
        self.subrun_index = subrun_index
        self.num_events = table['ucts_trigger_type'].count()

    def fill_pixel_wise_info(self, table, mask):
        """
        Fills the quantities that are calculated pixel-wise

        Parameters
        ----------
        table: DL1 parameters, event-wise python table "image" from DL1 files
        mask: indicates rows that have to be used for filling this container

        Returns
        -------
        None

        """
        charge = table.col('image')[mask]
        # count, for each pixel, the number of entries with charge>10pe:
        self.num_pulses_above_10_pe = np.sum(charge > 10., axis=0)
