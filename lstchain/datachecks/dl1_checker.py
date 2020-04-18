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
    dl1datacheck_flatfield = DL1DataCheckContainer()
    dl1datacheck_cosmics = DL1DataCheckContainer()

    # define criteria for detecting flatfield events, since as of 20200418
    # there is no reliable event tagging for those. We require a minimum
    # fraction of pixels with a charge above a sufficiently large value:
    ff_min_pixel_fraction = 0.8
    ff_charge_threshold = 50.

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
                num_bright_pixels = np.sum(image_table.col('image') >
                                        ff_charge_threshold, axis=1)
                flatfield_mask = num_bright_pixels > ff_min_pixel_fraction *\
                                 image_table.col('image').shape[1]
                cosmics_mask = ~(pedestal_mask | flatfield_mask)

                print('   pedestals:', np.sum(pedestal_mask),
                      'flatfield:', np.sum(flatfield_mask),
                      'cosmics:', np.sum(cosmics_mask))

                # fill quantities which depend on event-wise (not
                # pixel-wise) parameters:
                dl1datacheck_pedestals.fill_event_wise_info(subrun_index,
                                                            pedestals)
                dl1datacheck_cosmics.fill_event_wise_info(subrun_index,
                                                          cosmics)

                # now fill pixel-wise information:
                dl1datacheck_pedestals.fill_pixel_wise_info(image_table,
                                                            pedestal_mask)
                dl1datacheck_flatfield.fill_pixel_wise_info(image_table,
                                                            flatfield_mask)
                dl1datacheck_cosmics.fill_pixel_wise_info(image_table,
                                                          cosmics_mask)

                writer.write("dl1datacheck/pedestals", dl1datacheck_pedestals)
                writer.write("dl1datacheck/flatfield", dl1datacheck_flatfield)
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

    # aspect ratio of pdf pages:
    pagesize = [12., 7.5]

    pdf_filename = filename.replace('.h5', '.pdf')

    cam_description_table = \
        Table.read(filename, path='instrument/telescope/camera/LSTCam')
    geom = CameraGeometry.from_table(cam_description_table)
    engineering_geom = geom.transform_to(EngineeringCameraFrame())

    with PdfPages(pdf_filename) as pdf, tables.open_file(filename) as file:
        # first plot some results for interleaved pedestals:
        table_pedestals = file.root.dl1datacheck.pedestals
        table_flatfield = file.root.dl1datacheck.flatfield
        table_cosmics = file.root.dl1datacheck.cosmics

        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=pagesize)
        axes[0, 0].plot(table_cosmics.col('subrun_index'),
                        table_cosmics.col('num_events'))
        axes[0, 0].plot(table_pedestals.col('subrun_index'),
                        table_pedestals.col('num_events'))
        axes[0, 0].set_yscale('log')
        pdf.savefig()

        plot_mean_and_stddev(table_pedestals, engineering_geom,
                             'Pedestal', pagesize)
        pdf.savefig()

        plot_mean_and_stddev(table_flatfield, engineering_geom,
                             'Flat-field', pagesize)
        pdf.savefig()

        # for cosmics we plot the pixel rates above a few thresholds
        # We asume here that 5 such thresholds are present in the dl1datacheck
        # file
        fig, axes = plt.subplots(nrows=2, ncols=3, figsize=pagesize,
                                 sharey='row')
        fig.tight_layout(pad = 3.0, h_pad=3.0, w_pad=2.0)
        # find the thresholds (in pe) for which the event numbers are stored:
        colnames = [str for str in table_cosmics.colnames
                    if str.find('num_pulses_above') == 0]
        threshold = [int(str[str.find('above_')+6:str.find('_pe')])
                     for str in colnames]
        # sum (for all subruns) the number of events above the different
        # thresholds:
        sum_events = [np.sum(table_cosmics.col(colname), axis=0)
                      for colname in colnames]
        for i, colname in enumerate(colnames):
            zscale = 'log' if threshold[i] < 200 else 'lin'
            cam = CameraDisplay(engineering_geom, sum_events[i],
                                ax=axes.flatten()[i], norm=zscale,
                                title='Rate of >'+str(threshold[i])+
                                      ' p.e. pulses')
            cam.add_colorbar(ax=axes.flatten()[i])
            cam.show()
        axes[1, 2].axis('off')
        pdf.savefig()

        fig, axes = plt.subplots(nrows=2, ncols=3, figsize=pagesize)
        fig.tight_layout(pad = 3.0, h_pad=3.0, w_pad=2.0)
        axes[0, 0].set_xscale('log')
        axes[0, 0].set_yscale('log')
        for x, y in zip(threshold, sum_events):
            axes[0][0].plot(x*np.ones(len(y)), y, 'o')
        pdf.savefig()


def plot_mean_and_stddev(table, camgeom, label, pagesize):
    # calculate pixel-wise charge mean and standard deviation for the
    # whole run:
    mean = np.sum(np.multiply(table.col('charge_mean'),
                              table.col('num_events')[:,None]),
                  axis=0) / np.sum(table.col('num_events'))
    stddev = np.sqrt(np.sum(np.multiply(table.col('charge_stddev') ** 2,
                                        table.col('num_events')[:, None]),
                            axis=0) / np.sum(table.col('num_events')))

    # plot mean and std dev of pedestal charge, as camera display,
    # vs. pixel id, and as a histogram:
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=pagesize)
    fig.tight_layout(pad=3.0, h_pad=3.0, w_pad=2.0)
    cam = CameraDisplay(camgeom, mean, ax=axes[0, 0],
                        norm='log', title=label+' mean charge (p.e.)')
    cam.add_colorbar(ax=axes[0, 0])
    cam.show()
    axes[0, 1].plot(camgeom.pix_id, mean)
    axes[0, 1].set_xlabel('Pixel id')
    axes[0, 1].set_ylabel(label+' mean charge (p.e.)')
    axes[0, 2].set_yscale('log')
    axes[0, 2].hist(mean, bins=200)
    axes[0, 2].set_xlabel(label+' mean charge (p.e.)')
    axes[0, 2].set_ylabel('Pixels')
    # now the standard deviation:
    cam = CameraDisplay(camgeom, stddev, ax=axes[1, 0],
                        norm='log', title=label+' charge std dev (p.e.)')
    cam.add_colorbar(ax=axes[1, 0])
    cam.show()
    axes[1, 1].plot(camgeom.pix_id, stddev)
    axes[1, 1].set_xlabel('Pixel id')
    axes[1, 1].set_ylabel(label+' charge std dev (p.e.)')
    axes[1, 2].set_yscale('log')
    axes[1, 2].hist(stddev, bins=200)
    axes[1, 2].set_xlabel(label+' charge std dev (p.e.)')
    axes[1, 2].set_ylabel('Pixels')



class DL1DataCheckContainer(Container):
    """
    Container to store outcome of the DL1 data check
    """
    subrun_index = Field(-1, 'Subrun index')
    num_events = Field(-1, 'Total number of events')

    # pixel-wise quantities:
    charge_mean = Field(-1, 'Mean of pixel charge')
    charge_stddev = Field(-1, 'Standard deviation of pixel charge')
    # keep number of events above a few thresholds, like a low-res histogram
    # of pulse charges (2 points per decade in charge in p.e.):
    num_pulses_above_0010_pe = Field(None, 'Number of >10 p.e. pulses',
                                   unit=1./u.s)
    num_pulses_above_0030_pe = Field(None, 'Number of >30 p.e. pulses',
                                   unit=1./u.s)
    num_pulses_above_0100_pe = Field(None, 'Number of >100 p.e. pulses',
                                   unit=1./u.s)
    num_pulses_above_0300_pe = Field(None, 'Number of >300 p.e. pulses',
                                   unit=1./u.s)
    num_pulses_above_1000_pe = Field(None, 'Number of >1000 p.e. pulses',
                                   unit=1./u.s)
    # there must be a nicer way of doing the above...

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
        self.charge_mean = charge.mean(axis=0)
        self.charge_stddev = charge.std(axis=0)
        # count, for each pixel, the number of entries with charge>x pe:
        self.num_pulses_above_0010_pe = np.sum(charge > 10, axis=0)
        self.num_pulses_above_0030_pe = np.sum(charge > 30, axis=0)
        self.num_pulses_above_0100_pe = np.sum(charge > 100, axis=0)
        self.num_pulses_above_0300_pe = np.sum(charge > 300, axis=0)
        self.num_pulses_above_1000_pe = np.sum(charge > 1000, axis=0)
