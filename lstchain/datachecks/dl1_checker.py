"""
Functions to check the contents of LST DL1 files and associated muon ring files
"""

__all__ = [
    'check_dl1',
    'process_dl1_file',
    'plot_datacheck',
    'plot_mean_and_stddev',
    'DL1DataCheckContainer',
    'DL1DataCheckHistogramBins',
]

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import tables
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table
from ctapipe.coordinates import CameraFrame, EngineeringCameraFrame
from ctapipe.core import Container, Field
from ctapipe.instrument import CameraGeometry
from ctapipe.io import HDF5TableWriter
from ctapipe.visualization import CameraDisplay
from lstchain.io.io import dl1_params_lstcam_key
from matplotlib.backends.backend_pdf import PdfPages
from multiprocessing import Pool
from sys import platform


def check_dl1(filenames, output_path, max_cores=4):
    """

    Parameters
    ----------
    filenames: _sorted_ (by growing subrun index) list of input DL1 .h5 files
    output_path: directory where output will be written
    max_cores: maximum number of processes that the function will spawn (each
    processing a different subrun)

    Returns
    -------
    None

    """

    # to allow the use of multi-processing in linux:
    if platform == 'linux':
        os.system("taskset -p 0xff %d" % os.getpid())

    # obtain run number, and first part of file name, from first file:
    # NOTE: this assumes the string RunXXXXX.YYYY
    filename = filenames[0]
    run_number = int(filename[filename.find('Run')+3:][:5])
    filename_prefix = filename[:filename.find('Run')]

    # define output filename (overwrite if already existing)
    out_filename = output_path + '/datacheck_' + filename_prefix + 'Run' + \
                   str(run_number) + '.h5'
    # patch for DL1 files which contain the "stream tag" in the name: LST-1.1:
    out_filename = out_filename.replace('LST-1.1', 'LST-1')
    if os.path.exists(out_filename):
        os.remove(out_filename)

    # TBD: Check here that all the run_numbers coincide!
    # new_run_number = int(filename[filename.find('Run') + 3:][:5])
    # if new_run_number != run_number:
    #     raise RuntimeError('Error: found different run numbers among '
    #                       'input files. Exiting')

    # the list dl1datacheck will contain one entry per subrun. Each entry is a
    # list of 3 containers of type DL1DataCheckContainer, one for pedestals,
    # one for flatfield events and one for cosmics

    # check that all files exist:
    for filename in filenames:
        if not os.path.exists(filename):
            raise FileNotFoundError

    # create container for the histograms' binnings, to be saved in the hdf5
    # output file:
    histogram_binning = DL1DataCheckHistogramBins()

    # create the dl1_datacheck containers (one per subrun) for the three
    # event types, and add them to the list dl1datacheck:
    with Pool(max_cores) as pool:
        func_args = [(filename, histogram_binning) for filename in filenames]
        dl1datacheck = pool.starmap(process_dl1_file, func_args)
    # NOTE: the above does not seem to improve execution time at least on Mac
    # OS X. Perhaps related to numpy "sharing" between the processes?

    # for now we process the files sequentially:
    # dl1datacheck = list([None]*len(filenames))
    # for i, filename in enumerate(filenames):
    #     dl1datacheck[i] = process_dl1_file(filename, histogram_binning)

    with HDF5TableWriter(out_filename) as writer:
        # write the containers (3 per subrun) to the dl1 data check output file:
        for dcheck in dl1datacheck:
            writer.write("dl1datacheck/pedestals", dcheck[0])
            writer.write("dl1datacheck/flatfield", dcheck[1])
            writer.write("dl1datacheck/cosmics", dcheck[2])
        # write also the histogram binnings:
        writer.write("dl1datacheck/histogram_binning", histogram_binning)

    # we assume that cam geom is the same in all files, & write the first one:
    cam_description_table = \
        Table.read(filename, path='instrument/telescope/camera/LSTCam')
    geom = CameraGeometry.from_table(cam_description_table)
    geom.to_table().write(out_filename,
                          path=f'/instrument/telescope/camera/LSTCam',
                          append=True, serialize_meta=True)

    # do the plots and save them to a pdf file:
    plot_datacheck(out_filename)

    return


# noinspection PyTypeChecker
def process_dl1_file(filename, bins):
    """

    Parameters
    ----------
    filename: input DL1 .h5 file to be checked
    bins: DL1DataCheckHistogramBins container indicating binning of histograms

    Returns
    -------
    dl1datacheck_pedestals, dl1datacheck_flatfield, dl1datacheck_cosmics
    Containers of type DL1DataCheckContainer, with info on the three types of
    events: interleaved pedestals, interleaved flatfield events, and cosmics

    """

    # define criteria for detecting flatfield events, since as of 20200418
    # there is no reliable event tagging for those. We require a minimum
    # fraction of pixels with a charge above a sufficiently large value:
    ff_min_pixel_charge_median = 40.
    ff_max_pixel_charge_stddev = 20.

    print('Opening file', filename)
    subrun_index = int(filename[filename.find('Run') + 9:][:4])

    dl1datacheck_pedestals = DL1DataCheckContainer()
    dl1datacheck_flatfield = DL1DataCheckContainer()
    dl1datacheck_cosmics = DL1DataCheckContainer()

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
        #  (is this too memory consuming?)
        # pedestals = \
        #    parameters.loc[parameters['ucts_trigger_type'] == 32]
        # cosmics = parameters.loc[parameters['ucts_trigger_type'] != 32]

        # create masks for the images table. For the time being, trigger
        # type tags are not reliable. We first identify flatfield events by
        # their looks, then use ucts_trigger_type to identify pedestals:
        image = image_table.col('image')
        flatfield_mask = ((np.median(image, axis=1) >
                           ff_min_pixel_charge_median) &
                          (np.std(image, axis=1) <
                           ff_max_pixel_charge_stddev))
        pedestal_mask = ~flatfield_mask & \
                        (image_table.col('ucts_trigger_type') == 32)
        cosmics_mask = ~(pedestal_mask | flatfield_mask)

        # Now create the masks for the parameters table, just the
        # same event_id's (i.e. we do not assume that the rows in the
        # images and parameters tables correspond one to one, though
        # this should be the case if all events are saved)
        ped_indices = image_table.col('event_id')[pedestal_mask]
        params_pedestal_mask = np.array(
                [(True if evtid in ped_indices else False) for evtid in
                 parameters['event_id']])
        ff_indices = image_table.col('event_id')[flatfield_mask]
        params_flatfield_mask = np.array(
                [(True if evtid in ff_indices else False) for evtid in
                 parameters['event_id']])
        params_cosmics_mask = ~(params_pedestal_mask | params_flatfield_mask)

        print('   pedestals:', np.sum(pedestal_mask), 'flatfield:',
              np.sum(flatfield_mask), 'cosmics:', np.sum(cosmics_mask))

        # fill quantities which depend on event-wise (not
        # pixel-wise) parameters:
        dl1datacheck_pedestals.fill_event_wise_info(subrun_index, parameters,
                                                    params_pedestal_mask, bins)
        dl1datacheck_flatfield.fill_event_wise_info(subrun_index, parameters,
                                                    params_flatfield_mask, bins)
        dl1datacheck_cosmics.fill_event_wise_info(subrun_index, parameters,
                                                  params_cosmics_mask, bins)

        # now fill pixel-wise information:
        dl1datacheck_pedestals.fill_pixel_wise_info(image_table,
                                                    pedestal_mask)
        dl1datacheck_flatfield.fill_pixel_wise_info(image_table,
                                                    flatfield_mask)
        dl1datacheck_cosmics.fill_pixel_wise_info(image_table,
                                                  cosmics_mask)

    return dl1datacheck_pedestals, dl1datacheck_flatfield, dl1datacheck_cosmics


def plot_datacheck(filename='', out_path=None):
    """

    Parameters
    ----------
    filename: .h5 file produced by the method check_dl1
    out_path: optional, if not given it will be the same of file filename
    Returns
    -------
    None

    """

    # aspect ratio of pdf pages:
    pagesize = [12., 7.5]

    pdf_filename = filename.replace('.h5', '.pdf')
    if out_path is not None:
        pdf_filename = out_path+'/'+pdf_filename[pdf_filename.rfind('/')+1:]

    cam_description_table = \
        Table.read(filename, path='instrument/telescope/camera/LSTCam')
    geom = CameraGeometry.from_table(cam_description_table)
    engineering_geom = geom.transform_to(EngineeringCameraFrame())

    with PdfPages(pdf_filename) as pdf, tables.open_file(filename) as file:

        # get the binning of the stored histograms:
        hist_binning = file.root.dl1datacheck.histogram_binning

        # get the tables for each type of events:
        table_pedestals = file.root.dl1datacheck.pedestals
        table_flatfield = file.root.dl1datacheck.flatfield
        table_cosmics = file.root.dl1datacheck.cosmics
        param_tables = [table_cosmics, table_flatfield, table_pedestals]

        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=pagesize)
        for table in param_tables:
            axes[0, 0].plot(table.col('subrun_index'), table.col('num_events'))
        axes[0, 0].set_yscale('log')
        pdf.savefig()

        plot_mean_and_stddev(table_pedestals, engineering_geom,
                             'Pedestal', pagesize)
        pdf.savefig()

        plot_mean_and_stddev(table_flatfield, engineering_geom,
                             'Flat-field', pagesize)
        pdf.savefig()

        # We now plot the pixel rates above a few thresholds.
        # Find the thresholds (in pe) for which the event numbers are stored:
        colnames = [name for name in table_cosmics.colnames
                    if name.find('num_pulses_above') == 0]
        threshold = [int(name[name.find('above_')+6:name.find('_pe')])
                     for name in colnames]

        for table in [table_pedestals, table_cosmics]:
            # We asume here that 5 such thresholds are present in the
            # dl1datacheck file
            fig, axes = plt.subplots(nrows=2, ncols=3, figsize=pagesize,
                                     sharey='row')
            fig.tight_layout(pad=3.0, h_pad=3.0, w_pad=2.0)

            # sum (for all subruns) the number of events above the different
            # thresholds:
            sum_events = [np.sum(table.col(colname), axis=0)
                          for colname in colnames]
            for i, colname in enumerate(colnames):
                zscale = 'log' if threshold[i] < 200 else 'lin'
                cam = CameraDisplay(engineering_geom, sum_events[i],
                                    ax=axes.flatten()[i], norm=zscale,
                                    title='Rate of >' + str(threshold[i]) +
                                          ' p.e. pulses')
                cam.add_colorbar(ax=axes.flatten()[i])
                cam.show()
            axes[1, 2].axis('off')
            pdf.savefig()

        fig, axes = plt.subplots(nrows=2, ncols=3, figsize=pagesize)
        fig.tight_layout(pad=3.0, h_pad=3.0, w_pad=2.0)
        axes[0, 0].set_xscale('log')
        axes[0, 0].set_yscale('log')
        for x, y in zip(threshold, sum_events):
            axes[0][0].plot(x*np.ones(len(y)), y, 'o')
        pdf.savefig()

        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=pagesize)
        bins = hist_binning.col('hist_intensity')[0]
        for table in param_tables:
            axes[0, 0].hist(bins[:-1], bins,
                            weights=np.sum(table.col('hist_intensity'),
                                           axis=0))
        axes[0, 0].set_xscale('log')
        axes[0, 0].set_yscale('log')
        pdf.savefig()

        fig, axes = plt.subplots(nrows=2, ncols=3, figsize=pagesize)
        fig.tight_layout(pad=3.0, h_pad=3.0, w_pad=2.0)
        bins = hist_binning.col('hist_cog')[0]
        x = np.array([xx for xx in bins[0][:-1] for __ in bins[1][:-1]])
        y = np.array([yy for __ in bins[0][:-1] for yy in bins[1][:-1]])

        hists = ['hist_cog', 'hist_cog_intensity_gt_200']
        for i, hist in enumerate(hists):
            contents = np.sum(table_cosmics.col(hist), axis=0).flatten()
            _, _, _, image = axes[i, 0].hist2d(x, y, bins=bins,
                                               weights=contents)
            plt.colorbar(image, ax=axes[i, 0])
            axes[i, 0].set_aspect('equal')
            _, _, _, image = axes[i, 1].hist2d(x, y, bins=bins,
                                               weights=contents,
                                               norm=colors.LogNorm())
            plt.colorbar(image, ax=axes[i, 1])
            axes[i, 1].set_aspect('equal')
            axes[i, 2].set_xscale('log')
            axes[i, 2].set_xlabel('fraction of all events')
            axes[i, 2].set_ylabel('number of bins')
            event_fraction = contents[contents > 0]/contents[contents > 0].sum()
            axes[i, 2].hist(event_fraction,
                            bins=np.logspace(np.log10(event_fraction.min()),
                                             np.log10(event_fraction.max()),
                                             101))

        pdf.savefig()


def plot_mean_and_stddev(table, camgeom, label, pagesize):
    # calculate pixel-wise charge mean and standard deviation for the
    # whole run:
    mean = np.sum(np.multiply(table.col('charge_mean'),
                              table.col('num_events')[:, None]),
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
    cam = CameraDisplay(camgeom, stddev, ax=axes[1, 0],
                        norm='log', title=label+' charge std dev (p.e.)')
    cam.add_colorbar(ax=axes[1, 0])
    # line below needed to get the top and bottom camera displays of equal size:
    axes[1, 0].set_xlim((axes[0, 0].get_xlim()))
    cam.show()

    axes[0, 1].plot(camgeom.pix_id, mean)
    axes[0, 1].set_xlabel('Pixel id')
    axes[0, 1].set_ylabel(label+' mean charge (p.e.)')
    axes[0, 2].set_yscale('log')
    axes[0, 2].hist(mean, bins=200)
    axes[0, 2].set_xlabel(label+' mean charge (p.e.)')
    axes[0, 2].set_ylabel('Pixels')
    # now the standard deviation:

    axes[1, 1].plot(camgeom.pix_id, stddev)
    axes[1, 1].set_xlabel('Pixel id')
    axes[1, 1].set_ylabel(label+' charge std dev (p.e.)')
    axes[1, 2].set_yscale('log')
    axes[1, 2].hist(stddev, bins=200)
    axes[1, 2].set_xlabel(label+' charge std dev (p.e.)')
    axes[1, 2].set_ylabel('Pixels')


# noinspection PyUnresolvedReferences
class DL1DataCheckContainer(Container):
    """
    Container to store outcome of the DL1 data check
    """

    subrun_index = Field(-1, 'Subrun index')
    num_events = Field(-1, 'Total number of events')
    hist_intensity = Field(None, 'Histogram of image intensity')
    hist_cog = Field(None, 'Histogram of image center of gravity')
    hist_cog_intensity_gt_200 = Field(None, 'Histogram of image center of '
                                            'gravity, intensity>200')

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

    # noinspection PyTypeChecker,PyArgumentList
    def fill_event_wise_info(self, subrun_index, table, mask,
                             histogram_binnings):
        """
        Fills the container fields that depend on event-wise DL1 info

        Parameters
        ----------
        subrun_index
        table: DL1 parameters, event-wise pandas dataframe, "parameters" from
        DL1 files
        mask: defines which events in table should be considered
        histogram_binnings: container of type DL1DataCheckHistogramBins which
        defines the binning of the various histograms

        Returns
        -------
        None

        """
        self.subrun_index = subrun_index
        self.num_events = table['ucts_trigger_type'][mask].count()

        intensity = table['intensity'][mask]
        counts, _, _ = plt.hist(intensity,
                                bins=histogram_binnings.hist_intensity)
        self.hist_intensity = counts

        # center of gravity histograms
        x = table['x'][mask]
        y = table['y'][mask]
        # Transform coordinates to engineering camera frame:
        orig = SkyCoord(x=x, y=y, unit=u.m, frame=CameraFrame())
        engi = orig.transform_to(EngineeringCameraFrame())
        counts, _, _, _ = plt.hist2d(engi.x, engi.y,
                                     bins=histogram_binnings.hist_cog)
        self.hist_cog = counts

        select = intensity > 200
        counts, _, _, _ = \
            plt.hist2d(engi.x[select], engi.y[select],
                       bins=histogram_binnings.hist_cog_intensity_gt_200)
        self.hist_cog_intensity_gt_200 = counts

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


class DL1DataCheckHistogramBins(Container):
    hist_cog = Field(np.array([np.linspace(-1.25, 1.25, 51),
                               np.linspace(-1.25, 1.25, 51)]),
                     'hist_cog binning')
    hist_cog_intensity_gt_200 = Field(np.array([np.linspace(-1.25, 1.25, 51),
                                                np.linspace(-1.25, 1.25, 51)]),
                                      'hist_cog_intensity_gt_200 binning')
    hist_intensity = Field(np.logspace(1., 6., 51), 'hist_intensity binning')
