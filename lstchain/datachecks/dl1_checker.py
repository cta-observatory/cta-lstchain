"""
Functions to check the contents of LST DL1 files and associated muon ring files
"""

__all__ = [
    'check_dl1',
    'process_dl1_file',
    'plot_datacheck',
    'plot_trigger_types',
    'plot_mean_and_stddev',
    'merge_dl1datacheck_files'
]

import logging
import os
from datetime import datetime
from multiprocessing import Pool
from pathlib import Path

import h5py
import matplotlib.colors as colors
import matplotlib.dates as dates
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import tables
from astropy import units as u
from astropy.table import Table, vstack
from ctapipe.containers import EventType
from ctapipe.coordinates import EngineeringCameraFrame
from ctapipe.instrument import SubarrayDescription
from ctapipe.io import HDF5TableWriter
from ctapipe.visualization import CameraDisplay
# from lstchain.visualization.bokeh import plot_mean_and_stddev_bokeh
# from bokeh.models.widgets import Panel
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import poisson, sem

from lstchain.datachecks.containers import DL1DataCheckContainer
from lstchain.datachecks.containers import DL1DataCheckHistogramBins
from lstchain.io.io import dl1_params_lstcam_key
from lstchain.paths import parse_datacheck_dl1_filename, parse_dl1_filename, \
    run_to_muon_filename, run_to_datacheck_dl1_filename


def check_dl1(filenames, output_path, max_cores=4, create_pdf=False, batch=False):
    """

    Parameters
    ----------
    batch: bool, run in batch mode
    create_pdf: bool, create PDF file
    filenames: string, Path, or a list of them, _sorted_ (by growing subrun
    index). Name(s) of the input DL1 .h5 file(s)
    output_path: directory where output will be written
    max_cores: maximum number of processes that the function will spawn (each
    processing a different subrun)

    Returns
    -------
    None

    """

    logger = logging.getLogger(__name__)

    # convert to list if it is not yet a a list:
    if not isinstance(filenames, list):
        filenames = [filenames]

    # Define output filename (overwrite if already existing).
    # If there is a single input file (i.e. a single subrun) then the output
    # file name will keep the subrun index. If there is more than one file
    # (i.e. several subruns) the output file name omit the subrun index.
    first_file = parse_dl1_filename(os.path.basename(filenames[0]))
    if len(filenames) == 1:
        datacheck_filename = run_to_datacheck_dl1_filename(first_file.tel_id,
                                                           first_file.run,
                                                           first_file.subrun)
    else:
        datacheck_filename = run_to_datacheck_dl1_filename(first_file.tel_id,
                                                           first_file.run,
                                                           None)
    datacheck_filename = Path(output_path, datacheck_filename)

    # the list dl1datacheck will contain one entry per subrun. Each entry is a
    # list of 3 containers of type DL1DataCheckContainer, one for pedestals,
    # one for flatfield events and one for cosmics

    # check that all files exist:
    for filename in filenames:
        if not os.path.exists(filename):
            logger.error(f'File {str(filename)} not found!')
            raise FileNotFoundError

    # try to determine which trigger_type tag is more reliable for
    # identifying interleaved pedestals. We check which one has
    # more values == 32, which is the pedestal tag. The one called
    # "trigger_type" is the TIB trigger type. The fastest way to do
    # this for the whole run seems to be using normal pytables:
    trig_tags = {'trigger_type': [], 'ucts_trigger_type': []}
    for filename in filenames:
        with tables.open_file(filename,
                              root_uep='/dl1/event/telescope/parameters') as f:
            for name in trig_tags:
                trig_tags[name].extend(f.root.LST_LSTCam.col(name))
    num_pedestals = {'trigger_type':
                         (np.array(trig_tags['trigger_type']) == 32).sum(),
                     'ucts_trigger_type':
                         (np.array(trig_tags['ucts_trigger_type']) == 32).sum()}
    logger.info(f'Number of == 32 (pedestal) trigger tags: {num_pedestals}')

    # Choose what source to use for obtaining the trigger type:
    trigger_source = 'event_type'

    # create container for the histograms' binnings, to be saved in the hdf5
    # output file:
    histogram_binning = DL1DataCheckHistogramBins()

    # create the dl1_datacheck containers (one per subrun) for the three
    # event types, and add them to the list dl1datacheck:
    with Pool(max_cores) as pool:
        func_args = [(filename, histogram_binning) for
                     filename in filenames]
        dl1datacheck = pool.starmap(process_dl1_file, func_args)
    # NOTE: the above does not seem to improve execution time on Mac OS X.
    # Perhaps related to numpy "sharing" between the processes?

    # or... process the files sequentially:
    # dl1datacheck = list([None]*len(filenames))
    # for i, filename in enumerate(filenames):
    #     dl1datacheck[i] = process_dl1_file(filename, histogram_binning)

    # NOTE: I do not think we may have memory problems, but if needed we could
    # write out the containers as they are produced.

    writer_conf = tables.Filters(complevel=9, complib='blosc:zstd',
                                 fletcher32=True)
    with HDF5TableWriter(datacheck_filename, filters=writer_conf) as writer:
        # write the containers (3 per subrun) to the dl1 data check output file
        # If container is None it means the filling was unsuccessful due to
        # no events of the given type. Write only filled containers:
        for dcheck in dl1datacheck:
            if dcheck[0] is not None:
                writer.write("dl1datacheck/pedestals", dcheck[0])
            if dcheck[1] is not None:
                writer.write("dl1datacheck/flatfield", dcheck[1])
            if dcheck[2] is not None:
                writer.write("dl1datacheck/cosmics", dcheck[2])
        # write also the histogram binnings:
        writer.write("dl1datacheck/histogram_binning", histogram_binning)

    subarray_info = SubarrayDescription.from_hdf(filenames[0])
    subarray_info.to_hdf(datacheck_filename)

    # write out also which trigger tag has been used for finding pedestals:
    file = h5py.File(datacheck_filename, mode='a')
    file.create_dataset('/dl1datacheck/used_trigger_tag', (1,), 'S32',
                        [trigger_source.encode('ascii')])
    file.close()

    # do the plots and save them to a pdf file. We will look for the muons fits
    # files in the same directory as the DL1 files (assuming all of them are
    # in the same directory as the first one!)
    if create_pdf:
        plot_datacheck(
            datacheck_filename,
            output_path,
            batch,
            muons_dir=os.path.dirname(filenames[0]),
            tel_id=first_file.tel_id
        )

    return


def process_dl1_file(filename, bins, tel_id=1):
    """

    Parameters
    ----------
    filename: string, or Path, input DL1 .h5 file to be checked
    bins: DL1DataCheckHistogramBins container indicating binning of histograms
    tel_id: int
        Telescope ID (default=1)

    Returns
    -------
    dl1datacheck_pedestals, dl1datacheck_flatfield, dl1datacheck_cosmics
    Containers of type DL1DataCheckContainer, with info on the three types of
    events: interleaved pedestals, interleaved flatfield events, and cosmics.
    If one or more of them is None, it means they have not been filled,
    due to lack of events if the given type in the input DL1 file.

    """

    logger = logging.getLogger(__name__)

    logger.info(f'Opening file {filename}')
    subrun_index = parse_dl1_filename(os.path.basename(filename)).subrun

    dl1datacheck_pedestals = DL1DataCheckContainer()
    dl1datacheck_flatfield = DL1DataCheckContainer()
    dl1datacheck_cosmics = DL1DataCheckContainer()

    subarray_info = SubarrayDescription.from_hdf(filename)
    geom = subarray_info.tel[tel_id].camera.geometry
    equivalent_focal_length = subarray_info.tel[tel_id].optics.equivalent_focal_length
    m2deg = np.rad2deg(u.m / equivalent_focal_length * u.rad) / u.m

    with tables.open_file(filename) as file:
        # unfortunately pandas.read_hdf does not seem compatible with
        # 'with... as...' statements
        parameters = pd.read_hdf(filename, key=dl1_params_lstcam_key)

        # convert cog distance to camera center from meters to degrees:
        parameters['r'] *= m2deg
        # time gradient from ns/m to ns/deg
        parameters['time_gradient'] /= m2deg

        # We do not convert the x,y, cog coordinates, because only in m can
        # CameraGeometry find the pixel where a given cog falls

        # in order to read in the images we have to use tables,
        # because pandas is not compatible with vector columns
        image_table = file.root.dl1.event.telescope.image.LST_LSTCam

        # create flatfield mask from the images table. For the time being,
        # trigger type tags are not reliable. We first identify flatfield events
        # by their looks.
        image = image_table.col('image')

        flatfield_mask = (parameters['event_type'] == EventType.FLATFIELD.value)
        # The same mask should be valid for image_table, since the entry in
        # the two tables correspond one to one.

        pedestal_mask = (parameters['event_type'] ==
                         EventType.SKY_PEDESTAL.value)

        # Now obtain by exclusion the masks for cosmics:
        cosmics_mask = ~(pedestal_mask | flatfield_mask)

        logger.info(f'   pedestals: {np.sum(pedestal_mask)}, '
                    f' flatfield: {np.sum(flatfield_mask)}, '
                    f' cosmics: {np.sum(cosmics_mask)}')

        # Fill quantities which depend on event-wise (i.e. not
        # pixel-wise) parameters.
        # Set None for a container that has not been filled,
        # otherwise it will give trouble in the plotting stage.

        if pedestal_mask.sum() > 1:
            dl1datacheck_pedestals.fill_event_wise_info(subrun_index,
                                                        parameters,
                                                        pedestal_mask,
                                                        geom, bins)
            dl1datacheck_pedestals.fill_pixel_wise_info(image_table,
                                                        pedestal_mask, bins,
                                                        'pedestals')
        else:
            dl1datacheck_pedestals = None

        if flatfield_mask.sum() > 1:
            dl1datacheck_flatfield.fill_event_wise_info(subrun_index,
                                                        parameters,
                                                        flatfield_mask,
                                                        geom, bins)
            dl1datacheck_flatfield.fill_pixel_wise_info(image_table,
                                                        flatfield_mask, bins,
                                                        'flatfield')
        else:
            dl1datacheck_flatfield = None

        if cosmics_mask.sum() > 1:
            dl1datacheck_cosmics.fill_event_wise_info(subrun_index,
                                                      parameters,
                                                      cosmics_mask,
                                                      geom, bins)
            dl1datacheck_cosmics.fill_pixel_wise_info(image_table,
                                                      cosmics_mask, bins,
                                                      'cosmics')
        else:
            dl1datacheck_cosmics = None

        return dl1datacheck_pedestals, dl1datacheck_flatfield, \
               dl1datacheck_cosmics


def plot_datacheck(datacheck_filename, out_path=None, batch=False, muons_dir=None, tel_id=1):
    """

    Parameters
    ----------
    datacheck_filename: list of strings, or pathlib.Path, name(s) of .h5
    files produced by the function check_dl1, starting from DL1 event files
    If it is a list of file names, we expect each of the files to correspond to
    one subrun of the same run.
    out_path: optional; if not given, it will be the same of file filename
    batch: bool, run in batch mode
    muons_dir
    tel_id: int
        Telescope ID (default=1)

    Returns
    -------
    None

    """

    logger = logging.getLogger(__name__)

    # aspect ratio of pdf pages:
    pagesize = [12., 7.5]

    # in case of >1 input file, we assume they correspond to subruns of a
    # given run. We merge them before proceeding:
    if isinstance(datacheck_filename, list):
        if len(datacheck_filename) > 1:
            merged_filename = merge_dl1datacheck_files(datacheck_filename)
            datacheck_filename = merged_filename
        else:
            # just a single .h5 file:
            datacheck_filename = datacheck_filename[0]

    pdf_filename = Path(datacheck_filename).with_suffix('.pdf')
    # set output directory if provided:
    if out_path is not None:
        pdf_filename = Path(out_path, pdf_filename.name)

    # Read camera geometry
    subarray_info = SubarrayDescription.from_hdf(datacheck_filename)
    geom = subarray_info.tel[tel_id].camera.geometry
    engineering_geom = geom.transform_to(EngineeringCameraFrame())

    # For future bokeh-based display, turned off for now:
    # page1 = Panel()
    # page2 = Panel()

    with PdfPages(pdf_filename) as pdf:
        # first deal with the DL1 datacheck file, created from DL1 event data:
        file = tables.open_file(datacheck_filename)
        # Read the binning of the stored histograms, and the info on
        # the source from which the trigger type info has been read:
        hist_binning = file.root.dl1datacheck.histogram_binning

        group = file.root.dl1datacheck
        # get the tables for each type of events, check first in each case that
        # the table exists

        if '/dl1datacheck/pedestals' in group:
            table_pedestals = file.root.dl1datacheck.pedestals
        else:
            logger.warning('No pedestals table found in ' +
                           str(datacheck_filename))
            table_pedestals = None

        if '/dl1datacheck/flatfield' in group:
            table_flatfield = file.root.dl1datacheck.flatfield
        else:
            logger.warning('No flatfield table found in ' +
                           str(datacheck_filename))
            table_flatfield = None

        if '/dl1datacheck/cosmics' in group:
            table_cosmics = file.root.dl1datacheck.cosmics
        else:
            logger.error('No cosmics table found in ' +
                         str(datacheck_filename))
            raise RuntimeError

        dl1dcheck_tables = [table_flatfield, table_pedestals, table_cosmics]
        labels = ['flatfield (guessed)', 'pedestals',
                  'cosmics']
        labels = [x for i, x in enumerate(labels)
                  if dl1dcheck_tables[i] is not None]
        dl1dcheck_tables = [x for x in dl1dcheck_tables if x is not None]

        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=pagesize)
        fig.tight_layout(pad=0.)
        plt.text(0.1, 0.7, os.path.basename(datacheck_filename),
                 fontsize=32, horizontalalignment='left',
                 verticalalignment='center')
        plt.text(0.1, 0.6, 'First shower event UTC: ', fontsize=24,
                 horizontalalignment='left', verticalalignment='center')
        plt.text(0.1, 0.5, '    (from Dragon time): ' +
                 str(datetime.utcfromtimestamp \
                         (table_cosmics.col('dragon_time')[0][0])),
                 fontsize=24, horizontalalignment='left',
                 verticalalignment='center')
        axes.axis('off')
        pdf.savefig()

        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=pagesize)
        fig.tight_layout(pad=3.0, h_pad=3.0, w_pad=2.0)

        plot_trigger_types(dl1dcheck_tables, 'ucts_trigger_type', axes[0, 0])
        plot_trigger_types(dl1dcheck_tables, 'trigger_type', axes[0, 1])

        for table, label in zip(dl1dcheck_tables, labels):
            fmt = '-'
            # in case of just one subrun, to make index-wise plots visible:
            if len(table.col('subrun_index')) == 1:
                fmt = 'o'

            # elapsed time: would better to take it always from the cosmics
            # table (will be closer to the true one), but number of entries
            # of tables can be different if e.g. pedestals or flatfield events
            # are missing in some subruns!
            elapsed_t = table.col('elapsed_time')
            axes[1, 0].plot(table.col('subrun_index'),
                            table.col('num_events') / elapsed_t, fmt,
                            label=label)
            axes[1, 1].plot(table.col('subrun_index'),
                            table.col('num_cleaned_events') /
                            table.col('num_events'), fmt, label=label)

        axes[1, 0].set_ylabel('rate (events/s)')
        axes[1, 0].set_yscale('log')
        axes[1, 1].set_ylabel('Fraction of events surviving cleaning')
        for j in (0, 1):
            axes[1, j].set_xlabel('subrun index')
            axes[1, j].legend(loc='best')

        pdf.savefig()

        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=pagesize)
        fig.tight_layout(pad=3.0, h_pad=3.0, w_pad=3.0)

        axes[0, 0].plot(table_cosmics.col('sampled_event_ids').flatten(),
                        table_cosmics.col('dragon_time').flatten(),
                        label='dragon_time')
        axes[0, 0].set_xlabel('event id')
        axes[0, 0].set_ylabel('timestamp')
        axes[0, 0].legend(loc='best')

        hist = 'hist_delta_t'
        bins = hist_binning.col(hist)[0]
        axes[0, 1].hist(bins[:-1], bins,
                        weights=np.sum(table_cosmics.col(hist), axis=0),
                        histtype='step')
        axes[0, 1].set_xlabel('delta_t (ms) from Dragon timestamp')
        axes[0, 1].set_ylabel('events')
        axes[0, 1].set_yscale('log')

        alt_deg = np.rad2deg(table_cosmics.col('mean_alt_tel'))
        axes[1, 0].plot(np.rad2deg(table_cosmics.col('mean_az_tel')), alt_deg,
                        fmt)
        axes[1, 0].set_xlabel('telescope azimuth (deg)')
        axes[1, 0].set_ylabel('telescope altitude (deg)')
        axes[1, 0].xaxis.set_major_formatter(mtick.FormatStrFormatter('%.1f'))
        axes[1, 0].yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1f'))

        dragon_time = table_cosmics.col('dragon_time')
        # dragon_time contains for each table row a number of times sampled at
        # regular event intervals. We get the mean per row (typically =subrun):
        mean_dragon_time = np.mean(dragon_time, axis=1)
        mpl_times = np.array([dates.date2num(datetime.utcfromtimestamp(x))
                              for x in mean_dragon_time])
        axes[1, 1].plot_date(mpl_times, alt_deg, fmt=fmt, xdate=True,
                             tz='utc')
        axes[1, 1].set_xlabel('time (UTC)')
        axes[1, 1].set_ylabel('telescope altitude (deg)')
        axes[1, 1].yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1f'))
        pdf.savefig()

        if table_pedestals is None or len(table_pedestals) == 0:
            write_error_page('pedestals', pagesize)
        else:
            # For future bokeh-based display, turned off for now:
            # page1.child = \
            #     plot_mean_and_stddev_bokeh(table_pedestals,
            #                                engineering_geom,
            #                                ['charge_mean', 'charge_stddev'],
            #                                ['Pedestal mean charge (p.e.)',
            #                                'Pedestal charge std dev (p.e.)',
            #                                'PEDESTALS, pixel-wise charge info'])
            # page1.title = 'PEDESTALS, pixel-wise charge info'

            plot_mean_and_stddev(table_pedestals, engineering_geom,
                                 ['charge_mean', 'charge_stddev'],
                                 ['Pedestal mean charge (p.e.)',
                                  'Pedestal charge std dev (p.e.)',
                                  'PEDESTALS, pixel-wise charge info'],
                                 pagesize, batch, norm='log')
        pdf.savefig()

        if table_flatfield is None or len(table_flatfield) == 0:
            write_error_page('flatfield', pagesize)
        else:
            # For future bokeh-based display, turned off for now:
            # page2.child = \
            #     plot_mean_and_stddev_bokeh(table_flatfield, engineering_geom,
            #                                ['charge_mean', 'charge_stddev'],
            #                                ['Flat-field mean charge (p.e.)',
            #                                'Flat-field charge std dev (p.e.)',
            #                                'FLATFIELD, pixel-wise charge info'])
            # page2.title = 'FLATFIELD, pixel-wise charge info'

            plot_mean_and_stddev(table_flatfield, engineering_geom,
                                 ['charge_mean', 'charge_stddev'],
                                 ['Flat-field mean charge (p.e.)',
                                  'Flat-field charge std dev (p.e.)',
                                  'FLATFIELD, pixel-wise charge info'], pagesize,
                                 batch, norm='log')
        pdf.savefig()

        # Displaying and saving of FUTURE bokeh display, not yet active:
        # output_file(pdf_filename.with_suffix('.html'),
        #             title='LST1 DL1 data check')
        # tabs = Tabs(tabs=[page1, page2])
        # show(column(Div(text='<h1>'+os.path.basename(datacheck_filename)+'</h1>'),
        #             tabs))

        histograms = ['hist_pixelchargespectrum', 'hist_intensity',
                      'hist_npixels', 'hist_nislands']
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=pagesize)
        fig.tight_layout(pad=3.0, h_pad=3.0, w_pad=2.0)
        for i, hist in enumerate(histograms):
            bins = hist_binning.col(hist)[0]
            for table in dl1dcheck_tables:
                contents = np.sum(table.col(hist), axis=0)
                axes.flatten()[i].hist(bins[:-1], bins, histtype='step',
                                       weights=contents / contents.sum(),
                                       label=table.name)
            axes.flatten()[i].set_yscale('log')
            axes.flatten()[i].set_xscale('log')
            axes.flatten()[i].set_ylabel('fraction of events of the given type')
        axes[0, 0].legend(loc='best')
        axes[0, 0].set_xlabel('Pixel charge (p.e.)')
        axes[0, 1].set_xlabel('Image intensity (p.e.)')
        axes[1, 0].set_xlabel('Number of pixels in image')
        axes[1, 1].set_xlabel('Number of islands in image')
        pdf.savefig()

        # We now plot the pixel rates above a few thresholds.
        # Find the thresholds (in pe) for which the event numbers are stored:
        colnames = [name for name in table_cosmics.colnames
                    if name.find('num_pulses_above') == 0]
        threshold = [int(name[name.find('above_') + 6:name.find('_pe')])
                     for name in colnames]

        for table, tname in zip([table_pedestals, table_cosmics],
                                ['pedestals', 'cosmics']):
            if table is None or len(table) == 0:
                write_error_page(tname, pagesize)
                pdf.savefig()
                continue

            # We asume here that 5 such thresholds are present in the
            # dl1datacheck file
            fig, axes = plt.subplots(nrows=2, ncols=3, figsize=pagesize)
            fig.suptitle(table.name.upper() +
                         ', relative frequency of pixel charges',
                         fontsize='xx-large')
            fig.tight_layout(rect=[0, 0.03, 1, 0.95], pad=3.0, h_pad=3.0,
                             w_pad=3.0)
            # sum (for all subruns) the number of events per pixel above the
            # different thresholds:
            pix_events = [np.sum(table.col(colname), axis=0)
                          for colname in colnames]
            # total number of entries for this event type:
            norm = table.col('num_events').sum()
            if norm > 0:
                fraction = np.array(pix_events) / norm
            for i, frac in enumerate(fraction):
                zscale = 'log' if threshold[i] < 200 and frac.sum() > 0 \
                    else 'lin'
                cam = CameraDisplay(engineering_geom, frac,
                                    ax=axes.flatten()[i], norm=zscale,
                                    title='Fraction of >' + str(threshold[i]) +
                                          ' p.e. pulses')
                cam.add_colorbar(ax=axes.flatten()[i], format='%.0e', pad=0.01)
                # same range for all cameras:
                axes.flatten()[i].set_xlim((axes[0, 0].get_xlim()))
                if not batch:
                    cam.show()
            for i in [1, 2, 4]:
                axes.flatten()[i].set_ylabel('')
            axes[1, 2].set_xscale('log')
            axes[1, 2].set_yscale('log')

            fraction_transposed = fraction.transpose()
            for y in fraction_transposed:
                if y.sum() > 0:
                    axes[1, 2].plot(threshold, y, 'o', fillstyle='none',
                                    alpha=0.2)
            axes[1, 2].set_xlabel('pixel charge (p.e.)')
            axes[1, 2].set_ylabel('fraction of events with charge>x')
            pdf.savefig()

            # Show also an evolution of the camera average (relative) rate
            # of >10, 30 pe signals, which should be around 0 for pedestals.
            # Whenever pedestal-tagged events are contaminated by cosmics or
            # other events, these rates will jump up.
            fmt = '-'
            # in case of just one subrun, to make index-wise plots visible:
            if len(table.col('subrun_index')) == 1:
                fmt = 'o'
            figb, axesb = plt.subplots(nrows=2, ncols=1, figsize=pagesize)
            figb.tight_layout(pad=3.0, h_pad=3.0, w_pad=3.0)
            figb.suptitle(
                table.name.upper() + ', relative frequency of pixel '
                                     'charges, camera averages',
                fontsize='xx-large')
            for i, y in enumerate(['num_pulses_above_0010_pe',
                                   'num_pulses_above_0030_pe']):
                if np.mean(table.col(y), axis=1).max() > 0:
                    axesb[i].set_yscale('log')
                axesb[i].plot(table.col('subrun_index'),
                              np.mean(table.col(y), axis=1) /
                              table.col('num_events'), fmt)
                axesb[i].set_xlabel('subrun index')
                axesb[i].set_ylim(top=1.)
                axesb[0].set_ylabel('Fraction of >10 p.e. pulses')
                axesb[1].set_ylabel('Fraction of >30 p.e. pulses')
            pdf.savefig()

        # Some plots on pulse times:
        if table_flatfield is None or len(table_flatfield) == 0:
            write_error_page('flatfield', pagesize)
        else:
            plot_mean_and_stddev(table_flatfield, engineering_geom,
                                 ['time_mean', 'time_stddev'],
                                 ['Flat-field mean time (ns)',
                                  'Flat-field time std dev (ns)',
                                  'FLATFIELD, pixel-wise pulse time info'],
                                 pagesize, batch)
            pdf.savefig()
            plot_mean_and_stddev(table_flatfield, engineering_geom,
                                 ['relative_time_mean',
                                  'relative_time_stddev'],
                                 ['Flat-field mean time (ns)',
                                  'Flat-field time std dev (ns)',
                                  'FLATFIELD, pixel-wise pulse time relative '
                                  'to camera mean'],
                                 pagesize, batch)

        pdf.savefig()

        plot_mean_and_stddev(table_cosmics, engineering_geom,
                             ['time_mean', 'time_stddev'],
                             ['Cosmics mean time (ns)',
                              'Cosmics time std dev (ns)',
                              'COSMICS, pixel-wise pulse time info for pixel '
                              'charge > 1 p.e.'], pagesize, batch)
        pdf.savefig()

        fig, axes = plt.subplots(nrows=2, ncols=3, figsize=pagesize)
        fig.suptitle('COSMICS, image c.o.g. position', fontsize='xx-large')
        fig.tight_layout(rect=[0, 0.03, 1, 0.95], pad=3.0, h_pad=3.0, w_pad=3.0)
        items = ['cog_within_pixel', 'cog_within_pixel_intensity_gt_200']
        titles = ['Image c.o.g.', 'Image c.o.g., intensity>200pe']
        for i, item in enumerate(items):
            events_per_pix = np.sum(table_cosmics.col(item), axis=0)
            all_events = np.sum(events_per_pix)
            event_fraction = events_per_pix / all_events
            cam = CameraDisplay(engineering_geom, event_fraction, ax=axes[i, 0],
                                norm='lin', title=titles[i])
            cam.add_colorbar(ax=axes[i, 0])
            if not batch:
                cam.show()
            camlog = CameraDisplay(engineering_geom, event_fraction,
                                   ax=axes[i, 1], norm='log', title=titles[i])
            camlog.add_colorbar(ax=axes[i, 1])
            # lines below needed to get all camera displays of equal size:
            axes[i, 0].set_xlim((axes[0, 0].get_xlim()))
            axes[i, 1].set_xlim((axes[0, 0].get_xlim()))
            if not batch:
                cam.show()
            # select pixels which are not on the edge of the camera:
            pix_inside = np.array([len(neig) == 6 for neig in geom.neighbors])
            # histogram the fraction of image cogs contained in those inner
            # pixels, to test homogeneity of distribution:
            # (only positive ones, for log-plotting)
            gt0 = event_fraction > 0
            axes[i, 2].set_xscale('log')

            nbins = 1001
            epb = (events_per_pix[pix_inside & gt0].max() -
                   events_per_pix[pix_inside & gt0].min()) / (nbins - 1)
            epb = int(epb + 1.)
            # make sure the same number of integers in each bin (otherwise we
            # will get "spikes" in the Poisson distribution later.number of
            # bins has to be large to achieve reasonable bin width with linear
            # binning, needed to avoid the spikes.
            xmin = events_per_pix[pix_inside & gt0].min() - 0.5
            xmax = xmin + (nbins - 1) * epb
            #  convert to event fraction:
            xmin /= all_events
            xmax /= all_events

            _, bins, _ = axes[i, 2]. \
                hist(event_fraction[pix_inside & gt0],
                     bins=np.linspace(xmin, xmax, nbins))
            # bins=np.logspace(np.log10(xmin), np.log10(xmax), nbins))
            # average event content:
            mu = np.sum(events_per_pix[pix_inside]) / pix_inside.sum()
            # get distribution of contents according to Poisson, integrating
            # the distribution within the same bins of the histogram above:
            poiss = np.array([poisson.cdf(x2 * all_events, mu) -
                              poisson.cdf(x1 * all_events, mu) for
                              x1, x2 in zip(bins[:-1], bins[1:])])
            # from probability to number of pixels:
            npixels = poiss * pix_inside.sum()
            # log bin centers:
            k = np.sqrt(bins[:-1] * bins[1:])
            axes[i, 2].plot(k[npixels > 0], npixels[npixels > 0],
                            drawstyle='steps-mid',
                            label='Poisson for uniform density')
            axes[i, 2].set_ylim(top=1.2 * axes[i, 2].get_ylim()[1])
            axes[i, 2].legend(loc='best')
            axes[i, 2].set_xlabel('Fraction of events')
            axes[i, 2].set_ylabel('# of pixels (excluding edge pixels)')
        pdf.savefig()

        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=pagesize)
        fig.suptitle('COSMICS, image parameters', fontsize='xx-large')
        fig.tight_layout(rect=[0.05, 0.05, 1.0, 0.9],
                         pad=0., h_pad=3.0, w_pad=2.0)
        histos = ['hist_dist0', 'hist_dist0_intensity_gt_200']
        for i, hist in enumerate(histos):
            bins = hist_binning.col(hist)[0]
            # normalize bin content by area of the corresponding ring:
            ringarea = np.pi * (bins[1:] ** 2 - bins[:-1] ** 2) * u.deg ** 2

            axes[i, 0].hist(bins[:-1], bins,
                            weights=np.sum(table_cosmics.col(hist), axis=0) /
                                    ringarea.value, histtype='step')
            axes[i, 0].set_xlabel('distance (deg)')
            axes[i, 0].set_ylabel('events per deg2')
        axes[0, 0].set_title('cog radial distribution')
        axes[1, 0].set_title('cog radial distribution, intensity>200pe')

        histos = ['hist_width', 'hist_length']
        for i, hist in enumerate(histos):
            bins = hist_binning.col(hist)[0]
            x = np.array([xx for xx in bins[0][:-1] for __ in bins[1][:-1]])
            y = np.array([yy for __ in bins[0][:-1] for yy in bins[1][:-1]])
            contents = np.sum(table_cosmics.col(hist), axis=0).flatten()
            _, _, _, image = axes[i, 1].hist2d(x, y, bins=bins,
                                               weights=contents,
                                               norm=colors.LogNorm())
            plt.colorbar(image, ax=axes[i, 1])
            axes[i, 1].set_xscale('log')
            axes[i, 1].set_xlabel('Intensity (p.e.)')
        axes[0, 1].set_ylabel('Width (deg)')
        axes[1, 1].set_ylabel('Length (deg)')
        pdf.savefig()

        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=pagesize)
        fig.suptitle('COSMICS, image parameters', fontsize='xx-large')
        fig.tight_layout(rect=[0.05, 0.05, 1.0, 0.9],
                         pad=0., h_pad=3.0, w_pad=2.0)
        histos = ['hist_skewness', 'hist_intercept', 'hist_tgrad_vs_length',
                  'hist_tgrad_vs_length_intensity_gt_200']
        for i, hist in enumerate(histos):
            bins = hist_binning.col(hist)[0]
            x = np.array([xx for xx in bins[0][:-1] for __ in bins[1][:-1]])
            y = np.array([yy for __ in bins[0][:-1] for yy in bins[1][:-1]])
            contents = np.sum(table_cosmics.col(hist), axis=0).flatten()
            _, _, _, image = axes.flatten()[i].hist2d(x, y, bins=bins,
                                                      weights=contents,
                                                      norm=colors.LogNorm())
            plt.colorbar(image, ax=axes.flatten()[i])
        axes[0, 0].set_ylabel('Skewness')
        axes[0, 0].grid(linewidth=0.3, linestyle=':')
        axes[0, 1].set_ylabel('Intercept (fitted time @ charge cog) (ns)')
        for j in [0, 1]:
            axes[0, j].set_xscale('log')
            axes[0, j].set_xlabel('Intensity')
            axes[1, j].set_xlabel('Length (deg)')
            axes[1, j].set_ylabel('Time gradient (ns/deg)')
        axes[1, 0].set_title('Time gradient vs. Length')
        axes[1, 1].set_title('Time gradient vs. Length, intensity>200pe')
        pdf.savefig()
        # End of the plots created from the DL1 datacheck file
        # keep some info needed for muon ring plots:
        subrun_list = np.array(table_cosmics.col('subrun_index'))
        elapsed_t = np.array(table_cosmics.col('elapsed_time'))
        file.close()

        # Now we go for the muons .fits files, created in the R0 to DL1 stage.
        # We look for the files with the same subrun indices that have been
        # processed. Make sure files exist and contain some rings: we keep
        # those in files_with_muons. The files which also have rings with
        # containment>0.999 are kept in good_files, and their
        # indices in good_subruns. These are safeguards introduced against
        # bad data!

        files_with_rings = []
        good_files = []
        good_subruns = np.array([])
        # Some quantities we want to have vs. subrun index:
        num_rings = np.array([])
        num_contained_rings = np.array([])
        mean_width = np.array([])
        sem_width = np.array([])
        mean_effi = np.array([])
        sem_effi = np.array([])

        dcfile = \
            parse_datacheck_dl1_filename(os.path.basename(datacheck_filename))
        for subrun in subrun_list:
            if muons_dir is not None:
                dirname = muons_dir
            # if no directory is provided, we assume the muons fits files are
            # in the same directory of the datacheck file.
            else:
                dirname = os.path.dirname(datacheck_filename)
            name = run_to_muon_filename(dcfile.tel_id, dcfile.run, subrun, None,
                                        False)
            filename = Path(dirname, name)
            if filename.exists():
                t = Table.read(filename)
                if len(t) > 0:
                    files_with_rings.append(filename)
                tcont = t[t['ring_containment'] > 0.999]
                num_rings = np.append(num_rings, len(t))
                num_contained_rings = np.append(num_contained_rings, len(tcont))
                if len(tcont) > 0:
                    good_files.append(filename)
                    good_subruns = np.append(good_subruns, subrun)
                else:
                    logger.warning(f'File {str(filename)} has no valid muon '
                                   f'rings!')
            else:
                logger.warning(f'File {str(filename)} not found. '
                               f'No muon information will be plotted for that '
                               f'subrun!')
                num_rings = np.append(num_rings, 0)
                num_contained_rings = np.append(num_contained_rings, 0)

        if len(files_with_rings) == 0:
            write_error_page('Muons', pagesize)
            pdf.savefig()
            return

        # Now join the tables which indeed contain data (joining those
        # without results in an error, hence all this complication!)
        muons_table = Table.read(files_with_rings[0])
        for filename in files_with_rings[1:]:
            t = Table.read(filename)
            muons_table = vstack([muons_table, t])

        t = Table.read(good_files[0])
        tcont = t[t['ring_containment'] > 0.999]
        mean_width = np.mean(tcont['ring_width'])
        sem_width = sem(tcont['ring_width'])
        mean_effi = np.mean(tcont['muon_efficiency'])
        sem_effi = sem(tcont['muon_efficiency'])
        contained_muons = tcont
        for filename in good_files[1:]:
            t = Table.read(filename)
            tcont = t[t['ring_containment'] > 0.999]
            mean_width = np.append(mean_width, np.mean(tcont['ring_width']))
            sem_width = np.append(sem_width, sem(tcont['ring_width']))
            mean_effi = np.append(mean_effi, np.mean(tcont['muon_efficiency']))
            sem_effi = np.append(sem_effi, sem(tcont['muon_efficiency']))
            contained_muons = vstack([contained_muons, tcont])

        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=pagesize)
        fig.suptitle('MUON RINGS', fontsize='xx-large')
        fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.97],
                         pad=3.0, h_pad=3.0, w_pad=2.0)
        fmt = '-'
        if len(subrun_list) == 1:
            fmt = 'o'
        axes[0, 0].set_ylim(0, num_rings.max() * 1.15)
        axes[0, 0].plot(subrun_list, num_rings, fmt, label='all rings in files')
        axes[0, 0].plot(subrun_list, num_contained_rings, fmt,
                        label='contained rings')
        axes[0, 0].set_ylabel('number of muon rings per subrun')
        axes[0, 0].legend(loc='best')

        muon_rate = num_rings / elapsed_t
        contained_muon_rate = num_contained_rings / elapsed_t
        axes[0, 1].set_ylim(0, muon_rate.max() * 1.15)
        axes[0, 1].plot(subrun_list, muon_rate, fmt, label='all rings in files')
        axes[0, 1].plot(subrun_list, contained_muon_rate, fmt,
                        label='contained rings')
        axes[0, 1].set_ylabel('rate of muon rings (events/s)')
        axes[0, 1].legend(loc='best')
        for j in (0, 1):
            axes[0, j].set_xlabel('subrun index')
        axes[1, 0].hist(muons_table['ring_containment'],
                        bins=np.linspace(0., 1., 51),
                        weights=np.ones(len(muons_table)) / num_rings.sum())
        axes[1, 0].set_xlabel('ring containment')
        binning = np.linspace(0., 1., 31)
        axes[1, 1].hist(muons_table['ring_completeness'],
                        bins=binning, histtype='step',
                        weights=np.ones(len(muons_table)) / num_rings.sum(),
                        label='all rings in files')
        axes[1, 1].hist(contained_muons['ring_completeness'], bins=binning,
                        histtype='step',
                        weights=np.ones(len(contained_muons)) / num_rings.sum(),
                        label='contained rings')
        axes[1, 1].set_xlabel('ring completeness')
        axes[1, 1].legend(loc='best')
        for j in (0, 1):
            axes[1, j].set_ylabel('fraction of rings')
        pdf.savefig()

        fig, axes = plt.subplots(nrows=2, ncols=3, figsize=pagesize)
        fig.suptitle('MUON RINGS with containment = 1', fontsize='xx-large')
        fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.97],
                         pad=3.0, h_pad=3.0, w_pad=2.0)
        axes[0, 0].hist(np.sqrt(contained_muons['ring_center_x'] ** 2. +
                                contained_muons['ring_center_y'] ** 2.),
                        bins=np.linspace(0., 2., 51))
        axes[0, 0].set_xlabel('ring center, distance from camera center (m)')
        axes[0, 0].set_ylabel('number of rings')
        axes[1, 0].plot(contained_muons['impact_parameter'],
                        contained_muons['ring_completeness'], 'x', alpha=0.5)
        axes[1, 0].set_xlabel('reconstructed impact parameter (m)')
        axes[1, 0].set_ylabel('ring completeness')
        axes[0, 1].plot(contained_muons['ring_radius'],
                        contained_muons['ring_size'], 'x', alpha=0.5)
        axes[0, 1].set_xlabel('ring radius (deg)')
        axes[0, 1].set_ylabel('ring intensity (p.e.)')
        axes[1, 1].plot(contained_muons['ring_radius'],
                        contained_muons['ring_width'], 'x', alpha=0.5)
        axes[1, 1].set_ylim(0., 0.3)
        axes[1, 1].set_xlabel('ring radius (deg)')
        axes[1, 1].set_ylabel('ring width (deg)')
        axes[0, 2].hist(contained_muons['ring_size'],
                        bins=np.linspace(0., 4.e3, 41))
        axes[0, 2].set_xlabel('ring intensity (p.e.)')
        axes[0, 2].set_ylabel('number of rings')
        axes[1, 2].hist(contained_muons['ring_width'],
                        bins=np.linspace(0., 0.3, 61))
        axes[1, 2].set_xlabel('ring width (deg)')
        axes[1, 2].set_ylabel('number of rings')
        pdf.savefig()

        fig, axes = plt.subplots(nrows=2, ncols=3, figsize=pagesize)
        fig.suptitle('MUON RINGS with containment = 1', fontsize='xx-large')
        fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.97],
                         pad=3.0, h_pad=3.0, w_pad=2.0)
        binning = np.linspace(-0.5, 38.5, 39)
        axes[0, 0].hist(contained_muons['hg_peak_sample'], bins=binning,
                        histtype='step')
        axes[0, 0].set_xlabel('HG peak sample in R1 waveform')
        axes[0, 0].set_ylabel('number of rings')

        axes[1, 0].plot(contained_muons['ring_completeness'],
                        contained_muons['ring_size'] /
                        contained_muons['ring_radius'], 'x', alpha=0.5)
        axes[1, 0].set_xlabel('ring completeness')
        axes[1, 0].set_ylabel('ring light (p.e.) / ring radius (deg)')

        axes[0, 1].hist(contained_muons['muon_efficiency'],
                        bins=np.linspace(0., 0.5, 51))
        axes[0, 1].set_xlabel('estimated telescope efficiency for muons')
        axes[0, 1].set_ylabel('number of rings')

        axes[1, 1].plot(contained_muons['ring_width'],
                        contained_muons['muon_efficiency'], 'x', alpha=0.5)
        axes[1, 1].set_xlim(0., 0.3)
        axes[1, 1].set_ylim(0., 0.5)
        axes[1, 1].set_xlabel('ring width (deg)')
        axes[1, 1].set_ylabel('estimated telescope efficiency for muons')

        axes[0, 2].errorbar(good_subruns, mean_effi, yerr=sem_effi, fmt='o',
                            markersize=3.)
        axes[0, 2].set_xlabel('subrun index')
        axes[0, 2].set_ylabel('estimated telescope efficiency for muons')
        axes[0, 2].grid(linewidth=0.3, linestyle=':')
        axes[0, 2].set_ylim(0., 0.5)
        axes[1, 2].errorbar(good_subruns, mean_width, yerr=sem_width, fmt='o',
                            markersize=3.)
        axes[1, 2].set_xlabel('subrun index')
        axes[1, 2].set_ylabel('ring width (deg)')
        axes[1, 2].grid(linewidth=0.3, linestyle=':')
        axes[1, 2].set_ylim(0., 0.3)
        pdf.savefig()


def plot_trigger_types(dchecktables, trigger_name, axes):
    """

    Parameters
    ----------
    dchecktables: array of python tables created with DL1DataCheckContainer
    containers (each row is one subrun). The plotted trigger type statistics
    will be the global ones, adding up the numbers from all the tables and
    all the rows in each table.
    Inside the table the trigger type columns have shape (n,10,2). n is the
    number of rows (one per subrun). 10 is the number of possible trigger
    types (just fixed to a safely large value). The remaining 2 are the pairs
    (trigger_id, number of entries with that id)

    trigger_name: name of the trigger type column in the tables
    axes: where to place the plots

    Returns
    -------
    None

    """

    # find all trigger types found in the subruns, and display histogram:
    # first merge subrun-wise tables:
    tt = dchecktables[0].col(trigger_name)
    for table in dchecktables[1:]:
        tt = np.append(tt, table.col(trigger_name), axis=0)
    # keep only entries with number of events > 0 (existing trig types):
    tt = tt[tt[:, :, 1] > 0]
    trig_types = np.unique(tt[:, 0])
    num_triggers = np.array([(tt[:, 1][tt[:, 0] == trig]).sum()
                             for trig in trig_types])
    x = np.arange(2 + len(trig_types))
    # for better display, leave some space on the sides of the bars:
    y = np.append([0], np.append(num_triggers, [0]))
    labels = [''] + [str(i) for i in trig_types] + ['']
    width = 0.3
    axes.bar(x, y, width)
    axes.set_xticks(x)
    axes.set_xticklabels(labels)
    axes.set_yscale('log')
    axes.set_xlabel(trigger_name)
    axes.set_ylabel('number of events')


def plot_mean_and_stddev(table, camgeom, columns, labels, pagesize, batch=False, norm='lin'):
    """
    Parameters
    ----------
    batch: bool, run in batch mode
    table:  python table containing pixel-wise information to be displayed
    camgeom: camera geometry
    columns: list of 2 strings, columns of 'table', first one is the mean and
    the second the std deviation to be plotted
    labels: plot titles
    pagesize: [width, height] in cm
    norm:  lin or log, z-scale of camera displays

    Returns
    -------
    None

    The subrun-wise mean and std dev values are used to calculate the
    run-wise (i.e. for all processed subruns which appear in the table)
    counterparts of the same, which are then plotted.

    """

    logger = logging.getLogger(__name__)

    # calculate pixel-wise mean and standard deviation for the whole run,
    # from the subrun-wise values:
    mean = np.sum(np.multiply(table.col(columns[0]),
                              table.col('num_events')[:, None]),
                  axis=0) / np.sum(table.col('num_events'))
    stddev = np.sqrt(np.sum(np.multiply(table.col(columns[1]) ** 2,
                                        table.col('num_events')[:, None]),
                            axis=0) / np.sum(table.col('num_events')))

    if np.isnan(mean).sum() > 0:
        logger.info(f'Pixels with NaNs in {columns[0]}: '
                    f'{np.array(camgeom.pix_id.tolist())[np.isnan(mean)]}')

    # plot mean and std dev (of e.g. pedestal charge or time), as camera
    # display, vs. pixel id, and as a histogram:
    fig, axes = plt.subplots(nrows=2, ncols=3,
                             figsize=pagesize)
    fig.suptitle(labels[2], fontsize='xx-large')
    fig.tight_layout(rect=[0, 0.03, 1, 0.98], pad=3.0, h_pad=3.0, w_pad=2.0)
    cam = CameraDisplay(camgeom, mean, ax=axes[0, 0], norm=norm,
                        title=labels[0])
    cam.add_colorbar(ax=axes[0, 0])
    if not batch:
        cam.show()
    cam = CameraDisplay(camgeom, stddev, ax=axes[1, 0], norm=norm,
                        title=labels[1])
    cam.add_colorbar(ax=axes[1, 0])
    # line below needed to get the top and bottom camera displays of equal size:
    axes[1, 0].set_xlim((axes[0, 0].get_xlim()))
    if not batch:
        cam.show()
    # plot mean vs. pixel_id and as histogram:
    axes[0, 1].plot(camgeom.pix_id, mean)
    axes[0, 1].set_xlabel('Pixel id')
    axes[0, 1].set_ylabel(labels[0])
    axes[0, 2].set_yscale('log')
    axes[0, 2].hist(mean[~np.isnan(mean)], bins=200)
    axes[0, 2].set_xlabel(labels[0])
    axes[0, 2].set_ylabel('Number of pixels')
    # now the standard deviation:
    axes[1, 1].plot(camgeom.pix_id, stddev)
    axes[1, 1].set_xlabel('Pixel id')
    axes[1, 1].set_ylabel(labels[1])
    axes[1, 2].set_yscale('log')
    axes[1, 2].hist(stddev[~np.isnan(stddev)], bins=200)
    axes[1, 2].set_xlabel(labels[1])
    axes[1, 2].set_ylabel('Number of pixels')


def write_error_page(tablename, pagesize):
    """

    Parameters
    ----------
    tablename: name of the table which has no entries to be plotted
    pagesize: [width, height] (cm)
    Returns
    -------
    None
    """
    _, axes = plt.subplots(nrows=1, ncols=1, figsize=pagesize)
    plt.text(0.5, 0.5, 'Sorry, no ' + tablename + ' to plot here!',
             fontsize=44, horizontalalignment='center',
             verticalalignment='center')
    axes.axis('off')


def merge_dl1datacheck_files(file_list):
    """

    Parameters
    ----------
    file_list: list of strings, names of files of the kind produced by
    function check_dl1

    Returns
    -------
    merged_filename: name of the .h5 file which contains all the rows of the
    files in the list (in the tables cosmics, pedestals and flatfield)
    The camera geometry, histogram_binnings and used_trigger_tag are copied
    just from the first file

    """

    logger = logging.getLogger(__name__)

    first_file_name = file_list[0]
    first_file = tables.open_file(first_file_name)
    #  get run number and build the name of the merged file:
    file = parse_datacheck_dl1_filename(os.path.basename(first_file_name))
    merged_filename = run_to_datacheck_dl1_filename(file.tel_id, file.run,
                                                    None, None)
    # Store the merged file in the same directory as the subrun-wise files:
    merged_filename = Path(os.path.dirname(first_file_name), merged_filename)

    # The input (sub-run wise) list should never contain the name of the
    # run-wise file that we will produce by merging. Just to avoid accidents:
    if str(merged_filename) in file_list:
        file_list.remove(str(merged_filename))

    logger.info(file_list)

    merged_file = tables.open_file(merged_filename, 'w')
    merged_file.create_group('/', 'dl1datacheck')
    merged_file.create_group('/', 'instrument')

    # The tables in the merged file will be copied from the first file. If a
    # table is missing in the first file (e.g. pedestals) it will be left
    # empty in the whole merged file.

    if '/dl1datacheck/pedestals' in first_file.root.dl1datacheck:
        pedestals = \
            first_file.copy_node('/dl1datacheck', name='pedestals',
                                 newparent=merged_file.root.dl1datacheck)
    else:
        pedestals = None

    if '/dl1datacheck/flatfield' in first_file.root.dl1datacheck:
        flatfield = \
            first_file.copy_node('/dl1datacheck', name='flatfield',
                                 newparent=merged_file.root.dl1datacheck)
    else:
        flatfield = None

    # the ones below are compulsory, an exception will be raised if not present:
    cosmics = first_file.copy_node('/dl1datacheck', name='cosmics',
                                   newparent=merged_file.root.dl1datacheck)
    first_file.copy_node('/dl1datacheck', name='histogram_binning',
                         newparent=merged_file.root.dl1datacheck)
    first_file.copy_node('/dl1datacheck', name='used_trigger_tag',
                         newparent=merged_file.root.dl1datacheck)
    first_file.close()

    for filename in file_list[1:]:
        file = tables.open_file(filename)
        if pedestals is not None:
            if '/dl1datacheck/pedestals' in file.root.dl1datacheck:
                pedestals.append(file.root.dl1datacheck.pedestals[:])
            else:
                logger.warning('Table pedestals is missing in file ' +
                               str(filename))
        if flatfield is not None:
            if '/dl1datacheck/flatfield' in file.root.dl1datacheck:
                flatfield.append(file.root.dl1datacheck.flatfield[:])
            else:
                logger.warning('Table flatfield is missing in file ' +
                               str(filename))

        cosmics.append(file.root.dl1datacheck.cosmics[:])
        file.close()

    merged_file.close()

    # For copying the camera geometry we use astropy tables to avoid a
    # NaturalNameWarning from tables/path.py
    subarray_info = SubarrayDescription.from_hdf(first_file_name)
    subarray_info.to_hdf(merged_filename)

    return merged_filename
