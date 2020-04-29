"""
Functions to check the contents ofLST DL1 files and associated muon ring files
"""

__all__ = [
    'check_dl1',
    'process_dl1_file',
    'plot_datacheck',
    'plot_trigger_types',
    'plot_mean_and_stddev',
]

import h5py
import matplotlib.colors as colors
import matplotlib.dates as dates
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import os
import pandas as pd
import tables

from astropy import units as u
from astropy.table import Table, vstack
from ctapipe.coordinates import EngineeringCameraFrame
from ctapipe.instrument import CameraGeometry
from ctapipe.io import HDF5TableWriter
from ctapipe.visualization import CameraDisplay
from datetime import datetime
from lstchain.datachecks.containers import DL1DataCheckContainer
from lstchain.datachecks.containers import DL1DataCheckHistogramBins
from lstchain.io.io import dl1_params_lstcam_key
from matplotlib.backends.backend_pdf import PdfPages
from multiprocessing import Pool
from scipy.stats import poisson, sem
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

    # obtain the names of the corresponding muon .fits files:
    muon_filenames = []
    for filename in filenames:
        name = filename.replace('dl1', 'muons')
        # put the correct .fits extension (both for the XXXX.h5 and the
        # deprecated XXXX.fits.h5 conventions of DL1 files):
        name = name.replace('.fits.h5', '.fits')
        name = name.replace('.h5', '.fits')
        # patch for DL1 files which contain the "stream tag" in the name:
        name = name.replace('LST-1.1', 'LST-1')
        muon_filenames.append(name)

    # define output filename (overwrite if already existing)
    datacheck_filename = output_path + '/datacheck_' + filename_prefix + \
     f'Run{run_number:05}.h5'
    # patch for DL1 files which contain the "stream tag" in the name e.g.
    # LST-1.1:
    datacheck_filename = datacheck_filename.replace('LST-1.1', 'LST-1')

    if os.path.exists(datacheck_filename):
        os.remove(datacheck_filename)

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
            print("File", filename, "not found!")
            raise FileNotFoundError
    for filename in muon_filenames:
        if not os.path.exists(filename):
            print("File", filename, "not found!")
            raise FileNotFoundError

    # now try to determine which trigger_type tag is more reliable for
    # identifying interlaved pedestals. We choose (for now) the one which
    # has more values == 32 which is the pedestal tag. The one called
    # "trigger_type" seems to be the TIB trigger type. The fastest way to do
    # this for the whole run seems to be using normal pytables:
    trig_tags = {'trigger_type': [], 'ucts_trigger_type': []}
    for filename in filenames:
        with tables.open_file(filename,
                              root_uep='/dl1/event/telescope/parameters') as f:
            for name in trig_tags.keys():
                trig_tags[name].extend(f.root.LST_LSTCam.col(name))
    num_pedestals = {'trigger_type':
                         (np.array(trig_tags['trigger_type']) == 32).sum(),
                     'ucts_trigger_type':
                         (np.array(trig_tags['ucts_trigger_type']) == 32).sum()}
    print("Number of == 32 (pedestal) trigger tags:")
    print('   ', num_pedestals)

    trigger_source = 'trigger_type'
    if num_pedestals['ucts_trigger_type'] > num_pedestals['trigger_type']:
        trigger_source = 'ucts_trigger_type'

    # create container for the histograms' binnings, to be saved in the hdf5
    # output file:
    histogram_binning = DL1DataCheckHistogramBins()

    # create the dl1_datacheck containers (one per subrun) for the three
    # event types, and add them to the list dl1datacheck:
    with Pool(max_cores) as pool:
        func_args = [(filename, histogram_binning, trigger_source) for
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
        # write the containers (3 per subrun) to the dl1 data check output file:
        for dcheck in dl1datacheck:
            writer.write("dl1datacheck/pedestals", dcheck[0])
            writer.write("dl1datacheck/flatfield", dcheck[1])
            writer.write("dl1datacheck/cosmics", dcheck[2])
        # write also the histogram binnings:
        writer.write("dl1datacheck/histogram_binning", histogram_binning)

    # we assume that cam geom is the same in all files, & write the first one
    # we convert units from m to deg
    cam_description_table = \
        Table.read(filename, path='instrument/telescope/camera/LSTCam')
    geom = CameraGeometry.from_table(cam_description_table)
    geom.to_table().write(datacheck_filename,
                          path=f'/instrument/telescope/camera/LSTCam',
                          append=True, serialize_meta=True)

    # write out also which trigger tag has been used for finding pedestals:
    file = h5py.File(datacheck_filename, mode='a')
    file.create_dataset('/dl1datacheck/used_trigger_tag', (1,), 'S32',
                        [trigger_source.encode('ascii')])
    file.close()

    # do the plots and save them to a pdf file:
    plot_datacheck(datacheck_filename, output_path)

    return


def process_dl1_file(filename, bins, trigger_source='trigger_type'):
    """

    Parameters
    ----------
    filename: string, input DL1 .h5 file to be checked
    bins: DL1DataCheckHistogramBins container indicating binning of histograms
    trigger_source: string, name of one of the trigger tags present in the
    DL1 file

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

    cam_description_table = \
        Table.read(filename, path='instrument/telescope/camera/LSTCam')
    geom = CameraGeometry.from_table(cam_description_table)
    optics_description_table = \
        Table.read(filename, path='instrument/telescope/optics')
    equivalent_focal_length = \
        optics_description_table['equivalent_focal_length']
    m2deg = np.rad2deg(u.m/equivalent_focal_length*u.rad)/u.m

    with tables.open_file(filename) as file:
        # unfortunately pandas.read_hdf does not seem compatible with
        # 'with... as...' statements
        parameters = pd.read_hdf(filename, key=dl1_params_lstcam_key)

        # convert parameters from meters to degrees:
        for var in ['r', 'width', 'length']:
            parameters[var] *= m2deg
        # time gradient from ns/m to ns/deg
        parameters['time_gradient'] /= m2deg

        # We do not convert the x,y, cog coordinates, because only in m can
        # CameraGeometry find the pixel where a given cog falls

        # in order to read in the images we have to use tables,
        # because pandas is not compatible with vector columns
        image_table = file.root.dl1.event.telescope.image.LST_LSTCam

        # create masks for the images table. For the time being, trigger
        # type tags are not reliable. We first identify flatfield events by
        # their looks, then use trigger_source (name of one of the trigger
        # tags in the DL1 file) to identify pedestals:
        image = image_table.col('image')
        flatfield_mask = ((np.median(image, axis=1) >
                           ff_min_pixel_charge_median) &
                          (np.std(image, axis=1) <
                           ff_max_pixel_charge_stddev))
        pedestal_mask = ~flatfield_mask & \
                        (image_table.col(trigger_source) == 32)
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
                                                    params_pedestal_mask,
                                                    geom, bins)
        dl1datacheck_flatfield.fill_event_wise_info(subrun_index, parameters,
                                                    params_flatfield_mask,
                                                    geom, bins)
        dl1datacheck_cosmics.fill_event_wise_info(subrun_index, parameters,
                                                  params_cosmics_mask,
                                                  geom, bins)

        # now fill pixel-wise information:
        dl1datacheck_pedestals.fill_pixel_wise_info(image_table,
                                                    pedestal_mask, bins)
        dl1datacheck_flatfield.fill_pixel_wise_info(image_table,
                                                    flatfield_mask, bins)
        dl1datacheck_cosmics.fill_pixel_wise_info(image_table,
                                                  cosmics_mask, bins)

    return dl1datacheck_pedestals, dl1datacheck_flatfield, dl1datacheck_cosmics


def plot_datacheck(datacheck_filename, out_path=None):
    """

    Parameters
    ----------
    datacheck_filename: .h5 file produced by the method check_dl1, starting
    from DL1 event files
    out_path: optional, if not given it will be the same of file filename

    Returns
    -------
    None

    """

    # aspect ratio of pdf pages:
    pagesize = [12., 7.5]

    pdf_filename = datacheck_filename.replace('.h5', '.pdf')
    if out_path is not None:
        pdf_filename = out_path+'/'+pdf_filename[pdf_filename.rfind('/')+1:]

    cam_description_table = \
        Table.read(datacheck_filename,
                   path='instrument/telescope/camera/LSTCam')
    geom = CameraGeometry.from_table(cam_description_table)
    engineering_geom = geom.transform_to(EngineeringCameraFrame())

    with PdfPages(pdf_filename) as pdf:
        # first deal with the DL1 datacheck file, created from DL1 event data:
        file = tables.open_file(datacheck_filename)

        # get the binning of the stored histograms:
        hist_binning = file.root.dl1datacheck.histogram_binning

        # read which triger tag has been used to identify pedestals:
        trigger_source = file.root.dl1datacheck.used_trigger_tag[0].decode()

        # get the tables for each type of events:
        table_pedestals = file.root.dl1datacheck.pedestals
        table_flatfield = file.root.dl1datacheck.flatfield
        table_cosmics = file.root.dl1datacheck.cosmics
        dl1dcheck_tables = [table_flatfield, table_pedestals, table_cosmics]

        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=pagesize)
        fig.tight_layout(pad=3.0, h_pad=3.0, w_pad=2.0)

        plot_trigger_types(dl1dcheck_tables, 'ucts_trigger_type', axes[0, 0])
        plot_trigger_types(dl1dcheck_tables, 'trigger_type', axes[0, 1])

        labels = ['flatfield (guessed)', 'pedestals (from '+trigger_source+')',
                  'cosmics']
        for table, label in zip(dl1dcheck_tables, labels):
            axes[1, 0].plot(table.col('subrun_index'), table.col('num_events'),
                            label=label)
            # elapsed time better from the cosmics table, will be closer to
            # the true one:
            elapsed_t = table_cosmics.col('elapsed_time')
            axes[1, 1].plot(table.col('subrun_index'),
                            table.col('num_events') / elapsed_t,
                            label=label)
        axes[1, 0].set_ylabel('number of events')
        axes[1, 1].set_ylabel('rate (events/s)')
        for j in (0, 1):
            axes[1, j].set_xlabel('subrun index')
            axes[1, j].set_yscale('log')
            axes[1, j].legend(loc='best')

        pdf.savefig()

        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=pagesize)
        fig.tight_layout(pad=3.0, h_pad=3.0, w_pad=3.0)

        for time_type in ['ucts_time', 'tib_time', 'dragon_time']:
            axes[0, 0].plot(table_cosmics.col('sampled_event_ids').flatten(),
                            table_cosmics.col(time_type).flatten(),
                            drawstyle='steps-mid', label=time_type)
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
        axes[1, 0].plot(np.rad2deg(table_cosmics.col('mean_az_tel')), alt_deg)
        axes[1, 0].set_xlabel('telescope azimuth (deg)')
        axes[1, 0].set_ylabel('telescope altitude (deg)')
        axes[1, 0].xaxis.set_major_formatter(mtick.FormatStrFormatter('%.1f'))
        axes[1, 0].yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1f'))

        dragon_time = table_cosmics.col('dragon_time')
        # dragon_time contains for each table row a number of times sampled at
        # regular event intervals. We get the mean per row (typically =subrun):
        mean_dragon_time = np.mean(dragon_time, axis=1)
        mpl_times = np.array([dates.date2num(datetime.fromtimestamp(x))
                                             for x in mean_dragon_time])
        axes[1, 1].plot_date(mpl_times, alt_deg, fmt='-', xdate=True,
                             tz='utc')
        axes[1, 1].set_xlabel('time (UTC)')
        axes[1, 1].set_ylabel('telescope altitude (deg)')
        axes[1, 1].yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1f'))
        pdf.savefig()

        if len(table_pedestals) == 0:
            write_error_page('pedestals', pagesize)
        else:
            plot_mean_and_stddev(table_pedestals, engineering_geom,
                                 ['charge_mean', 'charge_stddev'],
                                 ['Pedestal mean charge (p.e.)',
                                  'Pedestal charge std dev (p.e.)',
                                  'PEDESTALS, pixel-wise charge info'],
                                 pagesize, norm='log')
        pdf.savefig()

        plot_mean_and_stddev(table_flatfield, engineering_geom,
                             ['charge_mean', 'charge_stddev'],
                             ['Flat-field mean charge (p.e.)',
                              'Flat-field charge std dev (p.e.)',
                              'FLATFIELD, pixel-wise charge info'], pagesize,
                             norm='log')
        pdf.savefig()

        histograms = ['hist_pixelchargespectrum', 'hist_intensity',
                      'hist_npixels', 'hist_nislands']
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=pagesize)
        fig.tight_layout(pad=3.0, h_pad=3.0, w_pad=2.0)
        for i, hist in enumerate(histograms):
            bins = hist_binning.col(hist)[0]
            for table in dl1dcheck_tables:
                contents = np.sum(table.col(hist), axis=0)
                axes.flatten()[i].hist(bins[:-1], bins, histtype='step',
                                       weights=contents/contents.sum())
            axes.flatten()[i].set_yscale('log')
            axes.flatten()[i].set_xscale('log')
            axes.flatten()[i].set_ylabel('fraction of events of the given type')
        axes[0, 0].set_xlabel('Pixel charge (p.e.)')
        axes[0, 1].set_xlabel('Image intensity (p.e.)')
        axes[1, 0].set_xlabel('Number of pixels in image')
        axes[1, 1].set_xlabel('Number of islands in image')
        pdf.savefig()

        # We now plot the pixel rates above a few thresholds.
        # Find the thresholds (in pe) for which the event numbers are stored:
        colnames = [name for name in table_cosmics.colnames
                    if name.find('num_pulses_above') == 0]
        threshold = [int(name[name.find('above_')+6:name.find('_pe')])
                     for name in colnames]

        for table in [table_pedestals, table_cosmics]:
            if len(table) == 0:
                write_error_page(table.name, pagesize)
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
            fraction = np.array(pix_events)/norm
            for i, colname in enumerate(colnames):
                zscale = 'log' if threshold[i] < 200 else 'lin'
                cam = CameraDisplay(engineering_geom, fraction[i],
                                    ax=axes.flatten()[i], norm=zscale,
                                    title='Fraction of >' + str(threshold[i]) +
                                          ' p.e. pulses')
                cam.add_colorbar(ax=axes.flatten()[i], format='%.0e', pad=0.01)
                # same range for all cameras:
                axes.flatten()[i].set_xlim((axes[0, 0].get_xlim()))
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

        # Some plots on pulse times:
        if len(table_flatfield) == 0:
            write_error_page(table_flatfield.name, pagesize)
        else:
            plot_mean_and_stddev(table_flatfield, engineering_geom,
                                 ['time_mean', 'time_stddev'],
                                 ['Flat-field mean time (ns)',
                                  'Flat-field time std dev (ns)',
                                  'FLATFIELD, pixel-wise pulse time info'],
                                 pagesize)
        pdf.savefig()

        plot_mean_and_stddev(table_cosmics, engineering_geom,
                             ['time_mean', 'time_stddev'],
                             ['Cosmics mean time (ns)',
                              'Cosmics time std dev (ns)',
                              'COSMICS, pixel-wise pulse time info'], pagesize)
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
            cam.show()
            camlog = CameraDisplay(engineering_geom, event_fraction,
                                   ax=axes[i, 1], norm='log', title=titles[i])
            camlog.add_colorbar(ax=axes[i, 1])
            # lines below needed to get all camera displays of equal size:
            axes[i, 0].set_xlim((axes[0, 0].get_xlim()))
            axes[i, 1].set_xlim((axes[0, 0].get_xlim()))
            cam.show()
            # select pixels which are not on the edge of the camera:
            pix_inside = np.array([len(neig) == 6 for neig in geom.neighbors])
            # histogram the fraction of image cogs contained in those inner
            # pixels, to test homogeneity of distribution:
            # (only positive ones, for log-plotting)
            gt0 = event_fraction > 0
            xmin = event_fraction[pix_inside & gt0].min()
            xmax = event_fraction[pix_inside & gt0].max()
            axes[i, 2].set_xscale('log')
            _, bins, _ = axes[i, 2].\
                hist(event_fraction[pix_inside & gt0],
                     bins=np.logspace(np.log10(xmin), np.log10(xmax), 201))
            # average event content:
            mu = np.sum(events_per_pix[pix_inside])/pix_inside.sum()
            # get distribution of contents according to Poisson, integrating
            # the distribution within the same bins of the histogram above:
            poiss = np.array([poisson.cdf(x2*all_events, mu) -
                              poisson.cdf(x1*all_events, mu) for
                              x1, x2 in zip(bins[:-1], bins[1:])])
            # from probability to number of pixels:
            npixels = poiss * pix_inside.sum()
            # log bin centers:
            k = np.sqrt(bins[:-1]*bins[1:])
            axes[i, 2].plot(k[npixels > 0], npixels[npixels > 0],
                            drawstyle='steps-mid',
                            label='Poisson for uniform density')
            axes[i, 2].set_ylim(top=1.2*axes[i, 2].get_ylim()[1])
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
            ringarea = np.pi*(bins[1:]**2-bins[:-1]**2)*u.deg**2

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
        subrun_list = np.array(table.col('subrun_index'))
        elapsed_t = np.array(table_cosmics.col('elapsed_time'))
        file.close()

        # Now we go for the muons .fits files, created in the R0 to DL1 stage.
        # We look for the files with the same subrun indices that have been
        # processed.
        muon_filenames = []
        for i in subrun_list:
            name = datacheck_filename.replace('datacheck_dl1', 'muons')
            name = name.replace('.h5', f'.{i:04}.fits')
            muon_filenames.append(name)

        muons_table = Table.read(muon_filenames[0])
        contained_muons = muons_table[muons_table['ring_containment'] > 0.999]
        # to get some quantities vs. subrun index:
        num_rings = np.array([len(muons_table)])
        num_contained_rings = np.array([len(contained_muons)])
        mean_width = np.array(np.mean(contained_muons['ring_width']))
        sem_width = np.array(sem(contained_muons['ring_width']))
        mean_effi = np.array(np.mean(contained_muons['muon_efficiency']))
        sem_effi = np.array(sem(contained_muons['muon_efficiency']))

        for filename in muon_filenames[1:]:
            t = Table.read(filename)
            tcont = t[t['ring_containment'] > 0.999]
            # to get some quantities vs. subrun index:
            num_rings = np.append(num_rings, len(t))
            num_contained_rings = np.append(num_contained_rings, len(tcont))
            mean_width = np.append(mean_width, np.mean(tcont['ring_width']))
            sem_width = np.append(sem_width, sem(tcont['ring_width']))
            mean_effi = np.append(mean_effi, np.mean(tcont['muon_efficiency']))
            sem_effi = np.append(sem_effi, sem(tcont['muon_efficiency']))
            # to get the whole muon rings tables:
            muons_table = vstack([muons_table, t])
            contained_muons = vstack([contained_muons, tcont])

        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=pagesize)
        fig.suptitle('MUON RINGS', fontsize='xx-large')
        fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.97],
                         pad=3.0, h_pad=3.0, w_pad=2.0)
        axes[0, 0].set_ylim(0, num_rings.max()*1.15)
        axes[0, 0].plot(subrun_list, num_rings, '-', label='all rings in files')
        axes[0, 0].plot(subrun_list, num_contained_rings, '-',
                        label='contained rings')
        axes[0, 0].set_ylabel('number of muon rings per subrun')
        axes[0, 0].legend(loc='best')

        muon_rate = num_rings / elapsed_t
        contained_muon_rate = num_contained_rings / elapsed_t
        axes[0, 1].set_ylim(0, muon_rate.max()*1.15)
        axes[0, 1].plot(subrun_list, muon_rate, '-', label='all rings in files')
        axes[0, 1].plot(subrun_list, contained_muon_rate, '-',
                        label='contained rings')
        axes[0, 1].set_ylabel('rate of muon rings')
        axes[0, 1].legend(loc='best')
        for j in (0, 1):
            axes[0, j].set_xlabel('subrun index')
        axes[1, 0].hist(muons_table['ring_containment'],
                        bins=np.linspace(0., 1., 51),
                        weights=np.ones(len(muons_table))/num_rings.sum())
        axes[1, 0].set_xlabel('ring containment')
        binning = np.linspace(0., 1., 31)
        axes[1, 1].hist(muons_table['ring_completeness'],
                        bins=binning, histtype='step',
                        weights=np.ones(len(muons_table))/num_rings.sum(),
                        label='all rings in files')
        axes[1, 1].hist(contained_muons['ring_completeness'], bins=binning,
                        histtype='step',
                        weights=np.ones(len(contained_muons))/num_rings.sum(),
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
        axes[0, 0].hist(np.sqrt(contained_muons['ring_center_x']**2. +
                                contained_muons['ring_center_y']**2.),
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
                         pad=2.0, h_pad=3.0, w_pad=3.0)
        axes[0, 0].plot(contained_muons['hg_peak_sample'],
                        contained_muons['lg_peak_sample'], 'x', alpha=0.5)
        axes[0, 0].set_xlabel('High gain peak sample in R1 waveform')
        axes[0, 0].set_ylabel('Low gain peak sample in R1 waveform')
        binning = np.linspace(-0.5, 38.5, 39)
        axes[1, 0].hist(contained_muons['hg_peak_sample'], bins=binning,
                        histtype='step', label='HG')
        axes[1, 0].hist(contained_muons['lg_peak_sample'], bins=binning,
                        histtype='step', label='LG')
        axes[1, 0].set_xlabel('peak sample in R1 waveform')
        axes[1, 0].set_ylabel('number of rings')
        axes[1, 0].legend(loc='best')
        axes[0, 1].hist(contained_muons['muon_efficiency'],
                        bins=np.linspace(0., 0.5, 51))
        axes[0, 1].set_xlabel('estimated telescope efficiency for muons')
        axes[0, 1].set_ylabel('number of rings')
        axes[1, 1].plot(contained_muons['ring_width'],
                        contained_muons['muon_efficiency'], 'x', alpha=0.5)
        axes[1, 1].set_ylim(0., 0.5)
        axes[1, 1].set_xlabel('ring width (deg)')
        axes[1, 1].set_ylabel('estimated telescope efficiency for muons')
        axes[0, 2].errorbar(subrun_list, mean_effi, yerr=sem_effi, fmt='o',
                            markersize=3.)

        axes[0, 2].set_xlabel('subrun index')
        axes[0, 2].set_ylabel('estimated telescope efficiency for muons')
        axes[0, 2].grid(linewidth=0.3, linestyle=':')
        axes[0, 2].set_ylim(0., 0.5)
        axes[1, 2].errorbar(subrun_list, mean_width, yerr=sem_width, fmt='o',
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
    x = np.arange(2+len(trig_types))
    # for better display, leave some space on the sides of the bars:
    y = np.append([0], np.append(num_triggers, [0]))
    labels = ['']+[str(i) for i in trig_types]+['']
    width = 0.3
    axes.bar(x, y, width)
    axes.set_xticks(x)
    axes.set_xticklabels(labels)
    axes.set_yscale('log')
    axes.set_xlabel(trigger_name)
    axes.set_ylabel('number of events')


def plot_mean_and_stddev(table, camgeom, columns, labels, pagesize, norm='lin'):

    # calculate pixel-wise mean and standard deviation for the whole run,
    # from the subrun-wise values:
    mean = np.sum(np.multiply(table.col(columns[0]),
                              table.col('num_events')[:, None]),
                  axis=0) / np.sum(table.col('num_events'))
    stddev = np.sqrt(np.sum(np.multiply(table.col(columns[1]) ** 2,
                                        table.col('num_events')[:, None]),
                            axis=0) / np.sum(table.col('num_events')))

    if np.isnan(mean).sum() > 0:
        print('Pixels with NaNs in '+columns[0]+':',
              np.array(camgeom.pix_id.tolist())[np.isnan(mean)])

    # plot mean and std dev (of e.g. pedestal charge or time), as camera
    # display, vs. pixel id, and as a histogram:
    fig, axes = plt.subplots(nrows=2, ncols=3,
                             figsize=pagesize)
    fig.suptitle(labels[2], fontsize='xx-large')
    fig.tight_layout(rect=[0, 0.03, 1, 0.98], pad=3.0, h_pad=3.0, w_pad=2.0)
    cam = CameraDisplay(camgeom, mean, ax=axes[0, 0], norm=norm,
                        title=labels[0])
    cam.add_colorbar(ax=axes[0, 0])
    cam.show()
    cam = CameraDisplay(camgeom, stddev, ax=axes[1, 0], norm=norm,
                        title=labels[1])
    cam.add_colorbar(ax=axes[1, 0])
    # line below needed to get the top and bottom camera displays of equal size:
    axes[1, 0].set_xlim((axes[0, 0].get_xlim()))
    cam.show()
    # plot mean vs. pixe_id and as histogram:
    axes[0, 1].plot(camgeom.pix_id, mean)
    axes[0, 1].set_xlabel('Pixel id')
    axes[0, 1].set_ylabel(labels[0])
    axes[0, 2].set_yscale('log')
    axes[0, 2].hist(mean[~np.isnan(mean)], bins=200)
    axes[0, 2].set_xlabel(labels[0])
    axes[0, 2].set_ylabel('Number of ixels')
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
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=pagesize)
    plt.text(0.5, 0.5, 'Sorry, no ' + tablename + ' to plot here!',
             fontsize=44, horizontalalignment='center',
             verticalalignment='center')
    axes.axis('off')
