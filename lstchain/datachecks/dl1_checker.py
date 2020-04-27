"""
Functions to check the contents ofLST DL1 files and associated muon ring files
"""

__all__ = [
    'check_dl1',
    'process_dl1_file',
    'plot_datacheck',
    'plot_trigger_types',
    'plot_mean_and_stddev',
    'DL1DataCheckContainer',
    'DL1DataCheckHistogramBins',
]

import h5py
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import tables
import warnings

from astropy import units as u
from astropy.table import Table
from ctapipe.coordinates import CameraFrame, EngineeringCameraFrame
from ctapipe.core import Container, Field
from ctapipe.instrument import CameraGeometry
from ctapipe.io import HDF5TableWriter
from ctapipe.visualization import CameraDisplay
from lstchain.io.io import dl1_params_lstcam_key
from matplotlib.backends.backend_pdf import PdfPages
from multiprocessing import Pool
from scipy.stats import poisson
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
                         (np.array(trig_tags['trigger_type'])==32).sum(),
                     'ucts_trigger_type':
                         (np.array(trig_tags['ucts_trigger_type'])==32).sum()}
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

    with HDF5TableWriter(out_filename) as writer:
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
    geom.to_table().write(out_filename,
                          path=f'/instrument/telescope/camera/LSTCam',
                          append=True, serialize_meta=True)

    # write out also which trigger tag has been used for finding pedestals:
    file = h5py.File(out_filename, mode='a')
    file.create_dataset('/dl1datacheck/used_trigger_tag', (1,), 'S32',
                        [trigger_source.encode('ascii')])
    file.close()

    # do the plots and save them to a pdf file:
    plot_datacheck(out_filename)

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

        # We do not convert the x,y, cog coordinates, because only in m can
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
        axes[1, 0].set_ylabel('number of events')
        axes[1, 0].set_xlabel('subrun index')
        axes[1, 0].set_yscale('log')
        axes[1, 0].legend(loc='best')

        for time_type in ['ucts_time', 'tib_time', 'dragon_time']:
            axes[1, 1].plot(table_cosmics.col('sampled_event_ids').flatten(),
                            table_cosmics.col(time_type).flatten(),
                            drawstyle='steps-mid', label=time_type)
        axes[1, 1].set_xlabel('event id')
        axes[1, 1].set_ylabel('timestamp')
        axes[1, 1].legend(loc='best')

        pdf.savefig()

        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=pagesize)
        fig.tight_layout(pad=3.0, h_pad=3.0, w_pad=2.0)
        hist = 'hist_delta_t'
        bins = hist_binning.col(hist)[0]
        axes[0, 0].hist(bins[:-1], bins,
                        weights=np.sum(table_cosmics.col(hist), axis=0),
                        histtype='step')
        axes[0, 0].set_xlabel('delta_t (ms) from Dragon timestamp')
        axes[0, 0].set_ylabel('events')
        axes[0, 0].set_yscale('log')
        pdf.savefig()


        if len(table_pedestals) == 0:
            fig, axes = plt.subplots(nrows=1, ncols=1, figsize=pagesize)
            plt.text(0.5, 0.5, 'Sorry, no pedestals to plot here!', fontsize=44,
                     horizontalalignment='center', verticalalignment='center')
            axes.axis('off')
        else:
            plot_mean_and_stddev(table_pedestals, engineering_geom,
                                 ['charge_mean', 'charge_stddev'],
                                 ['Pedestal mean charge (p.e.)',
                                  'Pedestal charge std dev (p.e.)',
                                  'PEDESTALS, pixel-wise charge info'], pagesize,
                                 norm='log')
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
            if (len(table) == 0):
                fig, axes = plt.subplots(nrows=1, ncols=1, figsize=pagesize)
                plt.text(0.5, 0.5, 'Sorry, no '+table.name+' to plot here!',
                         fontsize=44, horizontalalignment='center',
                         verticalalignment='center')
                axes.axis('off')
                pdf.savefig()
                continue

            # We asume here that 5 such thresholds are present in the
            # dl1datacheck file
            fig, axes = plt.subplots(nrows=2, ncols=3, figsize=pagesize)
            fig.suptitle(table.name.upper()+
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
            fraction = pix_events/norm
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
            for i in [1, 2, 4]: axes.flatten()[i].set_ylabel('')
            axes[1, 2].set_xscale('log')
            axes[1, 2].set_yscale('log')
            for x, y in zip(threshold, fraction):
                axes[1, 2].plot(x*np.ones(len(y)), y, 'o', fillstyle='none',
                                alpha=0.1)
                axes[1, 2].set_xlabel('pixel charge (p.e.)')
                axes[1, 2].set_ylabel('fraction of events with charge>x')
            pdf.savefig()

        # Some plots on pulse times:
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
            min = event_fraction[pix_inside & gt0].min()
            max = event_fraction[pix_inside & gt0].max()
            axes[i, 2].set_xscale('log')
            _, bins, _ = axes[i, 2].\
                hist(event_fraction[pix_inside & gt0],
                     bins=np.logspace(np.log10(min), np.log10(max), 201))
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
            axes[i, 2].plot(k[npixels>0], npixels[npixels>0],
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
                            weights=np.sum(table_cosmics.col(hist), axis=0)/
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
        histos = ['hist_skewness','hist_intercept', 'hist_tgrad_vs_length',
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
        axes[0, 1].set_ylabel('Intercept (fit t @ charge cog) (ns)')
        for j in [0,1]:
            axes[0, j].set_xscale('log')
            axes[0, j].set_xlabel('Intensity')
            axes[1, j].set_xlabel('Length (deg)')
            axes[1, j].set_ylabel('Time gradient (ns/deg)')
        axes[1, 0].set_title('Time gradient vs. Length')
        axes[1, 1].set_title('Time gradient vs. Length, intensity>200pe')
        pdf.savefig()

def plot_trigger_types(dchecktables, trigger_name, axes):
    """

    Parameters
    ----------
    dchecktables: array of python tables created with DL1DataCheckContainer
    containers (each row is one subrun). The plotted trigger type statistics
    will be the global ones, adding up the numbers from all the tables and
    all the rows in each table.

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
    tt = tt[tt[:, :, 1]>0]
    trig_types = np.unique(tt[:,0])
    num_triggers = np.array([(tt[:,1][tt[:,0]==trig]).sum()
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

    axes[0, 1].plot(camgeom.pix_id, mean)
    axes[0, 1].set_xlabel('Pixel id')
    axes[0, 1].set_ylabel(labels[0])
    axes[0, 2].set_yscale('log')
    axes[0, 2].hist(mean[~np.isnan(mean)], bins=200)
    axes[0, 2].set_xlabel(labels[0])
    axes[0, 2].set_ylabel('Pixels')
    # now the standard deviation:
    axes[1, 1].plot(camgeom.pix_id, stddev)
    axes[1, 1].set_xlabel('Pixel id')
    axes[1, 1].set_ylabel(labels[1])
    axes[1, 2].set_yscale('log')
    axes[1, 2].hist(stddev[~np.isnan(stddev)], bins=200)
    axes[1, 2].set_xlabel(labels[1])
    axes[1, 2].set_ylabel('Pixels')


class DL1DataCheckContainer(Container):
    """
    Container to store outcome of the DL1 data check
    """

    # scalar quantities:
    subrun_index = Field(-1, 'Subrun index')
    num_events = Field(-1, 'Total number of events')
    trigger_type = Field(None, 'Number of events per trigger type')
    ucts_trigger_type = Field(None, 'Number of events per ucts trigger type')

    # sampled quantities, stored every few events:
    sampled_event_ids = Field(None, 'sampled event ids')
    ucts_time = Field(None, 'ucts time', unit=u.s)
    tib_time = Field(None, 'tib_time', unit=u.s)
    dragon_time = Field(None, 'dragon_time', unit=u.s)

    # histograms; they store arrays of counts. Binning is defined in class
    # DL1DataCheckHistogramBins (see below)
    hist_delta_t = Field(None, 'Histogram of time difference between '
                               'consecutive events')
    hist_npixels = Field(None, 'Histogram of number of pixels in image')
    hist_nislands = Field(None, 'Histogram of number of islands in image')
    hist_intensity = Field(None, 'Histogram of image intensity')
    hist_dist0 = Field(None, 'Histogram of cog-camera center distance')
    hist_dist0_intensity_gt_200 = \
        Field(None, 'Histogram of cog-camera center distance')
    hist_width = Field(None, 'Histogram image width vs. intensity')
    hist_length = Field(None, 'Histogram image length vs. intensity')
    hist_skewness = Field(None, 'Histogram of image skewness')
    # the histogram hist_pixelchargespectrum shows the pixel charge
    # distribution, filled from all pixels:
    hist_pixelchargespectrum = Field(None, 'Histogram of pixel charges')

    hist_psi = Field(None, 'Histogram of image axis orientation')
    hist_intercept = Field(None, 'Histogram of fitted pulse time for charge '
                                 'c.o.g.')
    hist_tgrad_vs_length = Field(None, 'Histogram of time gradient vs. length')
    hist_tgrad_vs_length_intensity_gt_200 = \
        Field(None, 'Histogram of time gradient vs. length, intensity>200pe')

    # pixel-wise quantities, one entry per pixel. Used also for 2d
    # histogramming of cog position.
    cog_within_pixel = Field(None, 'Number of image cogs within pixel')
    cog_within_pixel_intensity_gt_200 = \
        Field(None, 'Number of image within pixel, intensity>200pe')
    charge_mean = Field(-1, 'Mean of pixel charge')
    charge_stddev = Field(-1, 'Standard deviation of pixel charge')
    time_mean = Field(-1, 'Mean of pulse time')
    time_stddev = Field(-1, 'Standard deviaton of pulse time')
    time_mean_above_030_pe = Field(-1, 'Mean of pulse time, >30 p.e. pulses')
    time_stddev_above_030_pe = Field(-1, 'Standard deviaton of pulse time, '
                                         '>30 p.e. pulses')
    # keep number of events above a few thresholds, like a low-res histogram
    # of pulse charges (2 points per decade in charge in p.e.)
    # This could be done in a cleaner way with a 2d hist charge vs. pixel (TBD)
    num_pulses_above_0010_pe = Field(None, 'Number of >10 p.e. pulses')
    num_pulses_above_0030_pe = Field(None, 'Number of >30 p.e. pulses')
    num_pulses_above_0100_pe = Field(None, 'Number of >100 p.e. pulses')
    num_pulses_above_0300_pe = Field(None, 'Number of >300 p.e. pulses')
    num_pulses_above_1000_pe = Field(None, 'Number of >1000 p.e. pulses')

    def fill_event_wise_info(self, subrun_index, table, mask, geom,
                             histogram_binnings):
        """
        Fills the container fields that depend on event-wise DL1 info

        Parameters
        ----------
        subrun_index
        table: DL1 parameters, event-wise pandas dataframe, "parameters" from
        DL1 files
        mask: defines which events in table should be considered
        geom: camera geometry (in standard frame, *not* engineering one)
        histogram_binnings: container of type DL1DataCheckHinstogramBins which
        defines the binning of the various histograms

        Returns
        -------
        None

        """
        self.subrun_index = subrun_index
        self.num_events = mask.sum()
        self.ucts_trigger_type = \
            self.count_trig_types(table['ucts_trigger_type'][mask])
        self.trigger_type = \
            self.count_trig_types(table['trigger_type'][mask])

        # number of time samples per subrun to be stored in the container:
        n_samples = 50
        n_jump = 1+int(self.num_events/n_samples)
        # keep some info every n-jump-th event:
        sampled_event_ids = np.array(table['event_id'][mask][0::n_jump])
        tib_time = np.array(table['tib_time'][mask][0::n_jump])
        ucts_time = np.array(table['ucts_time'][mask][0::n_jump])
        dragon_time = np.array(table['dragon_time'][mask][0::n_jump])
        # in case the resulting number of entries is <n_samples, we have to pad
        # the arrays, because hdf vector columns must have the same number of
        # elements in each row. We repeat the last value in the array
        padding = (0, n_samples-len(sampled_event_ids))
        self.sampled_event_ids = np.pad(sampled_event_ids, padding, mode='edge')
        self.tib_time = np.pad(tib_time, padding, mode='edge')
        self.ucts_time = np.pad(ucts_time, padding, mode='edge')
        self.dragon_time = np.pad(dragon_time, padding, mode='edge')


        delta_t = np.array(table['dragon_time'][mask][1:]) - \
                  np.array(table['dragon_time'][mask][:-1])
        counts, _, _, = plt.hist(delta_t*1.e3,
                                 bins=histogram_binnings.hist_delta_t)
        self.hist_delta_t = counts

        n_pixels = table['n_pixels'][mask]
        counts, _, _, = plt.hist(n_pixels,
                                bins=histogram_binnings.hist_npixels)
        self.hist_npixels = counts

        n_islands = table['n_islands'][mask]
        counts, _, _, = plt.hist(n_islands,
                                bins=histogram_binnings.hist_nislands)
        self.hist_nislands = counts

        intensity = table['intensity'][mask]
        counts, _, _ = plt.hist(intensity,
                                bins=histogram_binnings.hist_intensity)
        self.hist_intensity = counts

        dist0 = table['r'][mask]
        counts, _, _ = plt.hist(dist0, bins=histogram_binnings.hist_dist0)
        self.hist_dist0 = counts

        counts, _, _ = \
            plt.hist(dist0[intensity>200],
                     bins=histogram_binnings.hist_dist0_intensity_gt_200)
        self.hist_dist0_intensity_gt_200 = counts

        counts, _, _, _ = plt.hist2d(table['intensity'][mask],
                                     table['width'][mask],
                                     bins=histogram_binnings.hist_width)
        self.hist_width = counts

        counts, _, _, _ = plt.hist2d(table['intensity'][mask],
                                     table['length'][mask],
                                     bins=histogram_binnings.hist_length)
        self.hist_length = counts

        counts, _, _, _ = plt.hist2d(table['intensity'][mask],
                                     table['skewness'][mask],
                                     bins=histogram_binnings.hist_skewness)
        self.hist_skewness = counts

        intercept = table['intercept'][mask]
        counts, _, _, _ = \
            plt.hist2d(intensity, intercept,
                       bins=histogram_binnings.hist_intercept)
        self.hist_intercept = counts

        length = table['length'][mask]
        tgrad = np.abs(table['time_gradient'][mask])
        counts, _, _, _ = \
            plt.hist2d(length, tgrad,
                       bins=histogram_binnings.hist_tgrad_vs_length)
        self.hist_tgrad_vs_length = counts
        counts, _, _, _ = \
            plt.hist2d(length[intensity>200], tgrad[intensity>200],
                       bins=histogram_binnings.\
                       hist_tgrad_vs_length_intensity_gt_200)
        self.hist_tgrad_vs_length_intensity_gt_200 = counts

        x = table['x'][mask]
        y = table['y'][mask]
        # event-wise, id of camera pixel which contains the image's cog:
        cog_pixid = geom.position_to_pix_index(np.array(x)*u.m,
                                               np.array(y)*u.m)
        self.cog_within_pixel = np.zeros(geom.n_pixels)
        for pix in cog_pixid:
            self.cog_within_pixel[pix] += 1
        self.cog_within_pixel_intensity_gt_200 = np.zeros(geom.n_pixels)
        # now the same for relatively bright images (intensity > 200 p.e.)
        select = intensity > 200
        for pix in cog_pixid[select]:
            self.cog_within_pixel_intensity_gt_200[pix] += 1

    def fill_pixel_wise_info(self, table, mask, histogram_binnings):
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
        time = table.col('pulse_time')[mask]
        self.charge_mean = charge.mean(axis=0)
        self.charge_stddev = charge.std(axis=0)

        # as of ctapipe 0.7.0, pulse times can take absurd values for pixels
        # containing very little signal. For time plots we require at least 1
        # p.e. We also exclude NaNs
        charge_t = charge.transpose()
        time_t = time.transpose()
        # each row in the transposed matrices has all events for one pixel

        healthy_entries = np.array([t[~np.isnan(t) & (c > 1)]
                                    for t, c in zip(time_t, charge_t)])
        self.time_mean = np.array([h.mean() if len(h) > 0 else np.nan
                                   for h in healthy_entries])
        self.time_stddev = np.array([h.std() if len(h) > 0 else np.nan
                                     for h in healthy_entries])

        healthy_entries = np.array([t[~np.isnan(t) & (c > 30)]
                                    for t, c in zip(time_t, charge_t)])
        self.time_mean_above_030_pe = np.array([h.mean() if len(h) > 0
                                                else np.nan
                                                for h in healthy_entries])
        self.time_stddev_above_030_pe = np.array([h.std() if len(h) > 0
                                                  else np.nan
                                                  for h in healthy_entries])

        # count, for each pixel, the number of entries with charge>x pe:
        self.num_pulses_above_0010_pe = np.sum(charge > 10, axis=0)
        self.num_pulses_above_0030_pe = np.sum(charge > 30, axis=0)
        self.num_pulses_above_0100_pe = np.sum(charge > 100, axis=0)
        self.num_pulses_above_0300_pe = np.sum(charge > 300, axis=0)
        self.num_pulses_above_1000_pe = np.sum(charge > 1000, axis=0)

        counts, _, _ = plt.hist(charge[charge>0].flatten(),
                                bins=histogram_binnings.hist_pixelchargespectrum)
        self.hist_pixelchargespectrum = counts


    def count_trig_types(self, array):
        """

        Parameters
        ----------
        array: ndarray of event-wise trigger types

        Returns
        -------
        an ndarray of shape (10, 2) [i, j] means we found j events of type i

        """
        ucts_trig_types, counts = np.unique(array, return_counts=True)
        # write the different trigger types, then the number of events of
        # each type. Pad to 10 entries (more than enough for trigger types):
        ucts_trig_types = \
            np.append(ucts_trig_types, (10-len(ucts_trig_types))*[0])
        counts = np.append(counts, (10 - len(counts)) * [0])
        return np.array([[t, n] for t, n in zip(ucts_trig_types, counts)])

class DL1DataCheckHistogramBins(Container):

    # delta_t between consecutive events (ms)
    hist_delta_t = Field(np.linspace(-1.e-2, 2., 200),
                         'hist_delta_t binning')
    # pixel charge and image intensity (units: p.e):
    hist_pixelchargespectrum = Field(np.logspace(-1., 4.7, 121),
                                     'hist_pixelchargespectrum binning')
    hist_intensity = Field(np.logspace(1., 6., 101), 'hist_intensity binning')

    # dist0, width and length (units: degrees):
    hist_dist0 = Field(np.linspace(0., 2.5, 50), 'hist_dist0 binning')
    hist_dist0_intensity_gt_200 = Field(np.linspace(0., 2.5, 50),
                                        'hist_dist0_intensity_gt_200 binning')
    hist_psi = Field(np.linspace(-100., 100., 101), 'hist_psi binning')
    hist_psi_intensity_gt_200 = Field(np.linspace(-100., 100., 101),
                                      'hist_psi_intensity_gt_200 binning')

    hist_nislands = Field(np.linspace(-0.5, 29.5, 31), 'hist_nislands binning')
    hist_npixels = Field(np.linspace(0.5,2000.5,400), 'hist_npixels binning')

    # 2d histograms
    # width and length vs. image intensity:
    hist_width = Field(np.array([np.logspace(0.7, 5.7, 101),
                                 np.linspace(0., 1.2, 101)]),
                       'hist_width binning')
    hist_length = Field(np.array([np.logspace(0.7, 5.7, 101),
                                  np.linspace(0., 2., 101)]),
                       'hist_length binning')
    hist_skewness = Field(np.array([np.logspace(0.7, 5.7, 101),
                                    np.linspace(-4., 4., 101)]),
                          'hist_skewness binning')
    # time gradient vs. length:
    hist_tgrad_vs_length = Field(np.array([np.linspace(0., 2.0, 101),
                                           np.linspace(0., 200., 101)]),
                                 'hist_tgrad_vs_length binning')
    hist_tgrad_vs_length_intensity_gt_200 =\
        Field(np.array([np.linspace(0., 2.0, 101), np.linspace(0., 50., 101)]),
              'hist_tgrad_vs_length_intensity_gt_200 binning')
    # time intercept (image time @ charge c.o.g.) vs. image intensity:
    hist_intercept = Field(np.array([np.logspace(0.7, 5.7, 101),
                                     np.linspace(-30., 40., 101)]),
                           'hist_intercept binning')
