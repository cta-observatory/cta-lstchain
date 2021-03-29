#!/usr/bin/env python

"""
This script has to be run in a directory containing LST1 DL1 datacheck files, of
those containing inf for a whole run, with name pattern
datacheck_dl1_LST-1.Run?????.h5

The corresponding muon*fits files (which are subrun-wise) have to be present
in the same directory.

The output is the file longterm_dl1_check.h5 file, which contains tables with
some run-wise summary values for plotting long-term evolution of the DL1 data.
It also produces an interactive web page, longterm_dl1_check.html with plots
showing the evolution of many such values.

"""

import copy
import glob
from pathlib import Path

import numpy as np
import pandas as pd
import tables
from astropy.table import Table
from bokeh.io import output_file as bokeh_output_file
from bokeh.io import show
from bokeh.layouts import gridplot, column
from bokeh.models import Div, ColumnDataSource, Whisker, HoverTool, Range1d
from bokeh.models.widgets import Tabs, Panel
from bokeh.plotting import figure
from ctapipe.coordinates import EngineeringCameraFrame
from ctapipe.instrument import SubarrayDescription

from lstchain.visualization.bokeh import show_camera


def main():

    output_file_name = 'longterm_dl1_check.h5'
    files = glob.glob('datacheck_dl1_LST-1.Run?????.h5')
    files.sort()

    # hardcoded for now, to be eventually read from data:
    numpixels = 1855

    # subrun-wise tables: cosmics, pedestals, flatfield. One dictionary per
    # each. Note that the cosmics table contains also muon ring information!

    cosmics = {'runnumber': [],
               'subrun': [],
               'time': [],
               'elapsed_time': [],
               'events': [],
               'azimuth': [],
               'altitude': []}

    pedestals = copy.deepcopy(cosmics)
    flatfield = copy.deepcopy(cosmics)

    # add table-specific fields:

    cosmics['num_contained_mu_rings'] = []
    cosmics['mu_effi_mean'] = []
    cosmics['mu_effi_stddev'] = []
    cosmics['mu_width_mean'] = []
    cosmics['mu_width_stddev'] = []
    cosmics['mu_radius_mean'] = []
    cosmics['mu_radius_stddev'] = []
    cosmics['mu_intensity_mean'] = []
    cosmics['mu_hg_peak_sample'] = []
    cosmics['mu_hg_peak_sample_stddev'] = []
    cosmics['mu_lg_peak_sample'] = []
    cosmics['mu_lg_peak_sample_stddev'] = []
    cosmics['fraction_pulses_above10'] = [] # fraction of >10 pe pulses
    cosmics['fraction_pulses_above30'] = [] # fraction of >30 pe pulses

    pedestals['fraction_pulses_above10'] = [] # fraction of >10 pe pulses
    pedestals['fraction_pulses_above30'] = [] # fraction of >30 pe pulses
    pedestals['charge_mean'] = []
    pedestals['charge_stddev'] = []

    flatfield['charge_mean'] = []
    flatfield['charge_stddev'] = []
    flatfield['rel_time_mean'] = []
    flatfield['rel_time_stddev'] = []

    # now another dictionary for a run-wise table, with no pixel-wise info:

    runsummary = {'runnumber': [],
                  'time': [],
                  'elapsed_time': [],
                  'min_altitude': [],
                  'mean_altitude': [],
                  'max_altitude': [],
                   # currently (as of lstchain 0.5.3) event numbers are post-cleaning!:
                  'num_cosmics': [],
                  'num_pedestals': [],
                  'num_flatfield': [],
                  'num_pedestals_after_cleaning': [],
                  'num_contained_mu_rings': [],
                  'ff_charge_mean': [],   # camera average of mean pix FF charge
                  'ff_charge_mean_err': [], # uncertainty of the above
                  'ff_charge_stddev': [], # camera average
                  'ff_time_mean': [], # camera average of mean FF time
                  'ff_time_mean_err': [], # uncertainty of the above
                  'ff_time_stddev': [], # camera average
                  'ff_rel_time_stddev': [], # camera-averaged std dev of pixel t
                  # w.r.t. average of rest of pixels in camera (~ t-resolution)
                  'ped_charge_mean': [], # camera average of mean pix ped charge
                  'ped_charge_mean_err':[],  # uncertainty of the above
                  'ped_charge_stddev': [],  # camera average
                  'ped_fraction_pulses_above10': [], # in whole camera
                  'ped_fraction_pulses_above30': [], # in whole camera
                  'cosmics_fraction_pulses_above10': [], # in whole camera
                  'cosmics_fraction_pulses_above30': [], # in whole camera
                  'mu_effi_mean': [],
                  'mu_effi_stddev': [],
                  'mu_width_mean': [],
                  'mu_width_stddev': [],
                  'mu_hg_peak_sample_mean': [],
                  'mu_hg_peak_sample_stddev': [],
                  'mu_lg_peak_sample_mean': [],
                  'mu_lg_peak_sample_stddev': [],
                  'mu_intensity_mean': []}

    # and another one for pixel-wise run averages:
    pixwise_runsummary = {'ff_pix_charge_mean': [],
                          'ff_pix_charge_stddev': [],
                          'ff_pix_rel_time_mean': [],
                          'ff_pix_rel_time_stddev': [],
                          'ped_pix_charge_mean': [],
                          'ped_pix_charge_stddev': [],
                          'ped_pix_fraction_pulses_above10': [],
                          'ped_pix_fraction_pulses_above30': [],
                          'cosmics_pix_fraction_pulses_above10': [],
                          'cosmics_pix_fraction_pulses_above30': []}

    # Needed for the table description for writing it out to the hdf5 file. Because
    # of the vector columns we cannot write this out using pandas:
    class pixwise_info(tables.IsDescription):
        runnumber = tables.Int32Col()
        time = tables.Float64Col()
        ff_pix_charge_mean = tables.Float32Col(shape=(numpixels))
        ff_pix_charge_stddev = tables.Float32Col(shape=(numpixels))
        ff_pix_rel_time_mean = tables.Float32Col(shape=(numpixels))
        ff_pix_rel_time_stddev = tables.Float32Col(shape=(numpixels))
        ped_pix_charge_mean = tables.Float32Col(shape=(numpixels))
        ped_pix_charge_stddev = tables.Float32Col(shape=(numpixels))
        ped_pix_fraction_pulses_above10 = tables.Float32Col(shape=(numpixels))
        ped_pix_fraction_pulses_above30 = tables.Float32Col(shape=(numpixels))
        cosmics_pix_fraction_pulses_above10 = tables.Float32Col(shape=(numpixels))
        cosmics_pix_fraction_pulses_above30 = tables.Float32Col(shape=(numpixels))

    dicts = [cosmics, pedestals, flatfield]

    for file in files:

        try:
            a = tables.open_file(file)
        except FileNotFoundError:
            print('Could not read file', file, '- skipping...')
            continue

        print(file)
        runnumber = int(file[file.find('.Run')+4:file.find('.Run')+9])

        datatables = []
        for name in ['/dl1datacheck/cosmics',
                     '/dl1datacheck/pedestals',
                     '/dl1datacheck/flatfield']:
            try:
                node = a.get_node(name)
            except Exception:
                print('   Table', name, 'is missing!')
                datatables.append(None)
                continue

            datatables.append(node)

        subruns = None

        # fill data which are common to all tables:
        for table, d in zip(datatables, dicts):
            if table is None:
                continue
            d['runnumber'].extend(len(table)*[runnumber])
            d['subrun'].extend(table.col('subrun_index'))
            d['elapsed_time'].extend(table.col('elapsed_time'))
            d['events'].extend(table.col('num_events'))
            d['time'].extend(table.col('dragon_time').mean(axis=1))
            d['azimuth'].extend(table.col('mean_az_tel'))
            d['altitude'].extend(table.col('mean_alt_tel'))

        # now fill table-specific quantities. In some cases they are
        # pixel-averaged values:

        if datatables[0] is not None:
            table = a.root.dl1datacheck.cosmics
            cosmics['fraction_pulses_above10'].extend(
                    table.col('num_pulses_above_0010_pe').mean(axis=1) /
                    table.col('num_events'))
            cosmics['fraction_pulses_above30'].extend(
                    table.col('num_pulses_above_0030_pe').mean(axis=1) /
                    table.col('num_events'))

        if datatables[1] is not None:
            table = a.root.dl1datacheck.pedestals
            pedestals['fraction_pulses_above10'].extend(
                    table.col('num_pulses_above_0010_pe').mean(axis=1) /
                    table.col('num_events'))
            pedestals['fraction_pulses_above30'].extend(
                    table.col('num_pulses_above_0030_pe').mean(axis=1) /
                    table.col('num_events'))
            pedestals['charge_mean'].extend(
                    table.col('charge_mean').mean(axis=1))
            pedestals['charge_stddev'].extend(
                    table.col('charge_stddev').mean(axis=1))

        if datatables[2] is not None:
            table = a.root.dl1datacheck.flatfield
            flatfield['charge_mean'].extend(
                    table.col('charge_mean').mean(axis=1))
            flatfield['charge_stddev'].extend(
                    table.col('charge_stddev').mean(axis=1))
            flatfield['rel_time_mean'].extend(
                    table.col('relative_time_mean').mean(axis=1))
            flatfield['rel_time_stddev'].extend(
                    table.col('relative_time_stddev').mean(axis=1))


        table = a.root.dl1datacheck.cosmics

        # needed later for the muons:
        subruns = table.col('subrun_index')

        # now fill the run-wise table:
        runsummary['runnumber'].extend([runnumber])
        runsummary['time'].extend([table.col('dragon_time').mean()])
        runsummary['elapsed_time'].extend([table.col('elapsed_time').sum()])
        runsummary['min_altitude'].extend([table.col('mean_alt_tel').min()])
        runsummary['mean_altitude'].extend([table.col('mean_alt_tel').mean()])
        runsummary['max_altitude'].extend([table.col('mean_alt_tel').max()])
        runsummary['num_cosmics'].extend([table.col('num_events').sum()])
        runsummary['cosmics_fraction_pulses_above10'].extend(
                [(table.col('num_pulses_above_0010_pe').mean(axis=1)).sum() /
                 runsummary['num_cosmics'][-1]])
        runsummary['cosmics_fraction_pulses_above30'].extend(
                [(table.col('num_pulses_above_0030_pe').mean(axis=1)).sum() /
                 runsummary['num_cosmics'][-1]])
        pixwise_runsummary['cosmics_pix_fraction_pulses_above10'].extend(
                [table.col('num_pulses_above_0010_pe').sum(axis=0) /
                 runsummary['num_cosmics'][-1]])
        pixwise_runsummary['cosmics_pix_fraction_pulses_above30'].extend(
                [table.col('num_pulses_above_0030_pe').sum(axis=0) /
                 runsummary['num_cosmics'][-1]])

        if datatables[1] is not None:
            table = a.root.dl1datacheck.pedestals
            nevents = table.col('num_events') # events per subrun
            events_in_run = nevents.sum()

            runsummary['num_pedestals'].extend([table.col('num_events').sum()])
            runsummary['num_pedestals_after_cleaning'].extend([table.col(
                    'num_cleaned_events').sum()])

            runsummary['ped_fraction_pulses_above10'].extend([(table.col('num_pulses_above_0010_pe').mean(axis=1)).sum()/
                                                             runsummary['num_pedestals'][-1]])
            runsummary['ped_fraction_pulses_above30'].extend([(table.col('num_pulses_above_0030_pe').mean(axis=1)).sum()/
                                                              runsummary['num_pedestals'][-1]])

            # Mean pedestal charge through a run, for each pixel:
            charge_mean = np.sum(table.col('charge_mean')*nevents[:, None],
                                 axis=0) / events_in_run
            # Now store the pixel-averaged mean pedestal charge:
            runsummary['ped_charge_mean'].extend([np.nanmean(charge_mean)])
            npixels=len(charge_mean)
            runsummary['ped_charge_mean_err'].extend([np.nanstd(charge_mean) /
                                                     np.sqrt(npixels)])
            # Pedestal charge std dev through a run, for each pixel:
            charge_stddev =\
                np.sqrt(np.sum((table.col('charge_stddev')**2)*nevents[:, None],
                               axis=0) / events_in_run)
            # Store the pixel-averaged pedestal charge std dev:
            runsummary['ped_charge_stddev'].extend([np.nanmean(charge_stddev)])

            pixwise_runsummary['ped_pix_fraction_pulses_above10'].extend([table.col('num_pulses_above_0010_pe').sum(axis=0)/
                                                                          runsummary['num_pedestals'][-1]])
            pixwise_runsummary['ped_pix_fraction_pulses_above30'].extend([table.col('num_pulses_above_0030_pe').sum(axis=0)/
                                                                          runsummary['num_pedestals'][-1]])
            pixwise_runsummary['ped_pix_charge_mean'].extend(
                    [table.col('charge_mean').mean(axis=0)])
            pixwise_runsummary['ped_pix_charge_stddev'].extend(
                    [table.col('charge_stddev').mean(axis=0)])

        else:
            runsummary['num_pedestals'].extend([np.nan])
            runsummary['num_pedestals_after_cleaning'].extend([np.nan])
            runsummary['ped_fraction_pulses_above10'].extend([np.nan])
            runsummary['ped_fraction_pulses_above30'].extend([np.nan])
            runsummary['ped_charge_mean'].extend([np.nan])
            runsummary['ped_charge_mean_err'].extend([np.nan])
            runsummary['ped_charge_stddev'].extend([np.nan])
            pixwise_runsummary['ped_pix_fraction_pulses_above10'].extend([numpixels*[np.nan]])
            pixwise_runsummary['ped_pix_fraction_pulses_above30'].extend([numpixels*[np.nan]])
            pixwise_runsummary['ped_pix_charge_mean'].extend([numpixels*[np.nan]])
            pixwise_runsummary['ped_pix_charge_stddev'].extend([numpixels*[np.nan]])


        if datatables[2] is not None:
            table = a.root.dl1datacheck.flatfield
            nevents = table.col('num_events') # events per subrun
            events_in_run = nevents.sum()
            runsummary['num_flatfield'].extend([events_in_run])

            # Mean flat field charge through a run, for each pixel:
            charge_mean = np.sum(table.col('charge_mean') * nevents[:, None],
                                 axis=0) / events_in_run
            # Mean flat field time through a run, for each pixel:
            time_mean = np.sum(table.col('time_mean') * nevents[:, None],
                                 axis=0) / events_in_run

            # Now store the pixel-averaged mean charge:
            runsummary['ff_charge_mean'].extend([np.nanmean(charge_mean)])
            npixels=len(charge_mean)
            runsummary['ff_charge_mean_err'].extend([np.nanstd(charge_mean) /
                                                     np.sqrt(npixels)])
            # FF charge std dev through a run, for each pixel:
            charge_stddev =\
                np.sqrt(np.sum((table.col('charge_stddev')**2)*nevents[:, None],
                               axis=0) / events_in_run)
            # Store the pixel-averaged FF charge std dev:
            runsummary['ff_charge_stddev'].extend([np.nanmean(charge_stddev)])

            # Pixel-averaged mean time:
            runsummary['ff_time_mean'].extend([np.nanmean(time_mean)])
            runsummary['ff_time_mean_err'].extend([np.nanstd(time_mean) /
                                                     np.sqrt(npixels)])
            # FF time std dev through a run, for each pixel:
            time_stddev =\
                np.sqrt(np.sum((table.col('time_stddev')**2)*nevents[:, None],
                               axis=0) / events_in_run)
            # Store the pixel-averaged FF time std dev:
            runsummary['ff_time_stddev'].extend([np.nanmean(time_stddev)])

            rel_time_stddev =\
                np.sqrt(np.sum((table.col('relative_time_stddev')**2) *
                               nevents[:, None], axis=0) / events_in_run)
            runsummary['ff_rel_time_stddev'].\
                extend([np.nanmean(rel_time_stddev)])

            pixwise_runsummary['ff_pix_charge_mean'].extend(
                    [table.col('charge_mean').mean(axis=0)])
            pixwise_runsummary['ff_pix_charge_stddev'].extend(
                    [table.col('charge_stddev').mean(axis=0)])
            pixwise_runsummary['ff_pix_rel_time_mean'].extend(
                    [table.col('relative_time_mean').mean(axis=0)])
            pixwise_runsummary['ff_pix_rel_time_stddev'].extend(
                    [table.col('relative_time_stddev').mean(axis=0)])
        else:
            runsummary['num_flatfield'].extend([np.nan])
            runsummary['ff_charge_mean'].extend([np.nan])
            runsummary['ff_charge_mean_err'].extend([np.nan])
            runsummary['ff_charge_stddev'].extend([np.nan])
            runsummary['ff_time_mean'].extend([np.nan])
            runsummary['ff_time_mean_err'].extend([np.nan])
            runsummary['ff_time_stddev'].extend([np.nan])
            runsummary['ff_rel_time_stddev'].extend([np.nan])
            pixwise_runsummary['ff_pix_charge_mean'].extend([numpixels*[np.nan]])
            pixwise_runsummary['ff_pix_charge_stddev'].extend([numpixels*[np.nan]])
            pixwise_runsummary['ff_pix_rel_time_mean'].extend(
                    [numpixels * [np.nan]])
            pixwise_runsummary['ff_pix_rel_time_stddev'].extend(
                    [numpixels * [np.nan]])

        a.close()

        # Now process the muon files (one per subrun, containing one entry per ring):

        empty_files = 0

        contained_mu_wholerun = None
        num_contained_mu_rings_in_run = 0

        for subrun in subruns:
            mufile = 'muons_LST-1.Run{0:05d}.{1:04d}.fits'.format(runnumber, subrun)

            dat = None
            try:
                dat = Table.read(mufile, format='fits')
            except Exception:
                print('   File', mufile, 'not found - going on')
            if dat is None or len(dat) == 0:
                empty_files += 1
                cosmics['num_contained_mu_rings'].extend([0])
                cosmics['mu_effi_mean'].extend([np.nan])
                cosmics['mu_effi_stddev'].extend([np.nan])
                cosmics['mu_width_mean'].extend([np.nan])
                cosmics['mu_width_stddev'].extend([np.nan])
                cosmics['mu_radius_mean'].extend([np.nan])
                cosmics['mu_radius_stddev'].extend([np.nan])
                cosmics['mu_intensity_mean'].extend([np.nan])
                cosmics['mu_hg_peak_sample'].extend([np.nan])
                cosmics['mu_hg_peak_sample_stddev'].extend([np.nan])
                cosmics['mu_lg_peak_sample'].extend([np.nan])
                cosmics['mu_lg_peak_sample_stddev'].extend([np.nan])
                continue

            df_muons = dat.to_pandas()

            # contained and clean muon rings:
            contained_mu = df_muons[(df_muons['ring_containment']>0.99)&(df_muons['size_outside']<1.)]

            num_contained_mu_rings_in_run += len(contained_mu)

            cosmics['num_contained_mu_rings'].extend([len(contained_mu)])
            cosmics['mu_effi_mean'].extend([contained_mu['muon_efficiency'].mean()])
            cosmics['mu_effi_stddev'].extend([contained_mu['muon_efficiency'].std()])
            cosmics['mu_width_mean'].extend([contained_mu['ring_width'].mean()])
            cosmics['mu_width_stddev'].extend([contained_mu['ring_width'].std()])
            cosmics['mu_radius_mean'].extend([contained_mu['ring_radius'].mean()])
            cosmics['mu_radius_stddev'].extend([contained_mu['ring_radius'].std()])
            cosmics['mu_intensity_mean'].extend([contained_mu['ring_size'].mean()])
            cosmics['mu_hg_peak_sample'].extend([contained_mu['hg_peak_sample'].mean()])
            cosmics['mu_hg_peak_sample_stddev'].extend([contained_mu['hg_peak_sample'].std()])
            cosmics['mu_lg_peak_sample'].extend([contained_mu['lg_peak_sample'].mean()])
            cosmics['mu_lg_peak_sample_stddev'].extend([contained_mu['lg_peak_sample'].std()])

            if contained_mu_wholerun is None:
                contained_mu_wholerun = df_muons
            else:
                contained_mu_wholerun = pd.concat([contained_mu_wholerun, df_muons], ignore_index=True)


        if empty_files > 0:
            print('   Run {0:d} had {1:d} subruns with no valid muon rings!'.format(runnumber, empty_files))

        # fill the runsummary muons part:
        if contained_mu_wholerun is not None:
            runsummary['num_contained_mu_rings'].extend(
                    [num_contained_mu_rings_in_run])
            # The values below are mean and std dev for all contained muon
            # rings in a run:
            runsummary['mu_effi_mean'].extend([contained_mu_wholerun['muon_efficiency'].mean()])
            runsummary['mu_effi_stddev'].extend([contained_mu_wholerun['muon_efficiency'].std()])
            runsummary['mu_width_mean'].extend([contained_mu_wholerun['ring_width'].mean()])
            runsummary['mu_width_stddev'].extend([contained_mu_wholerun['ring_width'].std()])
            runsummary['mu_intensity_mean'].extend([contained_mu_wholerun['ring_size'].mean()])
            runsummary['mu_hg_peak_sample_mean'].\
                extend([contained_mu_wholerun['hg_peak_sample'].mean()])
            runsummary['mu_hg_peak_sample_stddev'].\
                extend([contained_mu_wholerun['hg_peak_sample'].std()])
            runsummary['mu_lg_peak_sample_mean'].\
                extend([contained_mu_wholerun['lg_peak_sample'].mean()])
            runsummary['mu_lg_peak_sample_stddev'].\
                extend([contained_mu_wholerun['lg_peak_sample'].std()])
        else:
            runsummary['num_contained_mu_rings'].extend([np.nan])
            runsummary['mu_effi_mean'].extend([np.nan])
            runsummary['mu_effi_stddev'].extend([np.nan])
            runsummary['mu_width_mean'].extend([np.nan])
            runsummary['mu_width_stddev'].extend([np.nan])
            runsummary['mu_intensity_mean'].extend([np.nan])
            runsummary['mu_hg_peak_sample_mean'].extend([np.nan])
            runsummary['mu_hg_peak_sample_stddev'].extend([np.nan])
            runsummary['mu_lg_peak_sample_mean'].extend([np.nan])
            runsummary['mu_lg_peak_sample_stddev'].extend([np.nan])

    pd.DataFrame(runsummary).to_hdf(output_file_name, key='runsummary', mode='w')

    # Now write the pixel-wise run summary info:
    h5file = tables.open_file(output_file_name, mode="a")
    table = h5file.create_table('/', 'pixwise_runsummary', pixwise_info)
    row = table.row
    for i in range(len(pixwise_runsummary['ff_pix_charge_mean'])):
        # we add run number and time info also to this pixwise table:
        row['runnumber'] = runsummary['runnumber'][i]
        row['time'] = runsummary['time'][i]
        for key in pixwise_runsummary:
            row[key] = pixwise_runsummary[key][i]
        row.append()
    table.flush()
    h5file.close()

    # Finally the tables with info by event type:
    for d, name in zip(dicts, ['cosmics', 'pedestals', 'flatfield']):
        pd.DataFrame(d).to_hdf(output_file_name, key=name, mode='a')

    # We write out the camera geometry information, assuming it is the same
    # for all files (hence we take it from the first one):
    subarray_info = SubarrayDescription.from_hdf(files[0])
    subarray_info.to_hdf(output_file_name)

    plot(output_file_name)


def plot(filename='longterm_dl1_check.h5', tel_id=1):

    # First read in the camera geometry:
    subarray_info = SubarrayDescription.from_hdf(filename)
    camgeom = subarray_info.tel[tel_id].camera.geometry
    engineering_geom = camgeom.transform_to(EngineeringCameraFrame())

    file = tables.open_file('longterm_dl1_check.h5')

    bokeh_output_file(Path(filename).with_suffix('.html'),
                      title='LST1 long-term DL1 data check')

    run_titles = []
    for i, run in enumerate(file.root.pixwise_runsummary.col('runnumber')):
        date = pd.to_datetime(file.root.pixwise_runsummary.col('time')[i],
                              origin='unix', unit='s')
        run_titles.append('Run {0:05d}, {date}'.\
                          format(run,
                                 date = date.strftime("%b %d %Y %H:%M:%S")))

    runsummary = pd.read_hdf(filename, 'runsummary')
    page0 = Panel()
    fig_ped_rates = show_graph(x=pd.to_datetime(runsummary['time'],
                                                origin='unix', unit='s'),
                               y=runsummary['num_pedestals'] /
                                 runsummary['elapsed_time'],
                               xlabel='date',
                               ylabel='Interleaved pedestals rate',
                               ey=np.sqrt(runsummary['num_pedestals']) /
                                  runsummary['elapsed_time'],
                               xtype='datetime', ytype='linear',
                               point_labels=run_titles)
    fig_ff_rates = show_graph(x=pd.to_datetime(runsummary['time'],
                                               origin='unix', unit='s'),
                               y=runsummary['num_flatfield'] /
                                 runsummary['elapsed_time'],
                               xlabel='date',
                               ylabel='Interleaved flat field rate',
                               ey=np.sqrt(runsummary['num_flatfield']) /
                                  runsummary['elapsed_time'],
                               xtype='datetime', ytype='linear',
                               point_labels=run_titles)
    fig_cosmic_rates = show_graph(x=pd.to_datetime(runsummary['time'],
                                                   origin='unix', unit='s'),
                                  y=runsummary['num_cosmics'] /
                                  runsummary['elapsed_time'],
                                  xlabel='date',
                                  ylabel='Cosmics rate',
                                  ey=np.sqrt(runsummary['num_cosmics']) /
                                     runsummary['elapsed_time'],
                                  xtype='datetime', ytype='linear',
                                  point_labels=run_titles)
    fig_muring_rates = show_graph(x=pd.to_datetime(runsummary['time'],
                                                   origin='unix', unit='s'),
                                  y=runsummary['num_contained_mu_rings'] /
                                  runsummary['elapsed_time'],
                                  xlabel='date',
                                  ylabel='Contained mu-rings rate',
                                  ey=np.sqrt(runsummary[
                                                 'num_contained_mu_rings']) /
                                                 runsummary['elapsed_time'],
                                  xtype='datetime', ytype='linear',
                                  point_labels=run_titles)

    pad_width = 550
    pad_height = 350
    row1 = [fig_ped_rates, fig_ff_rates]
    row2 = [fig_cosmic_rates, fig_muring_rates]
    grid0 = gridplot([row1, row2], sizing_mode=None, plot_width=pad_width,
                     plot_height=pad_height)
    page0.child = grid0
    page0.title = 'Event rates'

    page0b = Panel()
    altmin = np.rad2deg(runsummary['min_altitude'])
    altmean = np.rad2deg(runsummary['mean_altitude'])
    altmax = np.rad2deg(runsummary['max_altitude'])
    fig_altitude = show_graph(x=pd.to_datetime(runsummary['time'],
                                               origin='unix', unit='s'),
                              y=altmean,
                              xlabel='date',
                              ylabel='Telescope altitude (mean, min, max)',
                              eylow=altmean-altmin, eyhigh=altmax-altmean,
                              xtype='datetime', ytype='linear',
                              point_labels=run_titles)
    fig_altitude.y_range = Range1d(altmin.min()*0.95, altmax.max()*1.05)
    row1 = [fig_altitude]
    grid0b = gridplot([row1], sizing_mode=None, plot_width=pad_width,
                      plot_height=pad_height)
    page0b.child = grid0b
    page0b.title = 'Pointing'

    page1 = Panel()
    pad_width = 350
    pad_height = 370
    mean = []
    stddev = []
    for item in file.root.pixwise_runsummary.col('ped_pix_charge_mean'):
        mean.append(item)
    for item in file.root.pixwise_runsummary.col('ped_pix_charge_stddev'):
        stddev.append(item)
    row1 = show_camera(np.array(mean), engineering_geom, pad_width,
                       pad_height, 'Pedestals mean charge',
                       run_titles)
    row2 = show_camera(np.array(stddev), engineering_geom, pad_width,
                       pad_height, 'Pedestals charge std dev',
                       run_titles)
    grid1 = gridplot([row1, row2], sizing_mode=None, plot_width=pad_width,
                     plot_height=pad_height)
    page1.child = grid1
    page1.title = 'Interleaved pedestals'

    page2 = Panel()
    mean = []
    stddev = []
    for item in file.root.pixwise_runsummary.col('ff_pix_charge_mean'):
        mean.append(item)
    for item in file.root.pixwise_runsummary.col('ff_pix_charge_stddev'):
        stddev.append(item)
    row1 = show_camera(np.array(mean), engineering_geom, pad_width,
                       pad_height, 'Flat-Field mean charge (pe)', run_titles)
    row2 = show_camera(np.array(stddev), engineering_geom, pad_width,
                       pad_height, 'Flat-Field charge std dev (pe)', run_titles)
    grid2 = gridplot([row1, row2], sizing_mode=None, plot_width=pad_width,
                     plot_height=pad_height)
    page2.child = grid2
    page2.title = 'Interleaved flat field, charge'

    page3 = Panel()
    mean = []
    stddev = []
    for item in file.root.pixwise_runsummary.col('ff_pix_rel_time_mean'):
        mean.append(item)
    for item in file.root.pixwise_runsummary.col('ff_pix_rel_time_stddev'):
        stddev.append(item)
    row1 = show_camera(np.array(mean), engineering_geom, pad_width,
                       pad_height, 'Flat-Field mean relative time (ns)',
                       run_titles, showlog=False)
    row2 = show_camera(np.array(stddev), engineering_geom, pad_width,
                       pad_height, 'Flat-Field rel. time std dev (ns)',
                       run_titles, showlog=False)
    grid3 = gridplot([row1, row2], sizing_mode=None, plot_width=pad_width,
                     plot_height=pad_height)
    page3.child = grid3
    page3.title = 'Interleaved flat field, time'

    page4 = Panel()
    pulse_fraction_above_10 = []
    pulse_fraction_above_30 = []
    for item in file.root.pixwise_runsummary.col(
            'cosmics_pix_fraction_pulses_above10'):
        pulse_fraction_above_10.append(item)
    for item in file.root.pixwise_runsummary.col(
            'cosmics_pix_fraction_pulses_above30'):
        pulse_fraction_above_30.append(item)

    row1 = show_camera(np.array(pulse_fraction_above_10), engineering_geom,
                       pad_width, pad_height,
                       'Cosmics, fraction of >10pe pulses', run_titles)
    row2 = show_camera(np.array(pulse_fraction_above_30), engineering_geom,
                       pad_width, pad_height,
                       'Cosmics, fraction of >30pe pulses', run_titles)

    grid4 = gridplot([row1, row2], sizing_mode=None, plot_width=pad_width,
                     plot_height=pad_height)
    page4.child = grid4
    page4.title = 'Cosmics'

    file.close()

    page5 = Panel()
    pad_width = 550
    pad_height = 280
    fig_mu_effi = show_graph(x=pd.to_datetime(runsummary['time'], origin='unix',
                                              unit='s'),
                             y=runsummary['mu_effi_mean'],
                             xlabel='date',
                             ylabel='telescope efficiency from mu-rings',
                             ey=runsummary['mu_effi_stddev'] / np.sqrt(
                                     runsummary['num_contained_mu_rings']),
                             xtype='datetime', ytype='linear',
                             point_labels=run_titles)
    fig_mu_effi.y_range = Range1d(0.,1.1*np.max(runsummary['mu_effi_mean']))

    fig_mu_width = show_graph(x=pd.to_datetime(runsummary['time'],
                                               origin='unix', unit='s'),
                              y=runsummary['mu_width_mean'],
                              xlabel='date',
                              ylabel='muon ring width (deg)',
                              ey=runsummary['mu_width_stddev'] / np.sqrt(
                                      runsummary['num_contained_mu_rings']),
                              xtype='datetime', ytype='linear',
                              point_labels=run_titles)
    fig_mu_width.y_range = Range1d(0.,1.1*np.max(runsummary['mu_width_mean']))

    fig_mu_intensity = show_graph(
        x=pd.to_datetime(runsummary['time'], origin='unix', unit='s'),
        y=runsummary['mu_intensity_mean'], xlabel='date',
        ylabel='mean muon ring intensity (p.e.)',
        xtype='datetime', ytype='linear', point_labels=run_titles)
    fig_mu_intensity.y_range = \
        Range1d(0., 1.1 * np.max(runsummary['mu_intensity_mean']))

    fig_mu_hg_peak = show_graph(
        x=pd.to_datetime(runsummary['time'], origin='unix', unit='s'),
        y=runsummary['mu_hg_peak_sample_mean'], xlabel='date',
        ey=runsummary['mu_hg_peak_sample_stddev'],
        ylabel='HG global peak sample id (mean&RMS)',
        xtype='datetime', ytype='linear', point_labels=run_titles)
    fig_mu_hg_peak.y_range = Range1d(0., 38.)
    fig_mu_lg_peak = show_graph(
        x=pd.to_datetime(runsummary['time'], origin='unix', unit='s'),
        y=runsummary['mu_lg_peak_sample_mean'], xlabel='date',
        ey=runsummary['mu_lg_peak_sample_stddev'],
        ylabel='LG global peak sample id (mean&RMS)',
        xtype='datetime', ytype='linear', point_labels=run_titles)
    fig_mu_lg_peak.y_range = Range1d(0., 38.)
    row1 = [fig_mu_effi, fig_mu_width]
    row2 = [fig_mu_intensity]
    row3 = [fig_mu_hg_peak, fig_mu_lg_peak]

    grid5 = gridplot([row1, row2, row3], sizing_mode=None, plot_width=pad_width,
                     plot_height=pad_height)
    page5.child = grid5
    page5.title = "Muons"

    page6 = Panel()
    pad_width = 550
    pad_height = 350
    fig_ped = show_graph(x=pd.to_datetime(runsummary['time'],
                                          origin='unix',
                                          unit='s'),
                         y=runsummary['ped_charge_mean'],
                         xlabel='date',
                         ylabel='Camera-averaged pedestal charge (pe/pixel)',
                         ey=runsummary['ped_charge_mean_err'],
                         xtype='datetime', ytype='linear',
                         point_labels=run_titles)
    fig_ped.y_range = Range1d(0.,1.1*np.max(runsummary['ped_charge_mean']))

    fig_ped_stddev = show_graph(x=pd.to_datetime(runsummary['time'],
                                                 origin='unix',
                                                 unit='s'),
                                y=runsummary['ped_charge_stddev'],
                                xlabel='date',
                                ylabel='Camera-averaged pedestal charge std '
                                       'dev (pe/pixel)',
                                xtype='datetime', ytype='linear',
                                point_labels=run_titles)
    fig_ped_stddev.y_range = \
        Range1d(0.,1.1*np.max(runsummary['ped_charge_stddev']))

    frac = runsummary['num_pedestals_after_cleaning'] / \
           runsummary['num_pedestals']
    err = np.sqrt(frac*(1-frac)/runsummary['num_pedestals'])
    fig_ped_clean_fraction = show_graph(
            x=pd.to_datetime(runsummary['time'], origin='unix', unit='s'),
            y=frac, xlabel='date',
            ylabel='Fraction of pedestals surviving cleaning',
            ey=err, xtype='datetime', ytype='linear',
            point_labels=run_titles)

    row1 = [fig_ped, fig_ped_stddev]
    row2 = [fig_ped_clean_fraction]

    grid6 = gridplot([row1, row2], sizing_mode=None, plot_width=pad_width,
                     plot_height=pad_height)
    page6.child = grid6
    page6.title = "Interleaved pedestals, averages"

    page7 = Panel()
    pad_width = 550
    pad_height = 280
    fig_flatfield = show_graph(x=pd.to_datetime(runsummary['time'],
                                                origin='unix',
                                                unit='s'),
                               y=runsummary['ff_charge_mean'],
                               xlabel='date',
                               ylabel='Cam-averaged FF Q (pe/pixel)',
                               ey=runsummary['ff_charge_mean_err'],
                               xtype='datetime', ytype='linear',
                               point_labels=run_titles)
    fig_flatfield.y_range = Range1d(0.,1.1*np.max(runsummary['ff_charge_mean']))

    fig_ff_stddev = show_graph(x=pd.to_datetime(runsummary['time'],
                                                origin='unix',
                                                unit='s'),
                               y=runsummary['ff_charge_stddev'],
                               xlabel='date',
                               ylabel='Cam-averaged FF Q std '
                                      'dev (pe/pixel)',
                               xtype='datetime', ytype='linear',
                               point_labels=run_titles)
    fig_ff_stddev.y_range = \
        Range1d(0.,1.1*np.max(runsummary['ff_charge_stddev']))

    fig_ff_time = show_graph(x=pd.to_datetime(runsummary['time'],
                                              origin='unix',
                                              unit='s'),
                             y=runsummary['ff_time_mean'],
                             xlabel='date',
                             ylabel='Cam-averaged FF time (ns)',
                             ey=runsummary['ff_time_mean_err'],
                             xtype='datetime', ytype='linear',
                             point_labels=run_titles)

    fig_ff_time_std = show_graph(x=pd.to_datetime(runsummary['time'],
                                                  origin='unix',
                                                  unit='s'),
                                 y=runsummary['ff_time_stddev'],
                                 xlabel='date',
                                 ylabel='Cam-averaged FF t std '
                                        'dev (ns)',
                                 xtype='datetime', ytype='linear',
                                 point_labels=run_titles)
    fig_ff_rel_time_std = show_graph(x=pd.to_datetime(runsummary['time'],
                                                      origin='unix',
                                                      unit='s'),
                                     y=runsummary['ff_rel_time_stddev'],
                                     xlabel='date',
                                     ylabel='Cam-averaged FF '
                                            'rel. pix t std dev (ns)',
                                     xtype='datetime', ytype='linear',
                                     point_labels=run_titles)
    fig_ff_rel_time_std.y_range = \
        Range1d(0., np.max([1., runsummary['ff_rel_time_stddev'].max()]))

    row1 = [fig_flatfield, fig_ff_stddev]
    row2 = [fig_ff_time, fig_ff_time_std]
    row3 = [fig_ff_rel_time_std]

    grid7 = gridplot([row1, row2, row3], sizing_mode=None, plot_width=pad_width,
                     plot_height=pad_height)
    page7.child = grid7
    page7.title = "Interleaved FF, averages"

    tabs = Tabs(tabs=[page0, page0b, page1, page2,
                      page3, page4, page5, page6, page7])
    show(column(Div(text='<h1> Long-term DL1 data check </h1>'), tabs))


def show_graph(x, y, xlabel, ylabel, ey=None, eylow=None, eyhigh=None,
               xtype='linear', ytype='linear',
               point_labels=None):
    '''
    Function to display a simple "y vs. x" graph, with y error bars
    Parameters
    ----------
    x: ndarray, x coordinates
    y: ndarray, y coordinates
    ey: ndarray, size of y error bars
    xlabel: x-axis label
    ylabel: y-axis label
    xtype: 'log', 'linear', 'datetime'
    ytype: 'log', 'linear', 'datetime'
    point_labels: one label per point, to be displayed when mouse overs near
                  point

    Returns
    -------
    A bokeh.plotting.figure with the y vs. x graph
    '''

    fig = figure(background_fill_color='#ffffff', x_axis_label=xlabel,
                 x_axis_type=xtype, y_axis_type=ytype, y_axis_label=ylabel)
    source = ColumnDataSource(data=dict(x=x, y=y))
    if point_labels is not None:
        source.data['point_labels'] = point_labels
    fig.circle(x='x', y='y', size=2, source=source)

    if eylow is None:
        eylow = ey
    if eyhigh is None:
        eyhigh = ey

    if eylow is not None or eyhigh is not None:
        yhigh = y
        ylow = y
        if eylow is not None:
            ylow = y - eylow
        if eyhigh is not None:
            yhigh = y + eyhigh
        source_error = ColumnDataSource(data=dict(base=x, lower=ylow, upper=yhigh))
        error_bars = Whisker(source=source_error, base="base", lower="lower",
                             upper="upper")
        error_bars.line_color = 'steelblue'
        error_bars.upper_head.line_color = 'steelblue'
        error_bars.lower_head.line_color = 'steelblue'
        error_bars.upper_head.size = 4
        error_bars.lower_head.size = 4
        fig.add_layout(error_bars)

    if point_labels is not None:
        fig.add_tools(HoverTool(tooltips=[('value', '@y'),
                                          ('point id', '@point_labels')],
                                mode='mouse',
                                point_policy='snap_to_data'))

    return fig


if __name__ == '__main__':
    main()
