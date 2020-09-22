#!/usr/bin/env python

"""
This script has to be run in a directory containing LST1 DL1 datacheck files, of
those containing inf for a whole run, with name pattern
datacheck_dl1_LST-1.Run?????.h5

The corresponding muon*fits files (which are subrun-wise) have to be present
in the same directory.

The output is the file longterm_dl1_check.h5 file, which contains tables with
some run-wise summary values for plotting long-term evolution of the DL1 data.
"""

from astropy.table import Table
import copy
import datetime
import glob
import numpy as np
import pandas as pd
import tables

from bokeh.io import output_file as bokeh_output_file
from bokeh.io import show
from bokeh.layouts import gridplot, column
from bokeh.models import Div, ColumnDataSource, Whisker, HoverTool, Range1d
from bokeh.models.widgets import Tabs, Panel
from bokeh.plotting import figure
from ctapipe.instrument import CameraGeometry
from ctapipe.coordinates import EngineeringCameraFrame
from lstchain.datachecks import show_camera
from pathlib import Path

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
                   # currently (as of lstchain 0.5.3) event numbers are post-cleaning!:
                  'num_cosmics': [],
                  'num_pedestals': [],
                  'num_flatfield': [],
                  'num_contained_mu_rings': [],
                  'ff_charge_mean': [],   # in whole camera
                  'ff_charge_stddev': [], # in whole camera
                  'ped_fraction_pulses_above10': [], # in whole camera
                  'ped_fraction_pulses_above30': [], # in whole camera
                  'cosmics_fraction_pulses_above10': [], # in whole camera
                  'cosmics_fraction_pulses_above30': [], # in whole camera
                  'mu_effi_mean': [],
                  'mu_effi_stddev': [],
                  'mu_width_mean': [],
                  'mu_width_stddev': [],
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
        time = tables.Float32Col()
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
        except:
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
            except:
                print('   Table', name, 'is missing!')
                datatables.append(None)
                continue

            datatables.append(node)

        subruns = None

        # fill data which are common to all tables:
        for table, dict in zip(datatables, dicts):
            if table is None:
                continue
            dict['runnumber'].extend(len(table)*[runnumber])
            dict['subrun'].extend(table.col('subrun_index'))
            dict['elapsed_time'].extend(table.col('elapsed_time'))
            dict['events'].extend(table.col('num_events'))
            dict['time'].extend(table.col('dragon_time').mean(axis=1))
            dict['azimuth'].extend(table.col('mean_az_tel'))
            dict['altitude'].extend(table.col('mean_alt_tel'))

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
                 runsummary['num_cosmics'][
                                                                          -1]])

        if datatables[1] is not None:
            table = a.root.dl1datacheck.pedestals
            runsummary['num_pedestals'].extend([table.col('num_events').sum()])

            runsummary['ped_fraction_pulses_above10'].extend([(table.col('num_pulses_above_0010_pe').mean(axis=1)).sum()/
                                                             runsummary['num_pedestals'][-1]])
            runsummary['ped_fraction_pulses_above30'].extend([(table.col('num_pulses_above_0030_pe').mean(axis=1)).sum()/
                                                              runsummary['num_pedestals'][-1]])
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
            runsummary['ped_fraction_pulses_above10'].extend([np.nan])
            runsummary['ped_fraction_pulses_above30'].extend([np.nan])
            pixwise_runsummary['ped_pix_fraction_pulses_above10'].extend([numpixels*[np.nan]])
            pixwise_runsummary['ped_pix_fraction_pulses_above30'].extend([numpixels*[np.nan]])
            pixwise_runsummary['ped_pix_charge_mean'].extend([numpixels*[np.nan]])
            pixwise_runsummary['ped_pix_charge_stddev'].extend([numpixels*[np.nan]])


        if datatables[2] is not None:
            table = a.root.dl1datacheck.flatfield
            runsummary['num_flatfield'].extend([table.col('num_events').sum()])
            runsummary['ff_charge_mean'].extend([table.col('charge_mean').mean()])  # mean for all pixels and subruns of the subrun-pixel-mean
            runsummary['ff_charge_stddev'].extend([table.col('charge_stddev').mean()]) # mean for all pixels and subruns of the subrun-pixel-stddev
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
            runsummary['ff_charge_stddev'].extend([np.nan])
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

            dat = Table.read(mufile, format='fits')
            if len(dat) == 0:
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
            runsummary['mu_effi_mean'].extend([contained_mu_wholerun['muon_efficiency'].mean()])
            runsummary['mu_effi_stddev'].extend([contained_mu_wholerun['muon_efficiency'].std()])
            runsummary['mu_width_mean'].extend([contained_mu_wholerun['ring_width'].mean()])
            runsummary['mu_width_stddev'].extend([contained_mu_wholerun['ring_width'].std()])
            runsummary['mu_intensity_mean'].extend([contained_mu_wholerun['ring_size'].mean()])
        else:
            runsummary['num_contained_mu_rings'].extend([np.nan])
            runsummary['mu_effi_mean'].extend([np.nan])
            runsummary['mu_effi_stddev'].extend([np.nan])
            runsummary['mu_width_mean'].extend([np.nan])
            runsummary['mu_width_stddev'].extend([np.nan])
            runsummary['mu_intensity_mean'].extend([np.nan])

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
    for dict, name in zip(dicts, ['cosmics', 'pedestals', 'flatfield']):
        pd.DataFrame(dict).to_hdf(output_file_name, key=name, mode='a')

    # We write out the camera geometry information, assuming it is the same
    # for all files (hence we take it from the first one):
    cam_description_table = \
        Table.read(files[0], path='instrument/telescope/camera/LSTCam')
    geom = CameraGeometry.from_table(cam_description_table)
    geom.to_table().write(output_file_name,
                          path=f'/instrument/telescope/camera/LSTCam',
                          append=True, serialize_meta=True)

    plot(output_file_name)


def plot(filename='longterm_dl1_check.h5'):

    # First read in the camera geometry:
    cam_description_table = \
        Table.read(filename, path='instrument/telescope/camera/LSTCam')
    camgeom = CameraGeometry.from_table(cam_description_table)
    engineering_geom = camgeom.transform_to(EngineeringCameraFrame())

    file = tables.open_file('longterm_dl1_check.h5')

    bokeh_output_file(Path(filename).with_suffix('.html'),
                      title='LST1 long-term DL1 data check')

    pad_width = 350
    pad_height = 370

    run_titles = []
    for i, run in enumerate(file.root.pixwise_runsummary.col('runnumber')):
        date = pd.to_datetime(file.root.pixwise_runsummary.col('time')[i],
                              origin='unix', unit='s')
        run_titles.append('Run {0:05d}, {date}'.\
                          format(run,
                                 date = date.strftime("%b %d %Y %H:%M:%S")))

    page1 = Panel()
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

    pad_width = 550
    pad_height = 350

    runsummary = pd.read_hdf(filename, 'runsummary')
    page5 = Panel()
    fig_mu_effi = show_graph(x=pd.to_datetime(runsummary['time'], origin='unix',
                                              unit='s'),
                             y=runsummary['mu_effi_mean'],
                             ey=runsummary['mu_effi_stddev'],
                             xlabel='date',
                             ylabel='telescope efficiency from mu-rings (mean & std dev)',
                             xtype='datetime', ytype='linear',
                             point_labels=run_titles)
    fig_mu_effi.y_range = Range1d(0.,1.1*np.max(runsummary['mu_effi_mean']))
    row1 = [fig_mu_effi]
    grid5 = gridplot([row1], sizing_mode=None, plot_width=pad_width,
                     plot_height=pad_height)
    page5.child = grid5
    page5.title = "Muons"

    tabs = Tabs(tabs=[page1, page2, page3, page4, page5])
    show(column(Div(text='<h1> Long-term DL1 data check </h1>'), tabs))


def show_graph(x, y, ey, xlabel, ylabel, xtype='linear', ytype='linear',
               point_labels=None):
    '''
    Function to display a simple "y vs. x" graph, with y error bars
    Parameters
    ----------
    x
    y
    ey
    xlabel
    ylabel
    xtype:
    ytype:  'log', 'linear', 'datetime'
    point_labels

    Returns
    -------
    A bokeh.plotting.figure with the y vs. x graph
    '''

    fig = figure(background_fill_color='#ffffff', x_axis_label=xlabel,
                 x_axis_type=xtype, y_axis_type=ytype, y_axis_label=ylabel)
    ylow = y - ey
    yhigh = y + ey
    source = ColumnDataSource(data=dict(x=x, y=y))
    if point_labels is not None:
        source.data['point_labels'] = point_labels
    source_error = ColumnDataSource(data=dict(base=x, lower=ylow, upper=yhigh))
    fig.circle(x='x', y='y', size=2, source=source)
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
