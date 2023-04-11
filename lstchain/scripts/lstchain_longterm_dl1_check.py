#!/usr/bin/env python

"""
This script reads in LST1 DL1 datacheck files, of those containing info for a
whole run, with name pattern datacheck_dl1_LST-1.Run?????.h5. It takes all
files in the input directory. It also reads, in the directory indicated (
muons-dir), the corresponding muons*fits files if available.

The output is the file longterm_dl1_check.h5 file (the name can be modified via
commandline), which contains tables with some run-wise summary values for
plotting long-term evolution of the DL1 data.

It also produces an interactive web page, longterm_dl1_check.html with plots
showing the evolution of many such values. If not in batch mode, the page is
opened by the default browser at the end of execution.

It also produces a longterm_dl1_check.log file with warnings about values
which are beyond certain limits.

"""

import argparse
import copy
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import tables
import warnings

from astropy.table import Table
from astropy.coordinates import Angle
import astropy.units as u

from bokeh.io import output_file as bokeh_output_file
from bokeh.io import show, save
from bokeh.layouts import gridplot, column
from bokeh.models import (
    ColumnDataSource,
    Div,
    HoverTool,
    Range1d,
    Whisker,
)
from bokeh.models.widgets import Tabs, Panel
from bokeh.plotting import figure
from ctapipe.coordinates import EngineeringCameraFrame
# from ctapipe.instrument import SubarrayDescription
from ctapipe_io_lst import load_camera_geometry

from ctapipe.io import read_table
from ctapipe_io_lst import TriggerBits

from lstchain.visualization.bokeh import show_camera, get_pixel_location

log = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description="DL1 multi-run data checker")
parser.add_argument('--input-dir', '-d', type=Path,
                    help='path to the datacheck_dl1_LST-1.RunXXXXX.h5 files',
                    default='./')

parser.add_argument('--muons-dir', type=Path,
                    help='path to the muons_LST-1.RunXXXXX.YYYY.fits files',
                    default='./')

parser.add_argument('--output-file', '-o', type=Path,
                    help='.h5 output file name',
                    default='./longterm_dl1_check.h5')

parser.add_argument('--batch', '-b', action='store_true',
                    help='Run the script without opening html output'
                    )


def main():
    args = parser.parse_args()

    output_file_name = args.output_file
    # Create directories if needed:
    output_file_name.parent.mkdir(parents=True, exist_ok=True)

    log.setLevel(logging.INFO)
    logfilename = output_file_name.with_suffix('.log')
    handler = logging.FileHandler(logfilename, mode='w')
    logging.getLogger().addHandler(handler)

    files = sorted(args.input_dir.glob('datacheck_dl1_LST-1.Run?????.h5'))

    if not files:
        raise IOError("No input datacheck files found")

    print('Processing', len(files), 'datacheck files...')
    print('Writing out:')
    print('    ', logfilename)
    print('    ', output_file_name)
    print('    ', output_file_name.with_suffix('.html'))

    # hardcoded for now, to be eventually read from data:
    numpixels = 1855

    # subrun-wise tables: cosmics, pedestals, flatfield. One dictionary per
    # each. Note that the cosmics table contains also muon ring information!
    # The tables have one row per subrun

    cosmics = {'runnumber': [],
               'subrun': [],
               'time': [],
               'elapsed_time': [],
               'events': [],
               'azimuth': [],
               'altitude': [],
               # number of events with wrong trigger type:
               'wrong_ucts_trig_type': [],
               'wrong_tib_trig_type': [],
               'unknown_ucts_trig_type': [],
               'unknown_tib_trig_type': [],
               'num_ucts_jumps': []}

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
    cosmics['fraction_pulses_above10'] = []  # fraction of >10 pe pulses
    cosmics['fraction_pulses_above30'] = []  # fraction of >30 pe pulses

    pedestals['fraction_pulses_above10'] = []  # fraction of >10 pe pulses
    pedestals['fraction_pulses_above30'] = []  # fraction of >30 pe pulses
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
                  'min_azimuth': [],
                  'max_azimuth': [],
                  'mean_azimuth': [],
                  'mean_ra': [],
                  'mean_dec': [],
                  'num_cosmics': [],
                  'num_pedestals': [],
                  'num_flatfield': [],

                  'num_unknown_ucts_trigger_tags': [],
                  'num_wrong_ucts_tags_in_cosmics': [],
                  'num_wrong_ucts_tags_in_pedestals': [],
                  'num_wrong_ucts_tags_in_flatfield': [],
                  'num_ucts_jumps': [],

                  'num_unknown_tib_trigger_tags': [],
                  'num_wrong_tib_tags_in_cosmics': [],
                  'num_wrong_tib_tags_in_pedestals': [],
                  'num_wrong_tib_tags_in_flatfield': [],

                  'num_pedestals_after_cleaning': [],
                  'num_contained_mu_rings': [],

                  'ff_charge_mean': [],  # camera average of mean pix FF charge
                  'ff_charge_mean_err': [],  # uncertainty of the above
                  'ff_charge_stddev': [],  # camera average
                  'ff_time_mean': [],  # camera average of mean FF time
                  'ff_time_mean_err': [],  # uncertainty of the above
                  'ff_time_stddev': [],  # camera average
                  'ff_rel_time_stddev': [],  # camera-averaged std dev of pixel t
                  # w.r.t. average of rest of pixels in camera (~ t-resolution)
                  'ped_charge_mean': [],  # camera average of mean pix ped charge
                  'ped_charge_mean_err': [],  # uncertainty of the above
                  'ped_charge_stddev': [],  # camera average
                  'ped_fraction_pulses_above10': [],  # in whole camera
                  'ped_fraction_pulses_above30': [],  # in whole camera
                  'cosmics_fraction_pulses_above10': [],  # in whole camera
                  'cosmics_fraction_pulses_above30': [],  # in whole camera

                  'mu_effi_mean': [],
                  'mu_effi_stddev': [],
                  'mu_width_mean': [],
                  'mu_width_stddev': [],
                  'mu_hg_peak_sample_mean': [],
                  'mu_hg_peak_sample_stddev': [],
                  'mu_intensity_mean': [],
                  'mean_number_of_pixels_nearby_stars': []}


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
                          'cosmics_pix_fraction_pulses_above30': [],
                          'cosmics_cog_within_pixel': [],
                          'cosmics_cog_within_pixel_intensity_gt_200': [],
                          'ncosmics_per_pix': [],
                          'elapsed_time_per_pix': []
                          }
    pixwise_runsummary_no_stars = copy.deepcopy(pixwise_runsummary)

    # Needed for the table description for writing it out to the hdf5 file:
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
        cosmics_cog_within_pixel =  tables.Float32Col(shape=(numpixels))
        cosmics_cog_within_pixel_intensity_gt_200 = tables.Float32Col(shape=(numpixels))
        ncosmics_per_pix = tables.Float32Col(shape=(numpixels))
        elapsed_time_per_pix = tables.Float32Col(shape=(numpixels))

    dicts = [cosmics, pedestals, flatfield]

    # files are of the type datacheck_dl1_LST-1.RunXXXXX.h5
    for file in files:

        try:
            a = tables.open_file(file)
        except FileNotFoundError:
            log.warning(f'Could not read file {file} - skipping...')
            continue

        runnumber = int(file.name[file.name.find('.Run') + 4:
                                  file.name.find('.Run') + 9])

        # Lists to keep the datacheck tables for cosmics, pedestals and
        # flatfield. The "_no_stars" list will have nans for pixels which
        # were close to stars during a given subrun
        datatables = []
        datatables_no_stars = []

        for tablename in ['cosmics', 'pedestals', 'flatfield']:
            try:
                a.get_node('/dl1datacheck/' + tablename)
            except Exception:
                log.warning(f'Run {runnumber}: Table {tablename} is missing!')
                datatables.append(None)
                datatables_no_stars.append(None)
                continue

            datatables.append(get_datacheck_table(file, tablename))
            datatables_no_stars.append(get_datacheck_table(file, tablename,
                                                           exclude_stars=True))
        a.close()

        trig_tags = [TriggerBits.PHYSICS.value,
                     TriggerBits.PEDESTAL.value,
                     TriggerBits.CALIBRATION.value]

        # fill data which are common to all tables:

        total_num_ucts_jumps = 0   # To add up all jumps, for any event type
        total_num_unknown_ucts = 0 # To add up events with unknown ucts
        total_num_unknown_tib = 0  # The same for tib

        # The tables in each file correspond to a full run, and contain one
        # row per subrun, what we do below is
        # - to add the subrun-wise rows to tables that will span the whole
        #   processed set of runs
        #
        # - to calculate some average values for each run, some pixel-wise,
        #  some global for the whole camera
        #
        for table, d, tag in zip(datatables, dicts, trig_tags):
            if table is None:
                continue
            # check the number of wrong trigger types, i.e. those events which
            # do not have the expected trigger bits for a cosmic, pedestal or
            # flatfield event, depending on the table we are processing. Note
            # that the events were classified in the tables (see
            # dl1_checker.py) using event_type as filled by the LST event source
            #
            # num_wrong_tags = [ndarray, ndarray]  (for ucts & tib respectively)
            # Each array has one entry per subrun, with the number of mismatches
            num_wrong_tags = trigtag_mismatches(table, tag)
            d['wrong_ucts_trig_type'].extend(num_wrong_tags[0])
            d['wrong_tib_trig_type'].extend(num_wrong_tags[1])

            # Store and add up the number of unknown trigger tags of each type:
            # in trigger_type and ucts_trigger_type the first index is for
            # the subruns, the second goes over different trigger tags
            # present in the data, and in the third [0] means the trigger
            # tag value and [1] the number of events in the subrun with that
            # value:
            # TriggerBits.UNKNOWN (0)
            unknown_mask = table['trigger_type'][:, :, 0] == 0
            num_unknowns = np.ma.array(table['trigger_type'][:, :, 1],
                                       mask = ~unknown_mask).sum(axis=1).data
            d['unknown_tib_trig_type'].extend(num_unknowns)
            total_num_unknown_tib += num_unknowns.sum()

            unknown_mask = table['ucts_trigger_type'][:, :, 0] == 0
            num_unknowns = np.ma.array(table['ucts_trigger_type'][:, :, 1],
                                       mask = ~unknown_mask).sum(axis=1).data
            d['unknown_ucts_trig_type'].extend(num_unknowns)
            total_num_unknown_ucts += num_unknowns.sum()

            if 'num_ucts_jumps' in table.colnames:
                d['num_ucts_jumps'].extend(table['num_ucts_jumps'])
                total_num_ucts_jumps += np.sum(table['num_ucts_jumps'])
            # In case we are running over lstchain <v0.8 datacheck files:
            else:
                d['num_ucts_jumps'].extend(np.zeros(len(table),
                                                    dtype='int'))

            d['runnumber'].extend(len(table) * [runnumber])
            d['subrun'].extend(table['subrun_index'])
            d['elapsed_time'].extend(table['elapsed_time'])
            d['events'].extend(table['num_events'])
            d['time'].extend(table['dragon_time'].mean(axis=1))
            d['azimuth'].extend(table['mean_az_tel'])
            d['altitude'].extend(table['mean_alt_tel'])

        # now fill event-type-specific quantities. In some cases they are
        # pixel-averaged values:

        # Cosmics
        if datatables[0] is not None:
            table = datatables[0]

            cosmics['fraction_pulses_above10'].extend(
                table['num_pulses_above_0010_pe'].mean(axis=1) /
                table['num_events'])
            cosmics['fraction_pulses_above30'].extend(
                table['num_pulses_above_0030_pe'].mean(axis=1) /
                table['num_events'])
        # Pedestals
        if datatables[1] is not None:
            table = datatables[1]
            pedestals['fraction_pulses_above10'].extend(
                table['num_pulses_above_0010_pe'].mean(axis=1) /
                table['num_events'])
            pedestals['fraction_pulses_above30'].extend(
                table['num_pulses_above_0030_pe'].mean(axis=1) /
                table['num_events'])
            pedestals['charge_mean'].extend(
                table['charge_mean'].mean(axis=1))
            pedestals['charge_stddev'].extend(
                table['charge_stddev'].mean(axis=1))

        # Flatfield
        if datatables[2] is not None:
            table = datatables[2]

            flatfield['charge_mean'].extend(
                np.nanmean(table['charge_mean'], axis=1))
            flatfield['charge_stddev'].extend(
                np.nanmean(table['charge_stddev'], axis=1))
            flatfield['rel_time_mean'].extend(
                np.nanmean(table['relative_time_mean'], axis=1))
            flatfield['rel_time_stddev'].extend(
                np.nanmean(table['relative_time_stddev'], axis=1))

        # So far we have just filled the pedestals, cosmics and flatfield
        # subrun-wise tables (that we will later write out) for all the subruns
        # in this file


        #
        # Now we fill the runsummary table.
        #

        # Cosmics:
        table = datatables[0]
        table_no_stars = datatables_no_stars[0]

        # keep subrun list, needed later for the muons:
        subruns = table['subrun_index']

        # now fill the run-wise table - just one entry, which corresponds to
        # the file datacheck_dl1_LST-1.RunXXXXX.h5 we are processing in this
        # iteration:
        runsummary['runnumber'].extend([runnumber])
        runsummary['time'].extend([table['dragon_time'].mean()])
        runsummary['elapsed_time'].extend([table['elapsed_time'].sum()])
        runsummary['min_altitude'].extend([table['mean_alt_tel'].min()])
        runsummary['mean_altitude'].extend([table['mean_alt_tel'].mean()])
        runsummary['max_altitude'].extend([table['mean_alt_tel'].max()])

        runsummary['min_azimuth'].extend([table['mean_az_tel'].min()])
        az = table['mean_az_tel']
        mean_az = np.arctan2(np.mean(np.sin(az)), np.mean(np.cos(az)))
        mean_az = Angle(mean_az, u.rad).wrap_at('360d').rad
        
        runsummary['mean_azimuth'].extend([mean_az])
        runsummary['max_azimuth'].extend([table['mean_az_tel'].max()])

        ra = np.deg2rad(table['tel_ra'])
        mean_ra = np.rad2deg(np.arctan2(np.mean(np.sin(ra)), 
                                        np.mean(np.cos(ra))))
        mean_ra = Angle(mean_ra, u.deg).wrap_at('360d').deg
        
        runsummary['mean_ra'].extend([mean_ra])

        runsummary['mean_dec'].extend([table['tel_dec'].mean()])

        num_events = table['num_events'].sum()
        runsummary['num_cosmics'].extend([num_events])

        # Number of wrong trigger tags for this whole run (add up only
        # the numbers for its subruns (a total of len(table)):
        nwucts = np.sum(cosmics['wrong_ucts_trig_type'][-len(table):])
        runsummary['num_wrong_ucts_tags_in_cosmics'].extend([nwucts])
        nwtib = np.sum(cosmics['wrong_tib_trig_type'][-len(table):])
        runsummary['num_wrong_tib_tags_in_cosmics'].extend([nwtib])

        # Number of 'unknown' trigger tags for this whole run, for any kind
        # of event type:
        runsummary['num_unknown_tib_trigger_tags'].extend([
            total_num_unknown_tib])
        runsummary['num_unknown_ucts_trigger_tags'].extend([
            total_num_unknown_ucts])

        # number of ucts jumps in the run, for any kind of event type:
        runsummary['num_ucts_jumps'].extend([total_num_ucts_jumps])


        # Form camera-averaged values we use the "no_stars" version of the
        # table, in which values for pixels with nearbu stars are set to nan:
        runsummary['cosmics_fraction_pulses_above10'].extend(
            [np.nansum(table_no_stars['num_pulses_above_0010_pe']) /
             np.nansum(table_no_stars['nevents_per_pix'])])

        runsummary['cosmics_fraction_pulses_above30'].extend(
            [np.nansum(table_no_stars['num_pulses_above_0030_pe']) /
             np.nansum(table_no_stars['nevents_per_pix'])])

        # Pedestals:
        if datatables[1] is not None:
            table = datatables[1]
            table_no_stars = datatables_no_stars[1]
            nevents = table['num_events']  # events per subrun
            events_in_run = nevents.sum()

            # total events:
            num_events = table['num_events'].sum()
            runsummary['num_pedestals'].extend([num_events])

            # Number of wrong trigger tags for this whole run (add up only
            # the numbers for its subruns (a total of len(table)):
            nwucts = np.sum(pedestals['wrong_ucts_trig_type'][-len(table):])
            runsummary['num_wrong_ucts_tags_in_pedestals'].extend([nwucts])
            nwtib = np.sum(pedestals['wrong_tib_trig_type'][-len(table):])
            runsummary['num_wrong_tib_tags_in_pedestals'].extend([nwtib])

            runsummary['num_pedestals_after_cleaning'].\
                extend([table['num_cleaned_events'].sum()])

            runsummary['ped_fraction_pulses_above10'].extend(
                [np.nansum(table_no_stars['num_pulses_above_0010_pe']) /
                 np.nansum(table_no_stars['nevents_per_pix'])])

            runsummary['ped_fraction_pulses_above30'].extend(
                [np.nansum(table_no_stars['num_pulses_above_0030_pe']) /
                 np.nansum(table_no_stars['nevents_per_pix'])])

            # charge_mean is [pixels], containing pixwise means in the full run:
            charge_mean = \
                pix_subrun_mean_to_run_mean(table_no_stars['charge_mean'],
                                            table_no_stars['nevents_per_pix'])

            # Now store the pixel-averaged mean pedestal charge:
            runsummary['ped_charge_mean'].extend([np.nanmean(charge_mean)])
            # error on the mean:
            npixels = (~np.isnan(charge_mean)).sum()
            runsummary['ped_charge_mean_err'].extend([np.nanstd(charge_mean) /
                                                      np.sqrt(npixels)])

            nstars = table['num_nearby_stars']
            mean_number_of_pixels_nearby_stars = np.nanmean(np.nansum(nstars,
                                                                      axis=1))
            runsummary['mean_number_of_pixels_nearby_stars'].extend(
                [mean_number_of_pixels_nearby_stars])

            # charge_stddev is [pixels], containing pixwise means in the full
            # run:
            charge_stddev = pix_subrun_std_to_run_std(
                    table_no_stars['charge_stddev'],
                    table_no_stars['nevents_per_pix'])

            # Store the pixel-averaged pedestal charge std dev through a run:
            runsummary['ped_charge_stddev'].extend([np.nanmean(charge_stddev)])

        else:
            runsummary['num_pedestals'].extend([np.nan])
            runsummary['num_wrong_ucts_tags_in_pedestals'].extend([np.nan])
            runsummary['num_wrong_tib_tags_in_pedestals'].extend([np.nan])
            runsummary['num_pedestals_after_cleaning'].extend([np.nan])
            runsummary['ped_fraction_pulses_above10'].extend([np.nan])
            runsummary['ped_fraction_pulses_above30'].extend([np.nan])
            runsummary['ped_charge_mean'].extend([np.nan])
            runsummary['ped_charge_mean_err'].extend([np.nan])
            runsummary['ped_charge_stddev'].extend([np.nan])
            runsummary['mean_number_of_pixels_nearby_stars'].extend([np.nan])

        # Flatfield
        if datatables[2] is not None:
            table = datatables[2]
            table_no_stars = datatables_no_stars[2]

            nevents = table['num_events']  # events per subrun
            events_in_run = nevents.sum()
            runsummary['num_flatfield'].extend([events_in_run])

            # Number of wrong trigger tags for this run only (add up only
            # the numbers for its subruns (a total of len(table)):
            nwucts = np.sum(flatfield['wrong_ucts_trig_type'][-len(table):])
            runsummary['num_wrong_ucts_tags_in_flatfield'].extend([nwucts])
            nwtib = np.sum(flatfield['wrong_tib_trig_type'][-len(table):])
            runsummary['num_wrong_tib_tags_in_flatfield'].extend([nwtib])

            # Mean flat field charge through a run, for each pixel:
            charge_mean = pix_subrun_mean_to_run_mean(
                    table_no_stars['charge_mean'],
                    table_no_stars['nevents_per_pix'])

            # Mean flat field time through a run, for each pixel:
            time_mean = pix_subrun_mean_to_run_mean(
                    table_no_stars['time_mean'],
                    table_no_stars['nevents_per_pix'])

            # Now store the pixel-averaged mean charge:
            runsummary['ff_charge_mean'].extend([np.nanmean(charge_mean)])
            npixels = (~np.isnan(charge_mean)).sum()
            runsummary['ff_charge_mean_err'].extend([np.nanstd(charge_mean) /
                                                     np.sqrt(npixels)])

            # FF charge std dev through a run, for each pixel:
            charge_stddev = pix_subrun_std_to_run_std(
                    table_no_stars['charge_stddev'],
                    table_no_stars['nevents_per_pix'])
            # Store the pixel-averaged FF charge std dev:
            runsummary['ff_charge_stddev'].extend([np.nanmean(charge_stddev)])

            # Pixel-averaged mean time:
            runsummary['ff_time_mean'].extend([np.nanmean(time_mean)])
            npixels = (~np.isnan(time_mean)).sum()
            runsummary['ff_time_mean_err'].extend([np.nanstd(time_mean) /
                                                   np.sqrt(npixels)])
            # FF time std dev through a run, for each pixel:
            time_stddev = pix_subrun_std_to_run_std(
                    table_no_stars['time_stddev'],
                    table_no_stars['nevents_per_pix'])

            # Store the pixel-averaged FF time std dev:
            runsummary['ff_time_stddev'].extend([np.nanmean(time_stddev)])

            rel_time_stddev = pix_subrun_std_to_run_std(
                    table_no_stars['relative_time_stddev'],
                    table_no_stars['nevents_per_pix'])
            runsummary['ff_rel_time_stddev']. \
                extend([np.nanmean(rel_time_stddev)])

        else:
            runsummary['num_flatfield'].extend([np.nan])
            runsummary['num_wrong_ucts_tags_in_flatfield'].extend([np.nan])
            runsummary['num_wrong_tib_tags_in_flatfield'].extend([np.nan])
            runsummary['ff_charge_mean'].extend([np.nan])
            runsummary['ff_charge_mean_err'].extend([np.nan])
            runsummary['ff_charge_stddev'].extend([np.nan])
            runsummary['ff_time_mean'].extend([np.nan])
            runsummary['ff_time_mean_err'].extend([np.nan])
            runsummary['ff_time_stddev'].extend([np.nan])
            runsummary['ff_rel_time_stddev'].extend([np.nan])


        # Now we fill the pixel-wise run summaries, one in which we use all
        # subruns & pixels for calculations, and another one in which we
        # exclude in each subrun the pixels which were close to stars

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            # Just to avoid annoying rare warnings on invalid values

            for pwrs, dctable, tag in zip([pixwise_runsummary,
                                           pixwise_runsummary_no_stars],
                                          [datatables, datatables_no_stars],
                                          ['', '_no_stars']):
                # Cosmics
                table = dctable[0]
                total_ncosmics_per_pix = np.nansum(table['nevents_per_pix'],
                                                   axis=0)
                total_elapsed_time_per_pix =  \
                    np.nansum(table['elapsed_time_per_pix'], axis=0)
                pwrs['ncosmics_per_pix'].extend([total_ncosmics_per_pix])
                pwrs['elapsed_time_per_pix'].extend([
                    total_elapsed_time_per_pix])

                pwrs['cosmics_pix_fraction_pulses_above10'].extend(
                        [np.nansum(table['num_pulses_above_0010_pe'], axis=0) /
                         total_ncosmics_per_pix])
                pwrs['cosmics_pix_fraction_pulses_above30'].extend(
                        [np.nansum(table['num_pulses_above_0030_pe'], axis=0) /
                         total_ncosmics_per_pix])
                pwrs['cosmics_cog_within_pixel'].extend(
                    [np.nansum(table['cog_within_pixel'], axis=0)])
                pwrs['cosmics_cog_within_pixel_intensity_gt_200'].extend(
                    [np.nansum(table['cog_within_pixel_intensity_gt_200'], axis=0)])

                # Pedestals
                if dctable[1] is not None:
                    table = dctable[1]
                    total_nped_per_pix = np.nansum(table['nevents_per_pix'],
                                                      axis=0)
                    pwrs['ped_pix_fraction_pulses_above10'].\
                        extend([np.nansum(table['num_pulses_above_0010_pe'],
                                          axis=0) / total_nped_per_pix])
                    pwrs['ped_pix_fraction_pulses_above30'].\
                        extend([np.nansum(table['num_pulses_above_0030_pe'],
                                          axis=0) / total_nped_per_pix])

                    # For the means we consider the statistics in each subrun
                    # when recalculating the mean for the whole run:
                    qmean = pix_subrun_mean_to_run_mean(table['charge_mean'],
                                                        table['nevents_per_pix'])
                    pwrs['ped_pix_charge_mean'].extend([qmean])

                    # Similar procedure for the run-wise standard deviations:
                    qstd = pix_subrun_std_to_run_std(table['charge_stddev'],
                                                     table['nevents_per_pix'])
                    pwrs['ped_pix_charge_stddev'].extend([qstd])

                else:
                    for key in ['ped_pix_fraction_pulses_above10',
                                'ped_pix_fraction_pulses_above30',
                                'ped_pix_charge_mean',
                                'ped_pix_charge_stddev']:
                        pwrs[key].extend([numpixels * [np.nan]])

                # Flatfield
                if dctable[2] is not None:
                    table = dctable[2]

                    qmean = pix_subrun_mean_to_run_mean(table['charge_mean'],
                                                        table['nevents_per_pix'])
                    pwrs['ff_pix_charge_mean'].extend([qmean])
                    qstd = pix_subrun_mean_to_run_mean(table['charge_stddev'],
                                                        table['nevents_per_pix'])
                    pwrs['ff_pix_charge_stddev'].extend([qstd])
                    tmean = pix_subrun_mean_to_run_mean(table['relative_time_mean'],
                                                        table['nevents_per_pix'])
                    pwrs['ff_pix_rel_time_mean'].extend([tmean])
                    tstd = pix_subrun_std_to_run_std(table['relative_time_stddev'],
                                                     table['nevents_per_pix'])
                    pwrs['ff_pix_rel_time_stddev'].extend([tstd])
                else:
                    for key in ['ff_pix_charge_mean', 'ff_pix_charge_stddev',
                                'ff_pix_rel_time_mean', 'ff_pix_rel_time_stddev']:
                        pwrs[key].extend([numpixels * [np.nan]])



        # Now process the muon files (one per subrun, containing one entry per ring):

        empty_files = 0

        contained_mu_wholerun = None
        num_contained_mu_rings_in_run = 0

        for subrun in subruns:
            mufile = Path(args.muons_dir,
                          f'muons_LST-1.Run{runnumber:05d}.'
                          f'{subrun:04d}.fits')
            dat = None
            try:
                dat = Table.read(mufile, format='fits')
            except Exception:
                log.warning(f'File {mufile} not found - going on')
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
                continue

            df_muons = dat.to_pandas()

            # contained and clean muon rings:
            contained_mu = df_muons[(df_muons['ring_containment'] > 0.99) &
                                    (df_muons['size_outside'] < 1) &
                                    (df_muons['muon_efficiency'] < 1)]

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

            if contained_mu_wholerun is None:
                contained_mu_wholerun = contained_mu
            else:
                contained_mu_wholerun = pd.concat([contained_mu_wholerun,
                                                   contained_mu],
                                                  ignore_index=True)

        if empty_files > 0:
            log.warning(f'Run {runnumber:d} had {empty_files:d} subruns with '
                        f'no valid muon rings!')

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
            runsummary['mu_hg_peak_sample_mean']. \
                extend([contained_mu_wholerun['hg_peak_sample'].mean()])
            runsummary['mu_hg_peak_sample_stddev']. \
                extend([contained_mu_wholerun['hg_peak_sample'].std()])
        else:
            runsummary['num_contained_mu_rings'].extend([np.nan])
            runsummary['mu_effi_mean'].extend([np.nan])
            runsummary['mu_effi_stddev'].extend([np.nan])
            runsummary['mu_width_mean'].extend([np.nan])
            runsummary['mu_width_stddev'].extend([np.nan])
            runsummary['mu_intensity_mean'].extend([np.nan])
            runsummary['mu_hg_peak_sample_mean'].extend([np.nan])
            runsummary['mu_hg_peak_sample_stddev'].extend([np.nan])

    pd.DataFrame(runsummary).to_hdf(output_file_name, key='runsummary',
                                    mode='w', format='table',
                                    data_columns=runsummary.keys())

    # Now write the pixel-wise run summary info:
    with tables.open_file(output_file_name, mode="a") as h5file:
        for pwrs, name in zip([pixwise_runsummary, pixwise_runsummary_no_stars],
                              ['pixwise_runsummary',
                               'pixwise_runsummary_no_stars']):
            table = h5file.create_table('/', name, pixwise_info)
            row = table.row
            for i in range(len(pwrs['ff_pix_charge_mean'])):
                # we add run number and time info also to this pixwise table:
                row['runnumber'] = runsummary['runnumber'][i]
                row['time'] = runsummary['time'][i]
                for key in pwrs:
                    row[key] = pwrs[key][i]
                row.append()
            table.flush()

    # Finally the tables with info by event type:
    for d, name in zip(dicts, ['cosmics', 'pedestals', 'flatfield']):
        pd.DataFrame(d).to_hdf(output_file_name, key=name, mode='a',
                               format='table', data_columns=d.keys())

    log.info('_________________________________________________')
    log.info('')
    log.info('WARNINGS relative to the quality of the data:')
    log.info('')
    plot(output_file_name, args.batch)


def plot(filename='longterm_dl1_check.h5', batch=False, tel_id=1):
    # Some data needed as reference, to verify the validity of the measured
    # values. Some of these are just typical values obtained from the actual
    # observations between ~ July 2020 and August 20201

    deadtime_per_event = 7e-6  # s
    interleaved_rate = np.array([50, 100])  # Hz
    interleaved_rate_change_run = np.array([0, 2709])
    # run number in which interleaved rates switched to the values above.
    # Tolerances below are relative, w.r.t. expectation:
    interleaved_rate_tolerance = 0.05
    # Muon rates
    mu_a = 1.02
    mu_b = 2.180  # reference muon rate = mu_a + mu_b * cos(zenith)
    muon_rate_tolerance = 0.15

    # cosmics rates
    cosmics_a = 3065.64
    cosmics_b = 3403.23  # reference cosmics rate = mu_b + mu_b * cos(zenith)
    cosmics_rate_tolerance = 0.30

    # Fraction of processed runs in which a pixel must be beyond tolerances
    # in order to be reported:
    run_fraction = 0.5

    # Standard deviation of pedestal charge (to detect noisy pixels)
    ped_max_charge_stddev = 2.5

    # Flat field pixel charges (just for the mean; the std dev may be strongly
    # affected by stars), in pe.
    ff_charge = 73.5
    ff_charge_tolerance = 0.1  # relative tolerance

    # Limits of FF mean pixel time (ns) w.r.t. camera average:
    ff_min_rel_time = -0.3
    ff_max_rel_time = 0.3

    # Flat field std dev of pixel time relative to camera mean (ns):
    ff_max_rel_time_stdev = 0.5

    # Reference rate of >30 pe pulses in cosmics, parametrization vs. pixel
    # id & zenith:
    # (r30[0] + r30[1] * cos(zenith)) + r30[2] * pixel_index  events /
    # second
    r30 = [1.3733, 4.5928, -6.3577e-04]
    rate_above_30_tolerance = 0.25
    pixel_index_limit = 1729
    # The lower tolerance for the >30 pe pulse rate will be halved for 
    # pixels with id >= pixel_index_limit ("corner pixels" which indeed have
    # lower rates just due to geometrical reasons)
    
    # Minimum muon ring intensity (pe)
    min_muon_intensity = 1960

    # global peak HG sample id range in muon ring events:
    muon_peak_hg_sample_range = (8, 18)

    # maximum muon ring width:
    max_muon_ring_width = 0.08  # deg

    # minimum telescope efficiency as derived from muons:
    min_muon_efficiency = 0.16

    # maximum camera averages:
    max_fraction_surviving_pedestals = 0.05
    # ff charge & time, using some of the individual pixels' limits:
    max_average_ff_charge = ff_charge * (1 + ff_charge_tolerance)
    min_average_ff_charge = ff_charge * (1 - ff_charge_tolerance)
    max_average_ff_rel_time_stdev = ff_max_rel_time_stdev

    max_average_ff_time_stdev = 1.2
    min_average_ff_time = 12  # ns
    max_average_ff_time = 23  # ns
    max_average_ff_charge_stdev = 11  # pe

    camgeom = load_camera_geometry()
    engineering_geom = camgeom.transform_to(EngineeringCameraFrame())

    bokeh_output_file(Path(filename).with_suffix('.html'),
                      title='LST1 long-term DL1 data check')

    pixwise_runsummary = read_table(filename, '/pixwise_runsummary')
    pixwise_runsummary_no_stars = read_table(filename,
                                             '/pixwise_runsummary_no_stars')
    run_titles = []
    for i, run in enumerate(pixwise_runsummary['runnumber']):
        date = pd.to_datetime(pixwise_runsummary['time'][i],
                              origin='unix', unit='s')
        run_titles.append('Run {0:05d}, {date}'. \
                          format(run,
                                 date=date.strftime("%b %d %Y %H:%M:%S")))

    runsummary = read_table(filename, '/runsummary/table').to_pandas()
    # avoid issues with nans in bokeh (fill 0's instead):
    runsummary.fillna(0, inplace=True)

    if np.sum(runsummary['num_ucts_jumps']) > 0:
        log.info('Attention: UCTS jumps were detected and corrected:')
        for run, njumps in zip(runsummary['runnumber'],
                               runsummary['num_ucts_jumps']):
            if njumps == 0:
              continue
            log.info(f'   Run {run}: {njumps}  jumps')
        log.info('')

    ped_rate = runsummary['num_pedestals'] / runsummary['elapsed_time']
    err_ped_rate = (np.sqrt(runsummary['num_pedestals']) /
                    runsummary['elapsed_time'])
    ff_rate = runsummary['num_flatfield'] / runsummary['elapsed_time']
    err_ff_rate = (np.sqrt(runsummary['num_flatfield']) /
                   runsummary['elapsed_time'])
    cosmics_rate = runsummary['num_cosmics'] / runsummary['elapsed_time']
    err_cosmics_rate = (np.sqrt(runsummary['num_cosmics']) /
                        runsummary['elapsed_time'])
    total_rate = ped_rate + ff_rate + cosmics_rate
    deadtime_fraction = np.array(total_rate) * deadtime_per_event

    # Find expected rate of interleaveds, from the nominal rate and the deadtime
    nominal_interleaved_rate = []
    for run in runsummary['runnumber']:
        index = np.argwhere(interleaved_rate_change_run <= run).max()
        nominal_interleaved_rate.append(interleaved_rate[index])
    nominal_interleaved_rate = np.array(nominal_interleaved_rate)
    expected_interleaved_rate = ((1 - deadtime_fraction) *
                                 nominal_interleaved_rate)

    runtime = pd.to_datetime(runsummary['time'], origin='unix', unit='s')

    # Plot interleaved pedestal rates

    page0 = Panel()
    fig_ped_rates = show_graph(x=runtime, y=ped_rate, ey=err_ped_rate,
                               xlabel='date',
                               ylabel='Interleaved pedestals rate (Hz)',
                               xtype='datetime', ytype='linear',
                               point_labels=run_titles,
                               ylowlim=expected_interleaved_rate * (
                                       1 - interleaved_rate_tolerance),
                               yupplim=expected_interleaved_rate * (
                                       1 + interleaved_rate_tolerance))

    fig_ped_rates.y_range = Range1d(nominal_interleaved_rate.min() * 0.8,
                                    nominal_interleaved_rate.max() * 1.1)

    # Plot interleaved flatfield rates
    fig_ff_rates = show_graph(x=runtime, y=ff_rate, ey=err_ff_rate,
                              xlabel='date',
                              ylabel='Interleaved flat field rate (Hz)',
                              xtype='datetime', ytype='linear',
                              point_labels=run_titles,
                              ylowlim=expected_interleaved_rate * (
                                      1 - interleaved_rate_tolerance),
                              yupplim=expected_interleaved_rate * (
                                      1 + interleaved_rate_tolerance))
    fig_ff_rates.y_range = Range1d(nominal_interleaved_rate.min() * 0.8,
                                   nominal_interleaved_rate.max() * 1.1)

    coszenith = np.sin(runsummary['mean_altitude'])
    # empirical expression obtained from 2020-2021 data:
    expected_cosmics_rate = cosmics_a + cosmics_b * coszenith
    fig_cosmic_rates = show_graph(x=runtime, y=cosmics_rate,
                                  ey=err_cosmics_rate,
                                  xlabel='date',
                                  ylabel='Cosmics rate (/s)',
                                  xtype='datetime', ytype='linear',
                                  point_labels=run_titles,
                                  ylowlim=expected_cosmics_rate * (
                                          1 - cosmics_rate_tolerance),
                                  yupplim=expected_cosmics_rate * (
                                          1 + cosmics_rate_tolerance))
    fig_cosmic_rates.y_range = Range1d(0, np.max(expected_cosmics_rate) * 1.4)

    # empirical expression obtained from 2020-2021 data:
    expected_mu_rate = mu_a + mu_b * coszenith
    fig_muring_rates = show_graph(x=runtime,
                                  y=runsummary['num_contained_mu_rings'] /
                                    runsummary['elapsed_time'],
                                  ey=np.sqrt(runsummary[
                                                 'num_contained_mu_rings']) /
                                     runsummary['elapsed_time'],
                                  xlabel='date',
                                  ylabel='Contained mu-rings rate (/s)',
                                  xtype='datetime', ytype='linear',
                                  point_labels=run_titles,
                                  ylowlim=expected_mu_rate * (
                                          1 - muon_rate_tolerance),
                                  yupplim=expected_mu_rate * (
                                          1 + muon_rate_tolerance))
    if fig_muring_rates:
        fig_muring_rates.y_range = Range1d(0, expected_mu_rate.max() * 1.25)

    pad_width = 550
    pad_height = 350
    row1 = [fig_ped_rates, fig_ff_rates]
    row2 = [fig_cosmic_rates, fig_muring_rates]
    grid0 = gridplot([row1, row2], sizing_mode=None, width=pad_width,
                     height=pad_height)
    page0.child = grid0
    page0.title = 'Event rates'


    page0a = Panel()

    num_events = runsummary['num_cosmics'] + runsummary['num_pedestals'] + \
                 runsummary['num_flatfield']
    fig_ucts1 = show_graph(x=runtime,
                           y=runsummary['num_unknown_ucts_trigger_tags'] /
                             num_events,
                           xlabel='date',
                           ylabel='Fraction of UNKNOWN UCTS trigger tags',
                           size=4, xtype='datetime', ytype='linear',
                           point_labels=run_titles)
    fig_ucts2 = show_graph(x=runtime, y=runsummary['num_ucts_jumps'],
                           xlabel='date',
                           ylabel='Number of corrected UCTS jumps',
                           size=4, xtype='datetime', ytype='linear',
                           point_labels=run_titles)
    row1 = [fig_ucts1, fig_ucts2]
    fig_tib = show_graph(x=runtime,
                         y=runsummary['num_unknown_tib_trigger_tags'] /
                           num_events,
                         xlabel='date',
                         ylabel='Fraction of UNKNOWN TIB trigger tags',
                         size=4, xtype='datetime', ytype='linear',
                         point_labels=run_titles)
    row2 = [fig_tib]
    grid0a = gridplot([row1, row2], sizing_mode=None,
                      width=pad_width, height=pad_height)
    page0a.child = grid0a
    page0a.title = 'Trigger tags'



    page0b = Panel()
    items = []
    for trigtype in ['ucts', 'tib']:
        for evttype in ['pedestals', 'flatfield', 'cosmics']:
            wrong_fraction = (
                    runsummary['num_wrong_' + trigtype + '_tags_in_' + evttype] /
                    runsummary['num_' + evttype])
            fig = show_graph(x=runtime, y=wrong_fraction,
                             xlabel='date',
                             ylabel='Fraction of ' + evttype + ' with wrong ' + \
                                    trigtype + ' trigger types',
                             size=4,
                             xtype='datetime', ytype='linear',
                             point_labels=run_titles)
            items.append(fig)
    pad_width = 550
    pad_height = 350
    row1 = items[:3]
    row2 = items[3:]

    grid0b = gridplot([row1, row2], sizing_mode=None,
                      width=pad_width, height=pad_height)
    page0b.child = grid0b
    page0b.title = 'Trigger tags'

    page0c = Panel()
    altmin = np.rad2deg(runsummary['min_altitude'])
    altmean = np.rad2deg(runsummary['mean_altitude'])
    altmax = np.rad2deg(runsummary['max_altitude'])
    fig_altitude = show_graph(x=pd.to_datetime(runsummary['time'],
                                               origin='unix', unit='s'),
                              y=altmean,
                              xlabel='date',
                              ylabel='Telescope altitude (mean, min, '
                                     'max) (deg)',
                              eylow=altmean - altmin, eyhigh=altmax - altmean,
                              xtype='datetime', ytype='linear',
                              point_labels=run_titles)
    fig_altitude.y_range = Range1d(altmin.min() * 0.95, altmax.max() * 1.05)

    fig_azimuth = show_graph(x=pd.to_datetime(runsummary['time'],
                                              origin='unix', unit='s'),
                              y=np.rad2deg(runsummary['mean_azimuth']),
                              xlabel='date',
                              ylabel='Telescope mean azimuth (deg)',
                              xtype='datetime', ytype='linear',
                              point_labels=run_titles, size=4)

    fig_ra = show_graph(x=pd.to_datetime(runsummary['time'],
                                         origin='unix', unit='s'),
                        y=runsummary['mean_ra'],
                        xlabel='date',
                        ylabel='Telescope mean Right Ascension (deg)',
                        xtype='datetime', ytype='linear',
                        point_labels=run_titles, size=4)
    fig_dec = show_graph(x=pd.to_datetime(runsummary['time'],
                                          origin='unix', unit='s'),
                         y=runsummary['mean_dec'],
                         xlabel='date',
                         ylabel='Telescope mean declination (deg)',
                         xtype='datetime', ytype='linear',
                         point_labels=run_titles, size=4)

    row1 = [fig_altitude, fig_azimuth]
    row2 = [fig_dec, fig_ra]
    grid0c = gridplot([row1, row2], sizing_mode=None, width=pad_width,
                      height=pad_height)
    page0c.child = grid0c
    page0c.title = 'Pointing'

    # Lists to store arrays containing the number of runs in which a pixel
    # exceeds a limit in any of the checked quantities:
    pixel_problems = []
    check_name = []

    page1 = Panel()
    pad_width = 350
    pad_height = 370

    mean = np.array(pixwise_runsummary['ped_pix_charge_mean'])
    stddev = np.array(pixwise_runsummary['ped_pix_charge_stddev'])

    row1 = show_camera(mean, engineering_geom, pad_width,
                       'Pedestals mean charge', run_titles)
    row2 = show_camera(stddev, engineering_geom, pad_width,
                       'Pedestals charge std dev', run_titles,
                       content_upplim=ped_max_charge_stddev)
    stddev_no_stars = np.array(pixwise_runsummary_no_stars[
                                   'ped_pix_charge_stddev'])
    _, too_high = pixel_report('Pedestal standard deviation', stddev_no_stars,
                               0, ped_max_charge_stddev, run_fraction)
    pixel_problems.append(too_high)
    check_name.append("Too high pedestal charge std dev")

    grid1 = gridplot([row1, row2], sizing_mode='scale_height',
                     height=pad_height)
    page1.child = grid1
    page1.title = 'Interleaved pedestals'

    page2 = Panel()

    mean = np.array(pixwise_runsummary['ff_pix_charge_mean'])
    stddev = np.array(pixwise_runsummary['ff_pix_charge_stddev'])

    row1 = show_camera(mean, engineering_geom, pad_width,
                       'Flat-Field mean charge (pe)', run_titles,
                       display_range=[0, 100],
                       content_lowlim=ff_charge * (1 - ff_charge_tolerance),
                       content_upplim=ff_charge * (1 + ff_charge_tolerance))

    # For the check of whether the values are  in the allowed range we use
    # those calculated excluding the subruns in which a given pixel was close
    # to any bright star:
    mean_no_stars = np.array(pixwise_runsummary_no_stars['ff_pix_charge_mean'])
    too_low, too_high = pixel_report('Flat-Field mean charge', mean_no_stars,
                                     ff_charge * (1 - ff_charge_tolerance),
                                     ff_charge * (1 + ff_charge_tolerance),
                                     run_fraction)
    pixel_problems.extend([too_low, too_high])
    check_name.extend(["Too low flatfield mean charge",
                       "Too high flatfield mean charge"])

    row2 = show_camera(stddev, engineering_geom, pad_width,
                       'Flat-Field charge std dev (pe)',
                       run_titles,
                       display_range=[0, 14])
    grid2 = gridplot([row1, row2], sizing_mode='scale_height',
                     height=pad_height)

    page2.child = grid2
    page2.title = 'Interleaved flat field, charge'

    page3 = Panel()

    mean = np.array(pixwise_runsummary['ff_pix_rel_time_mean'])
    stddev = np.array(pixwise_runsummary['ff_pix_rel_time_stddev'])

    row1 = show_camera(mean, engineering_geom, pad_width,
                       'Flat-Field mean relative time (ns)',
                       run_titles, showlog=False, display_range=[-1, 1],
                       content_lowlim=ff_min_rel_time,
                       content_upplim=ff_max_rel_time)

    mean_no_stars = np.array(pixwise_runsummary_no_stars['ff_pix_rel_time_mean'])
    too_low, too_high = pixel_report('Flat-Field mean relative time',
                                     mean_no_stars,
                                     ff_min_rel_time, ff_max_rel_time,
                                     run_fraction)
    pixel_problems.extend([too_low, too_high])
    check_name.extend(["Too low flatfield mean relative time",
                       "Too high flatfield mean relative time"])

    row2 = show_camera(stddev, engineering_geom, pad_width,
                       'Flat-Field rel. time std dev (ns)',
                       run_titles, showlog=False,
                       display_range=[0.2, np.nanmax(np.append(1.1*stddev,
                                                               0.7))],
                       content_upplim=ff_max_rel_time_stdev)
    stddev_no_stars = np.array(pixwise_runsummary_no_stars[
                                   'ff_pix_rel_time_stddev'])
    _, too_high = pixel_report('Flat-Field rel. time std dev',
                               stddev_no_stars, 0, ff_max_rel_time_stdev,
                               run_fraction)
    pixel_problems.append(too_high)
    check_name.append("Too high flatfield relative time std dev")

    grid3 = gridplot([row1, row2], sizing_mode='scale_height',
                     height=pad_height)
    page3.child = grid3
    page3.title = 'Interleaved flat field, time'

    page4 = Panel()

    pulse_rate_above_10 = \
        np.array(pixwise_runsummary['cosmics_pix_fraction_pulses_above10'] *
                 pixwise_runsummary['ncosmics_per_pix'] /
                 pixwise_runsummary['elapsed_time_per_pix'])
    pulse_rate_above_30 = \
        np.array(pixwise_runsummary['cosmics_pix_fraction_pulses_above30'] *
                 pixwise_runsummary['ncosmics_per_pix'] /
                 pixwise_runsummary['elapsed_time_per_pix'])

    reference_rate_above_30 = (np.array(engineering_geom.n_pixels *
                                        [r30[0] + r30[1] * coszenith]).T +
                               np.array(len(runsummary) *
                                        [r30[2] * np.array(engineering_geom.pix_id)])
                               )
    r30_lowlim = reference_rate_above_30 * (1 - rate_above_30_tolerance)
    r30_upplim = reference_rate_above_30 * (1 + rate_above_30_tolerance)
    
    # The outermost pixels, on the camera corners, just because of their position, have 
    # lower values of this rate of >30 pe pulses. We just set a looser lower limit for
    # them, to avoid lots of unnecessary warnings:
    r30_lowlim[:, pixel_index_limit:] = 0.5 * r30_lowlim[:, pixel_index_limit:]

    row1 = show_camera(pulse_rate_above_10, engineering_geom,
                       pad_width,
                       'Cosmics, rate of >10pe pulses (/s)', run_titles,
                       display_range=[0, 150])
    row2 = show_camera(pulse_rate_above_30, engineering_geom,
                       pad_width,
                       'Cosmics, rate of >30pe pulses (/s)', run_titles,
                       display_range=[0, 12],
                       content_lowlim=r30_lowlim, content_upplim=r30_upplim)

    pulse_rate_above_30_no_stars = np.array(
        pixwise_runsummary_no_stars['cosmics_pix_fraction_pulses_above30'] *
        pixwise_runsummary_no_stars['ncosmics_per_pix'] /
        pixwise_runsummary_no_stars['elapsed_time_per_pix'])
    too_low, too_high = pixel_report('Cosmics, rate of >30pe pulses',
                                     pulse_rate_above_30_no_stars,
                                     r30_lowlim, r30_upplim, run_fraction)
    pixel_problems.extend([too_low, too_high])
    check_name.extend(["Too low rate of >30 pe cosmics pulses",
                       "Too high rate of >30 pe cosmics pulses"])

    grid4 = gridplot([row1, row2], sizing_mode='scale_height',
                     height=pad_height)
    page4.child = grid4
    page4.title = 'Cosmics'

    page4b = Panel()

    # For the image centroids distributions we use the table without cuts in
    # pixels with nearby stars, because centroid is not really as pixel-wise
    # quantity; we use pixels only as a convenient way of make a 2d-histogram
    # of centroid positions on the camera
    # We set units of rate:
    cogs = pixwise_runsummary['cosmics_cog_within_pixel'] / \
           pixwise_runsummary['elapsed_time_per_pix']
    row1 = show_camera(cogs, engineering_geom,
                       pad_width,
                       'Cosmics, image cog distribution (/s)', run_titles,
                       display_range=(0,1.1*np.nanmax(cogs)))
    cogs = pixwise_runsummary['cosmics_cog_within_pixel_intensity_gt_200'] / \
           pixwise_runsummary['elapsed_time_per_pix']

    row2 = show_camera(cogs, engineering_geom,
            pad_width,
            'Cosmics, >200 pe image cog distribution (/s)',
            run_titles, display_range=(0, 1.1*np.nanmax(cogs)))

    grid4b = gridplot([row1, row2], sizing_mode='scale_height',
                     height=pad_height)
    page4b.child = grid4b
    page4b.title = 'Cosmics'

    # Now we make a page with a summary of the pixel problems (in what
    # fraction of the runs each pixel showed a given problem)
    page4c = Panel()
    fraction_of_runs = np.array(pixel_problems) / len(runsummary)

    row = show_camera(fraction_of_runs, engineering_geom,
                      pad_width,
                      'Fraction of runs', titles=check_name,
                      showlog=False,
                      display_range=(1e-6, 1.1))
    # We set the minimum to non-zero so pixels without problems appear in grey

    row[0].title = 'issue'
    # show in red pixels with issues in more than half of the runs (for some
    # reason this does not work until one clicks on the z-range slider of one
    # of the plots):
    row[2].value=(1e-6, 0.5)

    grid4c = gridplot([row], height=pad_height)
    page4c.child = grid4c
    page4c.title = 'Pixel problems'

    page5 = Panel()
    pad_width = 550
    pad_height = 350
    fig_mu_effi = show_graph(x=pd.to_datetime(runsummary['time'], origin='unix',
                                              unit='s'),
                             y=runsummary['mu_effi_mean'],
                             xlabel='date',
                             ylabel='telescope efficiency from mu-rings',
                             ey=runsummary['mu_effi_stddev'] / np.sqrt(
                                 runsummary['num_contained_mu_rings']),
                             xtype='datetime', ytype='linear',
                             point_labels=run_titles,
                             ylowlim=min_muon_efficiency)
    if fig_mu_effi:
        fig_mu_effi.y_range = Range1d(0., 0.22)

    fig_mu_width = show_graph(x=pd.to_datetime(runsummary['time'],
                                               origin='unix', unit='s'),
                              y=runsummary['mu_width_mean'],
                              xlabel='date',
                              ylabel='muon ring width (deg)',
                              ey=runsummary['mu_width_stddev'] / np.sqrt(
                                  runsummary['num_contained_mu_rings']),
                              xtype='datetime', ytype='linear',
                              point_labels=run_titles,
                              yupplim=max_muon_ring_width)
    if fig_mu_width:
        fig_mu_width.y_range = Range1d(0., 0.1)

    fig_mu_intensity = show_graph(
        x=pd.to_datetime(runsummary['time'], origin='unix', unit='s'),
        y=runsummary['mu_intensity_mean'], xlabel='date',
        ylabel='mean muon ring intensity (p.e.)',
        xtype='datetime', ytype='linear', point_labels=run_titles,
        ylowlim=min_muon_intensity)
    if fig_mu_intensity:
        fig_mu_intensity.y_range = Range1d(0., 1.1 * np.max(runsummary[
                                                                'mu_intensity_mean']))
    fig_mu_hg_peak = show_graph(
        x=pd.to_datetime(runsummary['time'], origin='unix', unit='s'),
        y=runsummary['mu_hg_peak_sample_mean'], xlabel='date',
        ey=runsummary['mu_hg_peak_sample_stddev'],
        ylabel='HG global peak sample id (mean&RMS)',
        xtype='datetime', ytype='linear', point_labels=run_titles,
        ylowlim=muon_peak_hg_sample_range[0],
        yupplim=muon_peak_hg_sample_range[1]
    )
    if fig_mu_hg_peak:
        fig_mu_hg_peak.y_range = Range1d(0., 38.)
    row1 = [fig_mu_effi, fig_mu_width]
    row2 = [fig_mu_intensity, fig_mu_hg_peak]

    grid5 = gridplot([row1, row2], sizing_mode=None, width=pad_width,
                     height=pad_height)
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
    fig_ped.y_range = Range1d(0., 1.1 * np.max(runsummary['ped_charge_mean']))

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
        Range1d(0., 1.1 * np.max(runsummary['ped_charge_stddev']))

    frac = runsummary['num_pedestals_after_cleaning'] / \
           runsummary['num_pedestals']
    err = np.sqrt(frac * (1 - frac) / runsummary['num_pedestals'])
    fig_ped_clean_fraction = show_graph(
        x=pd.to_datetime(runsummary['time'], origin='unix', unit='s'),
        y=frac, xlabel='date',
        ylabel='Fraction of pedestals surviving cleaning',
        ey=err, xtype='datetime', ytype='linear',
        point_labels=run_titles,
        yupplim=max_fraction_surviving_pedestals)
    fig_ped_clean_fraction.y_range = Range1d(0, 0.12)

    fig_num_stars =  show_graph(
            x=pd.to_datetime(runsummary['time'], origin='unix', unit='s'),
            y=runsummary['mean_number_of_pixels_nearby_stars'],
            xlabel='date',
            ylabel='Mean number of pixels nearby bright stars',
            ey=err, xtype='datetime', ytype='linear',
            point_labels=run_titles)

    row1 = [fig_ped, fig_ped_stddev]
    row2 = [fig_ped_clean_fraction, fig_num_stars]

    grid6 = gridplot([row1, row2], sizing_mode=None, width=pad_width,
                     height=pad_height)
    page6.child = grid6
    page6.title = "Interleaved pedestals, averages"

    page7 = Panel()
    pad_width = 550
    pad_height = 350
    fig_flatfield = show_graph(x=pd.to_datetime(runsummary['time'],
                                                origin='unix',
                                                unit='s'),
                               y=runsummary['ff_charge_mean'],
                               xlabel='date',
                               ylabel='Cam-averaged FF Q (pe/pixel)',
                               ey=runsummary['ff_charge_mean_err'],
                               xtype='datetime', ytype='linear',
                               point_labels=run_titles,
                               ylowlim=min_average_ff_charge,
                               yupplim=max_average_ff_charge)
    fig_flatfield.y_range = Range1d(0., 100)

    fig_ff_stddev = show_graph(x=pd.to_datetime(runsummary['time'],
                                                origin='unix',
                                                unit='s'),
                               y=runsummary['ff_charge_stddev'],
                               xlabel='date',
                               ylabel='Cam-averaged FF Q std '
                                      'dev (pe/pixel)',
                               xtype='datetime', ytype='linear',
                               point_labels=run_titles,
                               yupplim=max_average_ff_charge_stdev)
    fig_ff_stddev.y_range = \
        Range1d(0., 1.1 * np.max(runsummary['ff_charge_stddev']))

    fig_ff_time = show_graph(x=pd.to_datetime(runsummary['time'],
                                              origin='unix',
                                              unit='s'),
                             y=runsummary['ff_time_mean'],
                             xlabel='date',
                             ylabel='Cam-averaged FF time (ns)',
                             ey=runsummary['ff_time_mean_err'],
                             xtype='datetime', ytype='linear',
                             point_labels=run_titles,
                             ylowlim=min_average_ff_time,
                             yupplim=max_average_ff_time)

    fig_ff_time_std = show_graph(x=pd.to_datetime(runsummary['time'],
                                                  origin='unix',
                                                  unit='s'),
                                 y=runsummary['ff_time_stddev'],
                                 xlabel='date',
                                 ylabel='Cam-averaged FF time std dev (ns)',
                                 xtype='datetime', ytype='linear',
                                 point_labels=run_titles,
                                 yupplim=max_average_ff_time_stdev)
    fig_ff_time_std.y_range = Range1d(0, 2)
    fig_ff_rel_time_std = show_graph(x=pd.to_datetime(runsummary['time'],
                                                      origin='unix',
                                                      unit='s'),
                                     y=runsummary['ff_rel_time_stddev'],
                                     xlabel='date',
                                     ylabel='Cam-averaged FF '
                                            'rel. pix t std dev (ns)',
                                     xtype='datetime', ytype='linear',
                                     point_labels=run_titles,
                                     yupplim=max_average_ff_rel_time_stdev)

    fig_ff_rel_time_std.y_range = \
        Range1d(0., np.max([1., runsummary['ff_rel_time_stddev'].max()]))

    row1 = [fig_flatfield, fig_ff_stddev]
    row2 = [fig_ff_time, fig_ff_time_std, fig_ff_rel_time_std]

    grid7 = gridplot([row1, row2], sizing_mode=None, width=pad_width,
                     height=pad_height)
    page7.child = grid7
    page7.title = "Interleaved FF, averages"

    page8 = Panel()
    pad_width = 550
    pad_height = 350

    average_pulse_rate_above_10 = \
        runsummary['cosmics_fraction_pulses_above10'] *  \
        runsummary['num_cosmics'] / runsummary['elapsed_time']
    average_pulse_rate_above_30 = \
        runsummary['cosmics_fraction_pulses_above30'] *  \
        runsummary['num_cosmics'] / runsummary['elapsed_time']

    fig_rate10pe = show_graph(x=pd.to_datetime(runsummary['time'],
                                                origin='unix',
                                                unit='s'),
                              y=average_pulse_rate_above_10,
                              xlabel='date',
                              ylabel='Camera-averaged rate of >10pe pulses (/s)',
                              xtype='datetime', ytype='linear', size=4,
                              point_labels=run_titles)
    fig_rate10pe.y_range = \
        Range1d(0., 1.5 * np.max(average_pulse_rate_above_10))

    # We use the pixel-averaged lower and upper limits to the single pixel rates
    # r30_lowlim.mean(axis=1)  and  r30_upplim.mean(axis=1)
    fig_rate30pe = show_graph(x=pd.to_datetime(runsummary['time'],
                                                origin='unix',
                                                unit='s'),
                              y=average_pulse_rate_above_30,
                              xlabel='date',
                              ylabel='Camera-averaged rate of >30pe pulses (/s)',
                              xtype='datetime', ytype='linear', size=4,
                              point_labels=run_titles,
                              ylowlim=r30_lowlim.mean(axis=1),
                              yupplim=r30_upplim.mean(axis=1))
    fig_rate30pe.y_range = \
        Range1d(0., 1.5 * np.max(average_pulse_rate_above_30))

    row1 = [fig_rate10pe, fig_rate30pe]
    grid8 = gridplot([row1], sizing_mode=None, width=pad_width,
                     height=pad_height)
    page8.child = grid8
    page8.title = "Cosmics, averages"

    tabs = Tabs(tabs=[page0, page0a, page0b, page0c, page1, page2,
                      page3, page4, page4b, page4c, page5, page6,
                      page7, page8])

    if batch:
        save(column(Div(text='<h1> Long-term DL1 data check </h1>'), tabs))
    else:
        show(column(Div(text='<h1> Long-term DL1 data check </h1>'), tabs))


def show_graph(x, y, xlabel, ylabel, ey=None, eylow=None, eyhigh=None,
               xtype='linear', ytype='linear', size=2,
               point_labels=None, ylowlim=None, yupplim=None):
    '''
    Function to display a simple "y vs. x" graph, with y error bars
    It also checks limits of the values (if provided) and writes out a text
    log reporting outliers

    Parameters
    ----------
    x: ndarray, x coordinates
    y: ndarray, y coordinates
    ey: ndarray, size of y error bars
    eylow, eyhigh: ndarrays, size of lower- and upper-side y-error bars (if
                   provided, they are used instead of ey)
    xlabel: x-axis label
    ylabel: y-axis label
    xtype: 'log', 'linear', 'datetime'
    ytype: 'log', 'linear', 'datetime'
    point_labels: one label per point, to be displayed when mouse overs near point
    ylowlim, yupplim: ndarrays, min and max acceptable values of the plotted
    quantity, same dimension as x and y

    Returns
    -------
    A bokeh.plotting.figure with the y vs. x graph
    '''

    # Nothing to plot if all nans (like e.g. for muon plots when no muon info
    # was read)
    if np.all(np.isnan(x)) or np.all(np.isnan(y)):
        return None

    fig = figure(background_fill_color='#ffffff', x_axis_label=xlabel,
                 x_axis_type=xtype, y_axis_type=ytype, y_axis_label=ylabel)

    source = ColumnDataSource(data=dict(x=x, y=y))
    if point_labels is not None:
        source.data['point_labels'] = point_labels
    datapoints = fig.circle(x='x', y='y', size=size, source=source)

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
                                renderers=[datapoints],
                                mode='mouse',
                                point_policy='snap_to_data'))

    if ylowlim is not None or yupplim is not None:
        log.info(f'Anomalies in {ylabel}:')

    too_high = np.array(len(y) * [False])
    too_low = np.array(len(y) * [False])
    if ylowlim is not None:
        fig.line(x=x, y=ylowlim, line_dash='dashed', color='orange',
                 line_width=2)
        too_low |= (y < ylowlim)
    if yupplim is not None:
        fig.line(x=x, y=yupplim, line_dash='dashed', color='red')
        too_high |= (y > yupplim)

    bad_runs = too_low | too_high

    if bad_runs.sum() > 0:
        for runlabel, val, low in zip(np.array(point_labels)[bad_runs],
                                      y[bad_runs], too_low[bad_runs]):
            log.info(f'    {runlabel}:')
            tag = '(too high)'
            if low:
                tag = '(too low)'
            log.info(f'       {ylabel}: {val:.2f} {tag}')
            log.info('')

    elif ylowlim is not None or yupplim is not None:
        log.info('    None')
        log.info('')

    return fig


def pixel_report(title, value, low_limit, upp_limit, run_fraction):
    '''
    Parameters
    ----------
    title: describes que quantity store in value
    value:  ndarray num_runs * num_pixels, run- and pixel-wise quantity
    low_limit: scalar or ndarray num_runs * num_pixels. Below this, value is
    not considered healthy
    upp_limit: if larger than this, value is not considered healthy. Same
    types as above.
    run_fraction: minimum fraction of runs in which a pixel must be
    unhealthy to report it

    Returns
    -------
    too_low_count, too_high_count: ndarrays [num_pixels]  Number of runs in
    which each pixel was below low_limit and above upp_limit respectively

    '''

    npixels = value.shape[1]

    # maximum fraction of faulty pixels that will be reported individually.
    # Beyond that, just a generic warning is displayed:
    max_fraction_for_detailed_warning = 0.05

    too_low = value < low_limit
    too_high = value > upp_limit
    # In how many of the processed runs is each pixel faulty?:
    too_low_count = np.sum(too_low, axis=0)  # [n_pixels]
    too_high_count = np.sum(too_high, axis=0)
    log.info(f'{title}, anomalous pixels:')
    num_runs = value.shape[0]

    too_low_pix_ids = np.flatnonzero(too_low_count > run_fraction * num_runs)
    too_high_pix_ids = np.flatnonzero(too_high_count > run_fraction * num_runs)

    if len(too_low_pix_ids) < max_fraction_for_detailed_warning * npixels:
        for pix_id in too_low_pix_ids:
            log.info(f'id {pix_id} ({get_pixel_location(pix_id)}) too low in'
                     f' {too_low_count[pix_id]} of {num_runs} runs')
    else:
        log.info(f'More than {int(100 * max_fraction_for_detailed_warning)}% of '
                 f'the pixels are too low in more than {int(run_fraction * 100)}%'
                 f'of the runs!')

    if len(too_high_pix_ids) < max_fraction_for_detailed_warning * npixels:
        for pix_id in too_high_pix_ids:
            log.info(f'id  {pix_id} ({get_pixel_location(pix_id)}) too high in'
                     f' {too_high_count[pix_id]} of {num_runs} runs')
    else:
        log.info(f'More than {int(100 * max_fraction_for_detailed_warning)}% of '
                 f'the pixels are too high in more than '
                 f'{int(run_fraction * 100)}% of the runs!')

    log.info('')

    return too_low_count, too_high_count

def get_datacheck_table(filename, tablename, exclude_stars=False):
    """

    Parameters
    ----------
    filename: str, datacheck_dl1_LST-1.RunXXXXX.h5 full-run datacheck file
    tablename: str, "pedestals" "cosmics" or "flatfield"
    exclude_stars: set to nan all pixel values for subruns (table rows) in
    which the given pixel had stars nearby, according to colum num_nearby_stars

    Returns
    -------
    table: astropy table

    """

    table = read_table(filename, f'/dl1datacheck/{tablename}')

    nstars = table['num_nearby_stars']
    npixels = nstars.shape[1]
    nevents = table['num_events']
    elapsed_time = table['elapsed_time']

    # Add two columns containing the subrun-wise and pixel-wise elapsed time
    # and number of events. Pixel-wise is needed in case we apply the
    # exclusion of pixels with nearby stars:
    nevents_per_pix = np.transpose([nevents] * npixels)
    elapsed_time_per_pix = np.transpose([elapsed_time] * npixels)
    table['nevents_per_pix'] = nevents_per_pix
    table['elapsed_time_per_pix'] = elapsed_time_per_pix

    if not exclude_stars:
        return table

    # Set to nan pixel entries for which there were nearby stars
    for k in table.keys():
        if k == 'num_nearby_stars':
            continue
        if table[k].shape != table['num_nearby_stars'].shape:
            continue

        # Note: because of the np.nan the line below converts the column to
        # float, but there is no problem with that (but it is more convenient,
        # w.r.t. using a masked array, for representing the values e.g. in
        # camera displays, where masked entries from a masked array appear for
        # some reason as zeros, which is obviously misleading).
        table[k] = np.where(nstars > 0, np.nan, table[k])

    return table

def pix_subrun_mean_to_run_mean(means, events):
    """
    Convert per-pixel subrun-wise mean values to run-wise mean values 
    Exclude from the means the subruns with inf and nan values

    Parameters
    ----------
    means [nsubruns, npixels]  pixel-wise mean of quantity per subrun
    events [nsubruns, npixels] number of events  per pixel and subrun used in
    the calculation

    Returns
    -------
    [npixels] pixel-wise mean for the whole run, calculated from the subrun
    means

    """

    allok = np.isfinite(means)
    okmeans = np.where(allok, means, np.nan)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nansum(okmeans * events, axis=0) / np.nansum(events, axis=0)

def pix_subrun_std_to_run_std(stds, events):
    """
    Convert per-pixel subrun-wise std dev values to run-wise std dev values 
    Exclude from the means the subruns with inf and nan values

    Parameters
    ----------
    stds [nsubruns, npixels]  pixel-wise std dev of quantity per subrun
    events [subruns, pixels] number of events  per pixel and subrun used in
    the calculation

    Returns
    -------
    [npixels] pixel-wise std dev for the whole run, calculated from the subrun
    std devs

    """

    allok = np.isfinite(stds)
    okstds = np.where(allok, stds, np.nan)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return (np.nansum(okstds**2 * events, axis=0) /
                np.nansum(events, axis=0)) ** 0.5


def trigtag_mismatches(table, tag_value):
    """

    Parameters
    ----------
    table: one events table of the datacheck_dl1*h5 file (cosmics, pedestals
    or flatfield)
    tag_value: a trigger type value for comparing it with the ones in
    trigger_type and ucts_trigger_type

    Returns
    -------
    num_wrong_tags [ndarray, ndarray] each of the arrays with same
    number of elements as number of rows in the table (i.e. = number of subruns)
    The arrays contain the number of non-matching tags, for ucts and tib
    respectively

    """

    num_wrong_tags = [np.zeros(len(table)), np.zeros(len(table))]
    # ^^^  to count [ucts, tib] in each subrun
    for k, type in enumerate(['ucts_trigger_type', 'trigger_type']):
        for j, subrun_trigger_statistics in enumerate(table[type]):
            for trigtype in subrun_trigger_statistics:
                # trigtype is an array of two elements: trigtype[0] (when
                # not =0) is the trigger type, and trigtype[1] the
                # number of events in the subrun which have that
                # trigger type.

                # skip the 'unknown' cases, we are counting here just the
                # wrong tags:
                if trigtype[0] == 0: # TriggerBits.UNKNOWN:
                    continue

                if trigtype[1] == 0:
                    # no more trigger types are stored for this subrun
                    break
                elif (trigtype[0] & tag_value) == 0:
                    num_wrong_tags[k][j] += trigtype[1]

    return num_wrong_tags


if __name__ == '__main__':
    main()
