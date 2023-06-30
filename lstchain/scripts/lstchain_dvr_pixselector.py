#!/usr/bin/env python

"""
This script uses DL1 files to determine for each event which pixels
contain relevant information and should hence be kept when reducing the raw
data volume.

We need to change the main parameter of the data reduction (a pixel selection
threshold, called "min_charge_for_certain_selection") depending on the NSB
level (otherwise high NSB data would not be reduced at all).
However, since we do not want to use different threshold for the subruns of a
given run (for reasons of stability & simplicity of analysis), we cannot
decide the threshold subrun by subrun.

The approach is the following: we run first the script over *all* subruns of a
run, for example:

lstchain_dvr_pixselector -f "/.../dl1_LST-1.Run12469.????.h5"

(the default "action" compute_dvr_settings will be chosen by default, and the
default parameters will be used for "picture_threshold" and "number_of_rings" -
see below)
This creates in the output directory (which is the current by default) a file
DVR_settings_LST-1.Run12469.h5 which contains a table "run_summary" which
includes the DVR algorithm parameters determined for each subrun.

Then we run again the script over all subruns, and using the option to create
the pixel maks:

lstchain_dvr_pixselector -f "/.../dl1_LST-1.Run12469.????.h5" --action create_pixel_masks

The script will detect that the file DVR_settings_LST-1.Run12469.h5 already
exists, read it, and use, as threshold for DVR, the average for all subruns,
in p.e., rounded to the closest lower integer (we do not want to have too many
different ways of reducing the data, so we "discretize" the threshold in
1-p.e. steps). Then the event-wise pixel maks (selected pixels) for the subrun
will be computed and written out to a file
Pixel_selection_LST-1.Run12469.xxxx.h5  for each subrun xxxx

Note that when the option --action create_pixel_masks is used, the options
--number-of-rings and --picture-threshold are ignored, since in that
case the DVR settings will be obtained from the previously created DVR_settings
file.

"""

import argparse
from pathlib import Path
import tables
import glob
import os
import numpy as np
import time

from lstchain.paths import parse_dl1_filename
from lstchain.io.io import dl1_params_lstcam_key, dl1_images_lstcam_key
from lstchain.io.io import dl1_params_tel_mon_cal_key

from ctapipe.io import read_table
from ctapipe.core import Container, Field
from ctapipe.io import HDF5TableWriter
from ctapipe.containers import EventType
from ctapipe_io_lst import LSTEventSource

parser = argparse.ArgumentParser(description="DVR pixel selector")

parser.add_argument('-f', '--dl1-files', dest='dl1_files',
                    type=str, default='',
                    help='Input DL1 file names')
parser.add_argument('-o', '--output-dir', dest='output_dir',
                    type=Path, default='./',
                    help='Path to the output directory (default: %(default)s)')
parser.add_argument('-t', '--picture-threshold', dest='picture_threshold',
                    type=float, default=8.,
                    help='Picture threshold (p.e., default: %(default)s)')
parser.add_argument('--action', dest='action', type=str,
                    default='compute_dvr_settings',
                    help='compute_dvr_settings (default) or create_pixel_masks')
parser.add_argument('-n', '--number-of-rings', dest='number_of_rings',
                    type=int, default=1,
                    help='Number of rings around picture pixels (default: %('
                         'default)s)')

class PixelMask(Container):
    event_id = Field(-1, 'event id')
    event_type = Field(-1, 'event_type')
    number_of_pixels_after_standard_cleaning = Field(-1, 'number_of_pixels_after_standard_cleaning')
    highest_removed_charge = Field(np.float32(0.), 'highest_removed_charge')
    pixmask = Field(None, 'selected pixels mask')
    
class RunSummary(Container):
    run_id = Field(-1, 'run_id')
    subrun_id = Field(-1, 'subrun_id')
    elapsed_time = Field(-1, 'elapsed_time') # seconds
    mean_zenith = Field(-1, 'mean_zenith') # degrees
    number_of_events = Field(-1, 'number_of_events')
    picture_threshold = Field(-1, 'picture_threshold') # photo-electrons
    min_charge_for_certain_selection = Field(-1, 'min_charge_for_certain_selection') # photo-electrons
    number_of_rings = Field(-1, 'number_of_rings')
    number_of_muon_candidates = Field(-1, 'number_of_muon_candidates')
    min_pixel_survival_fraction = Field(np.float32(np.nan), 'min_pixel_survival_fraction')
    max_pixel_survival_fraction = Field(np.float32(np.nan), 'max_pixel_survival_fraction')
    mean_pixel_survival_fraction = Field(np.float32(np.nan), 'mean_pixel_survival_fraction')
    fraction_of_full_shower_events = Field(np.float32(np.nan),
                                           'fraction_of_full_shower_events')
    number_of_always_saved_pixels = Field(0, 'number_of_always_saved_pixels')
    ped_mean = Field(np.float32(np.nan), '')  # photo-electrons
    ped_stdev = Field(np.float32(np.nan), '') # photo-electrons
    

def main():
    args = parser.parse_args()
    summary_info = RunSummary()

    write_pixel_masks = False
    if args.action == 'compute_dvr_settings':
        print('I will calculate the Data Volume Reduction parameters from '
              'the input DL1 files, and write them to the output file')
    elif args.action == 'create_pixel_masks':
        write_pixel_masks = True
        print('Option to write pixel masks in output file selected. I will '
              'read in the Data Volume Reduction parameters from the '
              'corresponding DVR_settings_*.h5 file')
        print('(the --picture_threshold and --number-of-rings command-line '
              'options, if provided, will be ignored!)')
    else:
        print('Unknown option in --action flag!')
        return 1

    if args.dl1_files == '':
        print('Please use --dl1-files provide a valid set of DL1 files!')
        return 1

    dl1_files = glob.glob(args.dl1_files)
    dl1_files.sort()

    # The pixel selection will be applied only to "physics triggers"
    # (showers), not to other events like interleaved pedestal and flatfield
    # events:
    event_types_to_be_reduced = [EventType.SUBARRAY.value,
                                 EventType.UNKNOWN.value,
                                 EventType.HARDWARE_STEREO.value,
                                 EventType.DAQ.value]
    # (Note: some of these types are not yet in use). Chances are that pixels
    # labeled as "unknown" are cosmics (=showers), so we include them among the
    # types to be reduced.

    # For the record, keep the input "picture threshold".
    # Note that it is the value that will be used for the pixel selection
    # algorithm, ONLY IF it results in an average fraction of surviving
    # pixels (see below) smaller than max_pix_survival_fraction, defined below.
    # If it allows a larger fraction of pixels to survive.
    summary_info.picture_threshold = args.picture_threshold

    # The threshold charge for pixel selection, called
    # min_charge_for_certain_selection is equal to args.picture_threshold by
    # default, but it will be modified, if a too high average fraction
    # (> max_pix_survival_fraction) of pixels (and FIRST neighbors) survive,
    # in shower events, with that picture threshold (this is to adapt the
    # selection to higher NSB levels). The survival fraction is computed
    # excluding the 10% noisiest pixels in the camera, because we want it to
    # be stable when e.g. stars get in and out of the camera, or when
    # changing wobbles (star field changes).
    max_pix_survival_fraction = 0.1
    # Note that in this calculation of survival fraction we use one ring,
    # no matter what the value of args.number_of_rings (see below) is. This
    # is because we want the pixels saved using args.number_of_rings=N
    # to be a superset of those saved when using N-1 rings (otherwise the
    # "adding one ring" operation becomes a complicated and hard to interpret
    # one!)


    # Number of "rings" of pixels to be finally kept around pixels which are
    # above min_charge_for_certain_selection
    number_of_rings = args.number_of_rings
    summary_info.number_of_rings = number_of_rings

    # Cuts to identify muon rings candidates (in order to save them fully)
    # These are conservative cuts which are fulfilled comfortable by all "good
    # quality" rigs that are actually used for calibration:
    muon_ring_min_intensity = 1000
    muon_ring_min_length = 0.5
    muon_ring_min_n_pixels = 50
    muon_ring_max_t_gradient = 5

    current_run_number = -1

    for dl1_file in dl1_files:
        print()
        print('Input file:', dl1_file)

        run_info = parse_dl1_filename(dl1_file)
        run_id, subrun_id = run_info.run, run_info.subrun
        summary_info.run_id = run_id
        summary_info.subrun_id = subrun_id

        output_dir = args.output_dir.absolute()
        output_dir.mkdir(exist_ok=True, parents=True)

        # The file DVR_settings_LST-1.Run*.h5 will be needed in case the
        # --action create_pixel_masks  option is used:
        input_dvr_settings_file = Path(output_dir,
                                       f'DVR_settings_LST-1.Run{run_id:05d}.h5')
        dvr_settings = None

        # If we are just calculating the Data Volume Reduction settings, we
        # write a single file for the whole run:
        output_file = Path(output_dir, f'DVR_settings_LST-1.Run{run_id:05d}.h5')

        # On the other hand, for writing the pixels masks we will create
        # subrun-wise files:
        if write_pixel_masks:
            output_file = Path(output_dir, f'Pixel_selection_LST-1.Run'
                                           f'{run_id:05d}.{subrun_id:04d}.h5')

            if not os.path.isfile(input_dvr_settings_file):
                print('ERROR: --action create_pixel_masks selected, but file ' +
                      input_dvr_settings_file.name + 'does not exist!')
                print('You must run first this script over all the subruns of '
                      'the run in one go, using --action compute_dvr_settings')
                exit(1)

            dvr_settings = read_table(input_dvr_settings_file, '/run_summary')
            # We just take the values below from the first table row, because by
            # construction they are all identical. These are the settings
            # used in the execution that resulted in the creation of the
            # DVR_settings_LST-1.Run*.h5 file:
            summary_info.number_of_rings = dvr_settings['number_of_rings'][0]
            summary_info.picture_threshold = dvr_settings['picture_threshold'][0]

        print('Output file:', output_file)

        data_parameters = read_table(dl1_file, dl1_params_lstcam_key)

        # Get the LST camera geometry:
        subarray_info = LSTEventSource.create_subarray(tel_id=1)
        camera_geom = subarray_info.tel[1].camera.geometry
      
        # Time between first and last timestamp:
        summary_info.elapsed_time = (data_parameters['dragon_time'][-1] -
                                     data_parameters['dragon_time'][0])

        summary_info.mean_zenith = np.round(90 -
                                            np.rad2deg(data_parameters['alt_tel'].mean()), 2)

        number_of_events = len(data_parameters)
        summary_info.number_of_events = number_of_events

        event_type_data = data_parameters['event_type'].data
        # event_type: physics trigger (32) interleaved flat-field(0) pedestal (2),
        # unknown(255)  are those currently in use in LST1

        found_event_types = np.unique(event_type_data, return_counts=True)
        print('Event types found:', found_event_types[0])
        print('Number of events of each type:', found_event_types[1])

        if not np.any(np.isin(found_event_types[0], 
                              event_types_to_be_reduced)):
            print('No reducible events were found in file! SKIPPING IT!')
            continue

        cosmic_mask   = event_type_data == EventType.SUBARRAY.value  # showers
        unknown_mask = event_type_data == EventType.UNKNOWN.value
        pedestal_mask = event_type_data == EventType.SKY_PEDESTAL.value

        # In some pathological runs (e.g. 1877) very few events are labeled
        # as cosmics, most get the tag "UNKNOWN" instead. Statistically,
        # chances are that they are actually cosmics. If for a run we spot 
        # that more than half of the events are labeled as UNKNOWN, we assume 
        # they are cosmics:
        if unknown_mask.sum() > 0.5 * len(event_type_data):
            print('Too many events tagged UNKNOWN! I will assume they are cosmics!')
            cosmic_mask |= unknown_mask

        data_images = read_table(dl1_file, dl1_images_lstcam_key)

        data_calib = read_table(dl1_file, dl1_params_tel_mon_cal_key)
        # data_calib['unusable_pixels'] , indices: (Gain  Calib_id  Pixel)

        # Get the "unusable" flags from the pedcal file:
        unusable_HG = data_calib['unusable_pixels'][0][0]
        unusable_LG = data_calib['unusable_pixels'][0][1]

        #  Keep pixels that could not be properly calibrated: save them always,
        #  in case their calibration can be later fixed:
        keep_always_mask = (unusable_HG | unusable_LG)
        summary_info.number_of_always_saved_pixels = keep_always_mask.sum()

        if summary_info.number_of_always_saved_pixels > 0:
            print(f'Pixels that will always be saved ('
                  f'{summary_info.number_of_always_saved_pixels}):')
            for pixindex in np.where(keep_always_mask)[0]:
                print(pixindex)
        else:
            print('Pixels that will always be saved: None')

        charges_data = data_images['image']
        # times_data = data_images['peak_time'] # not used now
        image_mask = data_images['image_mask']

        print("Original standard cleaning, pixel survival probabilities:")
        print('  Minimum: ', np.round(np.mean(image_mask, axis=0).min(), 5))
        print('  Maximum: ', np.round(np.mean(image_mask, axis=0).max(), 5))
        print('  Mean: ', np.round(np.mean(image_mask, axis=0).mean(), 5))

        charges_cosmics = charges_data[cosmic_mask]
        if not write_pixel_masks:
            # Calculate an adequate data volume reduction pixel threshold for
            # this subrun:
            min_charge_for_certain_selection = find_DVR_threshold(charges_cosmics,
                                                                  max_pix_survival_fraction,
                                                                  args.picture_threshold,
                                                                  camera_geom)
        else:
            runmask = dvr_settings['run_id'] == run_id
            meanq = dvr_settings['min_charge_for_certain_selection'][runmask].mean()
            min_charge_for_certain_selection  = np.floor(meanq)
            # we do not have to calculate it from scratch, we get the
            # value (same for all subruns) from the DVR_settings_LST*.h5 file,
            # just averaging the subrun-wise values and taking the closest
            # (lower) integer

        summary_info.min_charge_for_certain_selection = min_charge_for_certain_selection

        # Cuts to identify promising muon ring candidates:
        mucan_event_list = np.where(cosmic_mask &
                                    (data_parameters['intensity'] >
                                     muon_ring_min_intensity) &
                                    (data_parameters['length'] >
                                     muon_ring_min_length) &
                                    (data_parameters['n_pixels'] >
                                     muon_ring_min_n_pixels) &
                                    (abs(data_parameters['time_gradient']) <
                                     muon_ring_max_t_gradient))[0]

        mucan_event_id_list = np.array(data_parameters['event_id'][mucan_event_list])
        summary_info.number_of_muon_candidates = len(mucan_event_list)

        num_sel_pixels = []
        selected_pixels_masks = []
        event_ids = []
        highest_removed_charge = []

        full_camera = np.array(camera_geom.n_pixels*[True])

        for charge_map, event_id, event_type in zip(charges_data,
                                                    data_parameters['event_id'],
                                                    event_type_data):
            # keep full camera for muon candidates:
            if event_id in mucan_event_id_list:
                selected_pixels = full_camera
            elif event_type not in event_types_to_be_reduced:
                selected_pixels = full_camera
            else:
                selected_pixels = get_selected_pixels(charge_map,
                                                      min_charge_for_certain_selection,
                                                      number_of_rings,
                                                      camera_geom)

            # Include pixels that must be kept for all events:
            selected_pixels |= keep_always_mask
    
            num_sel_pixels.append(selected_pixels.sum())
            selected_pixels_masks.append(selected_pixels)
    
            event_ids.append(event_id)
            if selected_pixels.sum() < camera_geom.n_pixels:
                highest_removed_charge.append(np.nanmax(charge_map[~selected_pixels]))
            else:
                highest_removed_charge.append(0.)

        selected_pixels_masks = np.array(selected_pixels_masks)

        cr_masks = selected_pixels_masks[cosmic_mask]
        fraction_of_survival = cr_masks.sum() / len(cr_masks.flatten())
        print("Fraction in shower events of selected pixels:", np.round(
                fraction_of_survival, 3))

        num_sel_pixels = np.array(num_sel_pixels)
        print('Average number of selected pixels per event (of any type):',
        np.round(num_sel_pixels.sum() / len(data_parameters), 2))
        print(f'Fraction of whole camera: '
              f'{num_sel_pixels.sum()/len(data_parameters)/camera_geom.n_pixels:.3f}')

        # Keep track of how many cosmic events were fully saved (whole camera)>
        summary_info.fraction_of_full_shower_events = \
            np.round((np.sum(selected_pixels_masks[cosmic_mask],
                             axis=1) == camera_geom.n_pixels).sum() /
                             cosmic_mask.sum(), 5)

        summary_info.ped_mean = np.round(np.mean(charges_data[pedestal_mask]),3)
        summary_info.ped_stdev = np.round(np.std(charges_data[pedestal_mask]), 3)

        pixel_survival_fraction = np.mean(selected_pixels_masks, axis=0)
        summary_info.min_pixel_survival_fraction = np.round(np.min(pixel_survival_fraction), 5)
        summary_info.max_pixel_survival_fraction = np.round(np.max(pixel_survival_fraction), 5)
        summary_info.mean_pixel_survival_fraction = np.round(np.mean(pixel_survival_fraction), 5)

        # In case the system is temporarily unavailable writing of the file
        # may fail. If so, we try again (every minute) until we succeed,
        # and give up after a certain number of attempts:
        number_of_writing_attempts = 0
        while True:
            try:
                number_of_writing_attempts += 1
                writer_conf = tables.Filters(complevel=9, fletcher32=True)
                filemode = 'w'
                if not write_pixel_masks:
                    if run_id != current_run_number:
                        current_run_number = run_id
                    else:
                        filemode == 'a'
                        # In the "compute_dvr_settings" mode (not writing pixel
                        # masks) we will write only one file per run number,
                        # so for every new subrun we just append one row to the
                        # run_summary table.

                with HDF5TableWriter(output_file, filters=writer_conf,
                                     mode=filemode) as writer:
                    writer.write("run_summary", summary_info)

                # In the pixel selection mode we create a new file per subrun,
                # which contains the event-wise pixel masks:
                if write_pixel_masks:
                    data = PixelMask()
                    with HDF5TableWriter(output_file, filters=writer_conf, mode='a') as writer:
                        for evid, evtype, std_clean_npixels, highestQ, pixmask in zip(
                                event_ids, event_type_data, data_parameters['n_pixels'],
                                highest_removed_charge, selected_pixels_masks):
                            data.event_id = evid
                            data.event_type = evtype
                            data.number_of_pixels_after_standard_cleaning = std_clean_npixels
                            data.highest_removed_charge = highestQ
                            data.pixmask = pixmask
                            writer.write("selected_pixels_masks", data)
                break
            except:
                if number_of_writing_attempts > 60:
                    print("I gave up!")
                    exit(1)
                print(time.asctime(time.localtime()),
                      "Could not write output, attempt",
                      number_of_writing_attempts,
                      "... Will try again after 5 minutes")
                time.sleep(300)
                continue

    print()
    print('lstchain_dvr_pixselector finished successfully!')
    print()
    return 0

def get_selected_pixels(charge_map, min_charge_for_certain_selection,
                        number_of_rings, geom,
                        min_npixels_for_full_event=500):
    """
    Function to select the pixels which likely contain a Cherenkov signal

    Parameters
    ----------
    charge_map: ndarray, pixel-wise charges in photo-electrons

    min_charge_for_certain_selection: pixels above this charge will be selected

    number_of_rings: number of "rings" of pixels around the pixels selected by
    their charge that will also be selected (N=1 means just the immediate
    neighbors; N=2 adds the neighbors of neighbors and so on)

    geom: camera geometry

    min_npixels_for_full_event: full camera will be selected for events with
    more than this number of pixels passing the standard selection

    Returns
    -------
    ndarray (boolean): mask containing the selected pixels

    """

    # Proceed with the identification of interesting pixels to be saved.
    # Keep pixels that have a charge above min_charge_for_certain_selection:
    selected_pixels = (charge_map > min_charge_for_certain_selection)

    # Add "number_of_rings" rings of pixels around the already selected ones:
    for ring in range(number_of_rings):
        # we add-up (sum) the selected-pixel-wise map of neighbors, to find
        # those who appear at least once (>0). Those should be added:
        additional_pixels = (np.sum(geom.neighbor_matrix[selected_pixels],
                                    axis=0)>0)
        selected_pixels |= additional_pixels

    # if more than min_npixels_for_full_event were selected, keep whole camera:
    if selected_pixels.sum() > min_npixels_for_full_event:
        selected_pixels = np.array(geom.n_pixels * [True])

    return selected_pixels


def find_DVR_threshold(charges_cosmics, max_pix_survival_fraction,
                       picture_threshold, camera_geom):

    # By default use the provided picture threshold to keep pixels:
    min_charge_for_certain_selection = picture_threshold

    # Check what fraction of pixels in shower events is kept with the current
    # value of min_charge_for_certain_selection (and ONE ring of neighbors)
    # We compute the value excluding the noisiest pixels (e.g. from stars)
    fraction_of_survival = 1.
    target_nevents = 5000  # number of events for the calculation (to speed up!)

    event_jump = int(charges_cosmics.shape[0] / target_nevents) + 1
    charges_cosmics_sampled = charges_cosmics[::event_jump, :]

    while fraction_of_survival > max_pix_survival_fraction:
        selected_pixels_masks = []
        for charge_map in charges_cosmics_sampled:
            selected_pixels = get_selected_pixels(charge_map,
                                                  min_charge_for_certain_selection,
                                                  1, camera_geom)
            selected_pixels_masks.append(selected_pixels)

        selected_pixels_masks = np.array(selected_pixels_masks)
        per_pix_fraction_of_survival = (np.sum(selected_pixels_masks, axis=0) /
                                        selected_pixels_masks.shape[0])
        ninety_percent_dimmest = np.sort(per_pix_fraction_of_survival)[
                                 :int(0.9 * len(per_pix_fraction_of_survival))]

        # Compute the survival fraction using only the 90% dimmer pixels
        # - we do not want to change the thresholds just because of stars,
        # but rather based on the "bulk" of the diffuse NSB
        fraction_of_survival = np.mean(ninety_percent_dimmest)
        if fraction_of_survival <= max_pix_survival_fraction:
            break

        print("Fraction in shower events of pixels with >",
              min_charge_for_certain_selection, "pe & first neighbors:",
              np.round(fraction_of_survival, 3), "is higher than maximum "
                                                 "allowed:",
              max_pix_survival_fraction)
        # Modify the value of min_charge_for_certain_selection to get a lower
        # survival fraction
        min_charge_for_certain_selection += 1.

    if min_charge_for_certain_selection > picture_threshold:
        print("min_charge_for_certain_selection changed to",
              min_charge_for_certain_selection)
    print("Fraction in shower events of pixels with >",
          min_charge_for_certain_selection, "pe & first neighbors:",
          np.round(fraction_of_survival, 3)),

    return min_charge_for_certain_selection
