#!/usr/bin/env python

import argparse
from pathlib import Path
import tables
import numpy as np

from lstchain.paths import parse_dl1_filename
from ctapipe.instrument import SubarrayDescription
from ctapipe.io import read_table
from ctapipe.core import Container, Field
from ctapipe.io import HDF5TableWriter
from ctapipe.containers import EventType

parser = argparse.ArgumentParser(description="DVR pixel selector")

parser.add_argument('-f', '--dl1-file', dest='dl1_file',
                    type=Path,
                    help='Input DL1 file name')
parser.add_argument('-o', '--output-dir', dest='output_dir',
                    type=Path, default='./',
                    help='Path to the output directory')
parser.add_argument('-t', '--picture-threshold', dest='picture_threshold',
                    type=float, default=8.,
                    help='Picture threshold (p.e.)')
parser.add_argument('-n', '--number-of-rings', dest='number_of_rings',
                    type=int, default=1,
                    help='Number of rings around picture pixels')

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

    # The pixel selection will be applied only to "physics triggers"
    # (showers), not to other events like interleaved pedestal and flatfield
    # events:
    event_types_to_be_reduced = [EventType.SUBARRAY.value,
                                 EventType.UNKNOWN.value,
                                 EventType.HARDWARE_STEREO.value,
                                 EventType.DAQ.value]
    # (Note: some of these types are not yet in use). Chances are that pixels
    # labeled as "unknown" are cosmics, so we include them among the types to
    # be reduced.

    # By default use the provided picture threshold to keep pixels:
    min_charge_for_certain_selection = args.picture_threshold

    # For the record, keep the input "picture threshold". It is not necessarily
    # the value that will be used for pixel selection, if it allows too many
    # pixels too survive.
    summary_info.picture_threshold = args.picture_threshold

    # Value will be modified, if a too high average fraction (
    # > max_pix_survival_fraction_in_ped) of pixels (and FIRST neighbors)
    # survive, in pedestal events, with that picture threshold
    max_pix_survival_fraction_in_ped = 0.05

    # The new value, if survival fraction turns out to be too high, will be the
    # charge above which a fraction ped_cut_fraction of pedestal charges lie.
    # This will be computed using the 90% of pixels with lower mean pedestal
    # (to avoid pixels illuminated by bright stars to dominate the calculation)
    ped_cut_fraction = 0.001

    # Number of "rings" of pixels to be finally kept around pixels which are
    # above min_charge_for_certain_selection
    number_of_rings = args.number_of_rings
    summary_info.number_of_rings = number_of_rings

    dl1_file = args.dl1_file
    run_info = parse_dl1_filename(dl1_file)
    run_id, subrun_id = run_info.run, run_info.subrun
    summary_info.run_id = run_id
    summary_info.subrun_id = subrun_id

    output_dir = args.output_dir.absolute()
    output_dir.mkdir(exist_ok=True, parents=True)
    output_file = Path(output_dir, f'Pixel_selection_LST-1.Run{run_id:05d}.'
                                   f'{subrun_id:04d}.h5')
    print('Output file: ', output_file)

    data_parameters = read_table(dl1_file,
                                 "/dl1/event/telescope/parameters/LST_LSTCam")

    # Read some useful extra info from the file:
    subarray_info = SubarrayDescription.from_hdf(dl1_file)
    focal_length = subarray_info.tel[1].optics.equivalent_focal_length
    print('Focal length:', focal_length)
    camera_geom = subarray_info.tel[1].camera.geometry
    print('Camera geometry:', camera_geom)

    # Time between first and last timestamp:
    summary_info.elapsed_time = (data_parameters['dragon_time'][-1] -
                                 data_parameters['dragon_time'][0])

    summary_info.mean_zenith = np.round(90 - np.rad2deg(data_parameters['alt_tel'].mean()), 2)

    number_of_events = len(data_parameters)
    summary_info.number_of_events = number_of_events

    event_type_data = data_parameters['event_type']
    # event_type: physics trigger (32) interleaved flat-field(0) pedestal (2),
    # unknown(255)  are those currently in use in LST1

    cosmic_mask   = event_type_data == EventType.SUBARRAY.value
    pedestal_mask = event_type_data == EventType.SKY_PEDESTAL.value

    data_images = read_table(dl1_file, "/dl1/event/telescope/image/LST_LSTCam")

    data_calib = read_table(dl1_file, "/dl1/event/telescope/monitoring/calibration")
    # data_calib['unusable_pixels'] , indices: (Gain  Calib_id  Pixel)

    # Get the "unusable" flags from the pedcal file:
    unusable_HG = data_calib['unusable_pixels'][0][0]
    unusable_LG = data_calib['unusable_pixels'][0][1]

    #  Keep pixels that could not be properly calibrated: save them always,
    #  in case their calibration can be later fixed:
    keep_always_mask = (unusable_HG | unusable_LG)
    summary_info.number_of_always_saved_pixels = keep_always_mask.sum()

    if summary_info.number_of_always_saved_pixels > 0:
        print(f'Pixels that will always be saved ({summary_info.number_of_always_saved_pixels}):')
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

    # NOTE: for files which have no interleaved pedestals or which have too few
    # because it is the last subrun in a run, we may have trouble...

    ped_mean = np.nanmean(charges_data[pedestal_mask], axis=0)
    # Exclude 10% of brightest pixels (to avoid influence of stars, here we want
    # to estimate the "~diffuse" NSB, or "average in the bulk of the pixels, not
    # including bright stars")
    ninety_percent_dimmest_pixels = np.argsort(ped_mean)[:-int(len(ped_mean)*0.1)]
    # pedestal charges for those 90% of the pixels:
    pq = charges_data[pedestal_mask][:,ninety_percent_dimmest_pixels].flatten()
    # Cut that keeps a given fraction (pedcutfraction) of the pedestal charges:

    charge_per_mil_pedestal = np.sort(pq)[-int(ped_cut_fraction*len(pq))]
    charge_per_mil_pedestal = np.round(charge_per_mil_pedestal)
    # We round it to an integer number of photo-electrons. We want to avoid too
    # many different ways of reducing the data. This is also why we use the
    # fixed value picture_threshold unless it results in a too high survival
    # rate

    # Check what fraction of pixels in PEDESTAL events is kept with the current
    # value of min_charge_for_certain_selection (and ONE ring of neighbors)

    fraction_of_survival = 1.
    while fraction_of_survival > max_pix_survival_fraction_in_ped:
        selected_pixels_masks = []
        for charge_map, event_type in zip(charges_data, event_type_data):
            # use only pedestal events:
            if event_type != EventType.SKY_PEDESTAL.value:
                continue
            selected_pixels = get_selected_pixels(charge_map,
                                                  min_charge_for_certain_selection,
                                                  1, camera_geom)
            selected_pixels_masks.append(selected_pixels)

        selected_pixels_masks = np.array(selected_pixels_masks)
        fraction_of_survival = (selected_pixels_masks.sum() /
                                len(selected_pixels_masks.flatten()))
        if fraction_of_survival <= max_pix_survival_fraction_in_ped:
            break

        print("Fraction in shower events of pixels with >",
              min_charge_for_certain_selection, "pe & first neighbors:",
              np.round(fraction_of_survival, 3), "is higher than maximum "
                                                 "allowed:",
              max_pix_survival_fraction_in_ped)
        # Modify the value of min_charge_for_certain_selection to get a lower
        # survival fraction
        min_charge_for_certain_selection += 1.

    print("min_charge_for_certain_selection changed to",
          min_charge_for_certain_selection)

    summary_info.min_charge_for_certain_selection = min_charge_for_certain_selection

    # Cuts to identify promising muon ring candidates:
    mucan_event_list = np.where(cosmic_mask &
                                (data_parameters['intensity']>1000) &
                                (data_parameters['length']>0.5) &
                                (data_parameters['n_pixels']>50) &
                                (abs(data_parameters['time_gradient'])<5))[0]

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

    data = PixelMask()
    writer_conf = tables.Filters(complevel=9, fletcher32=True)
    with HDF5TableWriter(output_file, filters=writer_conf) as writer:
        for evid, evtype, std_clean_npixels, highestQ, pixmask in zip(
                event_ids, event_type_data, data_parameters['n_pixels'],
                highest_removed_charge, selected_pixels_masks):
            data.event_id = evid
            data.event_type = evtype
            data.number_of_pixels_after_standard_cleaning = std_clean_npixels
            data.highest_removed_charge = highestQ
            data.pixmask = pixmask
            writer.write("selected_pixels_masks", data)

    with HDF5TableWriter(output_file, filters=writer_conf, mode='a') as writer:
        writer.write("run_summary", summary_info)



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
