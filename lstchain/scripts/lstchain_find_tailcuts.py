#!/usr/bin/env python

"""
This script uses DL1 files to determine tailcuts which are adequate for the
bulk of the pixels in a given run. It does so simply based on the median (for
the whole camera) of the average pixel charge for pedestal events.

For reasons of stability & simplicity of analysis, we cannot decide the
cleaning levels on a subrun-by-subrun basis. We select values which are ok
for the whole run.

The script also returns the suggested NSB adjustment needed in the "dark-sky" MC
to match the data.

The script will process a maximum of 10 subruns of those provided as input,
distributed uniformly through the run - just for speed reasons: the result
would hardly change if all subruns are used.

lstchain_find_tailcuts -f "/.../dl1_LST-1.Run12469.????.h5"

The program creates in the output directory a .json file containing the
cleaning configuration.

"""

import argparse
import logging
from pathlib import Path
import tables
import glob
import os
import numpy as np
import time
import sys

from lstchain.paths import parse_dl1_filename
from lstchain.io.io import dl1_params_lstcam_key, dl1_images_lstcam_key
from lstchain.io.io import dl1_params_tel_mon_cal_key
from lstchain.io.config import get_standard_config, dump_config

from ctapipe.io import read_table
from ctapipe.core import Container, Field
from ctapipe.io import HDF5TableWriter
from ctapipe.containers import EventType
from ctapipe_io_lst import LSTEventSource

parser = argparse.ArgumentParser(description="Tailcut finder")

parser.add_argument('-f', '--dl1-files', dest='dl1_files',
                    type=str, default='',
                    help='Input DL1 file names')
parser.add_argument('-o', '--output-dir', dest='output_dir',
                    type=Path, default='./',
                    help='Path to the output directory (default: %(default)s)')
parser.add_argument('--log', dest='log_file',
                    type=str, default=None,
                    help='Log file name')

log = logging.getLogger(__name__)


def main():
    args = parser.parse_args()
    log.setLevel(logging.INFO)
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    output_dir = args.output_dir.absolute()
    output_dir.mkdir(exist_ok=True, parents=True)

    if args.dl1_files == '':
        log.error('Please use --dl1-files to provide a valid set of DL1 files!')
        sys.exit(1)

    all_dl1_files = glob.glob(args.dl1_files)
    all_dl1_files.sort()

    log_file = args.log_file
    if log_file is not None:
        handler = logging.FileHandler(log_file, mode='w')
        logging.getLogger().addHandler(handler)

    # Number of subruns (uniformly distributed through the run) to be processed:
    max_number_of_processed_subruns = 10

    # process at most max_number_of_processed_subruns for each run:
    dl1_files = get_input_files(all_dl1_files, max_number_of_processed_subruns)
    number_of_pedestals = []
    usable_pixels = []
    median_ped_mean_pix_charge = []

    for dl1_file in dl1_files:
        log.info('\nInput file: %s', dl1_file)

        data_parameters = read_table(dl1_file, dl1_params_lstcam_key)
        event_type_data = data_parameters['event_type'].data
        pedestal_mask = event_type_data == EventType.SKY_PEDESTAL.value

        number_of_pedestals.append(pedestal_mask.sum())
        data_images = read_table(dl1_file, dl1_images_lstcam_key)
        data_calib = read_table(dl1_file, dl1_params_tel_mon_cal_key)
        # data_calib['unusable_pixels'] , indices: (Gain  Calib_id  Pixel)

        # Get the "unusable" flags from the pedcal file:
        unusable_HG = data_calib['unusable_pixels'][0][0]
        unusable_LG = data_calib['unusable_pixels'][0][1]

        reliable_pixels = ~(unusable_HG | unusable_LG)
        usable_pixels.append(reliable_pixels)

        charges_data = data_images['image']
        charges_pedestals = charges_data[pedestal_mask]
        mean_ped_charge = np.mean(charges_pedestals, axis=0)
        median_ped_mean_pix_charge.append(np.median(mean_ped_charge[
                                                        reliable_pixels]))

    median_ped_mean_pix_charge = np.array(median_ped_mean_pix_charge)
    number_of_pedestals = np.array(number_of_pedestals)

    # Now compute the median for all processed subruns, which is more robust
    # against e.g. subruns affected by car flashes. We also exclude subruns
    # which have less than half of the median statistics per subrun.
    good_stats = number_of_pedestals > 0.5 * np.median(number_of_pedestals)
    qped = np.median(median_ped_mean_pix_charge[good_stats])

    picture_threshold = pic_th(qped)
    boundary_threshold = picture_threshold / 2

    # We now create a .json files with recommended image cleaning
    # settings for lstchain_dl1ab.
    newconfig = get_standard_config()['tailcuts_clean_with_pedestal_threshold']
    # casts below are needed, json does not like numpy's int64:
    newconfig['picture_thresh'] = int(picture_threshold)
    newconfig['boundary_thresh'] = int(boundary_threshold)

    run_info = parse_dl1_filename(dl1_files[0])
    run_id = run_info.run

    json_filename = Path(output_dir, f'dl1ab_Run{run_id:05d}.json')
    dump_config({'tailcuts_clean_with_pedestal_threshold': newconfig,
                 'dynamic_cleaning': get_standard_config()['dynamic_cleaning']},
                json_filename, overwrite=True)
    log.info(json_filename)
    log.info('lstchain_find_tailcuts finished successfully!')


def get_input_files(all_dl1_files, max_number_of_processed_subruns):
    """
    Reduce the number of DL1 files in all_dl1_files to a maximum of
    max_number_of_processed_subruns per run
    """

    runlist = np.unique([parse_dl1_filename(f).run for f in all_dl1_files])

    dl1_files = []
    for run in runlist:
        file_list = [f for f in all_dl1_files 
                     if f.find(f'dl1_LST-1.Run{run:05d}')>0]
        if len(file_list) <= max_number_of_processed_subruns:
            dl1_files.extend(file_list)
            continue
        step = len(file_list) / max_number_of_processed_subruns
        k = 0
        while np.round(k) < len(file_list):
            dl1_files.append(file_list[int(np.round(k))])
            k += step

    return dl1_files


def pic_th(mean_ped):
    """
    mean_ped: mean pixel charge in pedestal events (for the standard
    LocalPeakWindowSearch algo & settings in lstchain)

    Returns:
        recommended picture threshold for image cleaning (from a table)
    """
    mp_edges = np.array([2.4, 3.1, 3.8, 4.5, 5.2])
    picture_threshold = np.array([8, 10, 12, 14, 16, 18])

    if mean_ped >= mp_edges[-1]:
        return picture_threshold[-1]
    return picture_threshold[np.where(mp_edges>mean_ped)[0][0]]
