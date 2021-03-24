#!/usr/bin/env python3

"""
Script to perform the analysis of muon events.

- Inputs are a DL1a data file (pixel information is needed) and a
calibration file
- Output is a table with muon parameters (to be updated to a dataframe!)

Usage:

$> python lstchain_muon_analysis_dl1.py
--input-file dl1_Run01566_0322.h5
--output-file Data_table.fits
--calibration-file calibration.Run2029.0000.hdf5

"""

import argparse
import glob

import numpy as np
import pandas as pd
from astropy.table import Table
from ctapipe.instrument import SubarrayDescription

from lstchain.image.muon import (
    analyze_muon_event,
    create_muon_table,
    fill_muon_event,
    tag_pix_thr,
)
from lstchain.io.io import dl1_params_lstcam_key, dl1_images_lstcam_key
from lstchain.io.io import read_telescopes_descriptions, read_subarray_description
from lstchain.visualization import plot_calib

parser = argparse.ArgumentParser()

# Required arguments
parser.add_argument(
    "--input-file", '-f', type=str,
    dest='input_file',
    required=True,
    help="Path to DL1a data file (containing charge information).",
)

parser.add_argument(
    "--output-file", '-o', type=str,
    dest='output_file',
    required=True,
    help="Path to create the output fits table with muon parameters",
)

parser.add_argument(
    "--calibration-file", '--calib',
    dest='calib_file',
    type=str, default=None, required=False,
    help="Path to corresponding calibration file (containing bad pixel information).",
)


# Optional argument
parser.add_argument(
    "--plot-rings",
    dest='plot_rings',
    default=False, action='store_true',
    help="Plot figures of the stored rings",
)

parser.add_argument(
    "--plots-path",
    dest='plots_path',
    default=None, type=str,
    help="Path to the plots",
)

parser.add_argument(
    "--max-muons",
    dest='max_muons',
    type=int,
    help="Maximum number of processed muon ring candidates",
)

args = parser.parse_args()


def main():

    print("input files: {}".format(args.input_file))
    print("calib file: {}".format(args.calib_file))
    print("output file: {}".format(args.output_file))

    max_muons = args.max_muons

    # Definition of the output parameters for the table
    output_parameters = create_muon_table()

    if args.calib_file is not None:
        plot_calib.read_file(args.calib_file)
        bad_pixels = plot_calib.calib_data.unusable_pixels[0]
        print(f"Found a total of {np.sum(bad_pixels)} bad pixels.")

    # image = pd.read_hdf(args.input_file, key = dl1_image_lstcam_key)
    # The call above does not work, because of the file's vector columns (pixel-wise charges & times)
    # So we use tables for the time being.

    print(glob.glob(args.input_file))

    filenames = glob.glob(args.input_file)
    filenames.sort()

    lst1_tel_id = 1

    num_muons = 0

    for filename in filenames:
        print('Opening file', filename)

        subarray_info = SubarrayDescription.from_hdf(filename)
        geom = subarray_info.tel[lst1_tel_id].camera.geometry

        subarray = read_subarray_description(filename, subarray_name='LST-1')

        images = Table.read(filename, path=dl1_images_lstcam_key)['image']

        parameters = pd.read_hdf(filename, key=dl1_params_lstcam_key)
        telescope_description = read_telescopes_descriptions(filename)[lst1_tel_id]

        equivalent_focal_length = telescope_description.optics.equivalent_focal_length
        mirror_area = telescope_description.optics.mirror_area

        # fill dummy event times with NaNs in case they do not exist (like in MC):
        if 'dragon_time' not in parameters.keys():
            dummy_times = np.empty(len(parameters['event_id']))
            dummy_times[:] = np.nan
            parameters['dragon_time'] = dummy_times

        for full_image, event_id, dragon_time, mc_energy in zip(
                images, parameters['event_id'], parameters['dragon_time'], parameters['mc_energy']):
            if args.calib_file is not None:
                image = full_image*(~bad_pixels)
            else:
                image = full_image
            # print("Event {}. Number of pixels above 10 phe: {}".format(event_id,
            #                                                           np.size(image[image > 10.])))
            # if((np.size(image[image > 10.]) > 300) or (np.size(image[image > 10.]) < 50)):
            #     continue
            if not tag_pix_thr(image): # default skips pedestal and calibration events
                continue

            # default values apply no filtering.
            # This filter is rather useless for biased extractors anyway
            # if not muon_filter(image)
            #    continue

            (
                muonintensityparam, dist_mask, size, size_outside_ring,
                muonringparam, good_ring, radial_distribution,
                mean_pixel_charge_around_ring, muonparameters
            ) = analyze_muon_event(subarray,
                event_id, image, geom, equivalent_focal_length,
                mirror_area, args.plot_rings, args.plots_path
            )

            if good_ring:
                num_muons += 1
                print("Number of good muon rings found {}, EventID {}".format(num_muons, event_id))

            # write ring data, including also "not-so-good" rings
            # in case we want to reconsider ring selections!:
            fill_muon_event(
                mc_energy, output_parameters, good_ring, event_id,
                dragon_time, muonintensityparam, dist_mask,
                muonringparam, radial_distribution, size,
                size_outside_ring, mean_pixel_charge_around_ring,
                muonparameters
            )

            if max_muons is not None and num_muons == max_muons:
                break

        if max_muons is not None and num_muons == max_muons:
            break

    table = Table(output_parameters)
    table.write(args.output_file, format='fits', overwrite=True)

if __name__ == '__main__':
    main()
