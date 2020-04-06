'''
Script to perform the analysis of muon events.
To run it, type:

python lstchain_data_muon_analysis_dl1.py
--input_file dl1_Run01566_0322.h5
--output_file Data_table.fits
'''
import argparse
import glob
import os
import numpy as np
from ctapipe.instrument import CameraGeometry
from astropy import units as u

from lstchain.image.muon import (
    analyze_muon_event, tag_pix_thr, create_muon_table, fill_muon_event
)
from lstchain.io.io import dl1_params_lstcam_key
from lstchain.visualization import plot_calib

from astropy.table import Table
import pandas as pd
import tables

parser = argparse.ArgumentParser()

# Required arguments
parser.add_argument(
    "--input_file", type=str, required=True,
    help="Path to DL1a data file (containing charge information).",
)

parser.add_argument(
    "--calib_file",
    type=str, default=None, required=False,
    help="Path to corresponding calibration file (containing bad pixel information).",
)

parser.add_argument(
    "--output_file", type=str, required=True,
    help="Path to create the output fits table with muon parameters",
)

# Optional argument
parser.add_argument(
    "--plot_rings", default=False, action='store_true',
    help="Plot figures of the stored rings",
)

parser.add_argument(
    "--plots_path", default=None, type=str,
    help="Path to the plots",
)

parser.add_argument(
    "--max_muons", type=int,
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

    num_muons = 0

    for filename in filenames:
        print('Opening file', filename)

        cam_description_table = Table.read(filename, path="instrument/telescope/camera/LSTCam")
        geom = CameraGeometry.from_table(cam_description_table)

        with tables.open_file(filename) as file:

            # unfortunately pandas.read_hdf does not seem compatible with "with... as..." statements
            parameters = pd.read_hdf(filename, key = dl1_params_lstcam_key)
            telescope_description = pd.read_hdf(filename, key='instrument/telescope/optics')

            group = file.root.dl1.event.telescope.image.LST_LSTCam
            images = [x['image'] for x in group.iterrows()]

            equivalent_focal_length = telescope_description['equivalent_focal_length'].values * u.m
            mirror_area = telescope_description['mirror_area'].values * pow(u.m,2)

            # fill dummy event times with NaNs in case they do not exist (like in MC):
            if 'dragon_time' not in parameters.keys():
                dummy_times = np.empty(len(parameters['event_id']))
                dummy_times[:] = np.nan
                parameters['dragon_time'] = dummy_times

            for full_image, event_id, dragon_time in zip(images, parameters['event_id'], parameters['dragon_time']):
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
                    muonintensityparam, size_outside_ring, muonringparam, good_ring,
                    radial_distribution, mean_pixel_charge_around_ring,
                ) = analyze_muon_event(
                    event_id, image, geom, equivalent_focal_length,
                    mirror_area, args.plot_rings, args.plots_path
                )

                if good_ring:
                    num_muons += 1
                    print("Number of good muon rings found {}, EventID {}".format(num_muons, event_id))

                # write ring data, including also "not-so-good" rings
                # in case we want to reconsider ring selections!:
                fill_muon_event(
                    output_parameters, good_ring, event_id, dragon_time,
                    muonintensityparam, muonringparam, radial_distribution,
                    size_outside_ring, mean_pixel_charge_around_ring,
                )

                if max_muons is not None and num_muons == max_muons:
                    break

            if max_muons is not None and num_muons == max_muons:
                break

    table = Table(output_parameters)
    table.write(args.output_file, format='fits', overwrite=True)

if __name__ == '__main__':
    main()
