#!/usr/bin/env python3

"""
Read a HDF5 DL1 file, recompute parameters based on calibrated using pedestal
tailcuts cleaning method images and pulse times and a config file and write
a new HDF5 file.
Updated parameters are : Hillas paramaters, wl, r, leakage, n_islands,
intercept, time_gradient

- Input: DL1 data file.
- Output: DL1 data file.

Usage:

$> python lstchain_data_dl1ab_pedestal_cleaning.py
-f dl1_LST-1.Run03004.0000.h5 -o dl1b_LST-1.Run03004.0000.h5 -c cleaning_pedestal_config.json

"""

import argparse
from distutils.util import strtobool

import numpy as np
import tables
import astropy.units as u
from astropy.table import Table
from ctapipe.containers import HillasParametersContainer
from ctapipe.image import hillas_parameters
from ctapipe.image.cleaning import tailcuts_clean
from ctapipe.image.morphology import number_of_islands
from ctapipe.instrument import CameraGeometry, OpticsDescription

from lstchain.io import get_dataset_keys, auto_merge_h5files
from lstchain.io.config import get_standard_config
from lstchain.io.config import read_configuration_file, replace_config
from lstchain.io.io import dl1_params_lstcam_key, dl1_images_lstcam_key
from lstchain.io.lstcontainers import DL1ParametersContainer
from lstchain.image.cleaning.cleaning import (
    tailcuts_clean_with_pedestal_threshold,
    get_threshold_from_dl1_file
)

parser = argparse.ArgumentParser(
    description="Recompute DL1b parameters from a DL1a file")

# Required arguments
parser.add_argument('--input-file', '-f', action='store', type=str,
                    dest='input_file',
                    help='path to the DL1a file ',
                    default=None, required=True)

parser.add_argument('--output-file', '-o', action='store', type=str,
                    dest='output_file',
                    help='key for the table of new parameters',
                    default=None, required=True)

parser.add_argument('--config', '-c', action='store', type=str,
                    dest='config_file',
                    help='Path to a configuration file with parameters to pedestal cleaning.',
                    default=None
                    )
# Optional arguments
parser.add_argument('--no-image', action='store',
                    type=lambda x: bool(strtobool(x)),
                    dest='noimage',
                    help='Boolean. True to remove the images in output file',
                    default=False)

args = parser.parse_args()



def main():

    config = read_configuration_file(args.config_file)
    print(config)
    pic_th = config['tailcuts_clean_with_pedestal_threshold']['picture_thresh']
    bound_th = config['tailcuts_clean_with_pedestal_threshold']['boundary_thresh']
    sigma = config['tailcuts_clean_with_pedestal_threshold']['sigma']
    keep_isolated_pixels = config['tailcuts_clean_with_pedestal_threshold']['keep_isolated_pixels']
    min_number_picture_neighbors = config['tailcuts_clean_with_pedestal_threshold']['min_number_picture_neighbors']
    th_ped_interleave = get_threshold_from_dl1_file(args.input_file, sigma)
    print(th_ped_interleave)

    foclen = OpticsDescription.from_name('LST').equivalent_focal_length
    cam_table = Table.read(args.input_file, path="instrument/telescope/camera/LSTCam")
    camera_geom = CameraGeometry.from_table(cam_table)

    dl1_container = DL1ParametersContainer()
    parameters_to_update = list(HillasParametersContainer().keys())
    parameters_to_update.extend([
        'concentration_cog',
        'concentration_core',
        'concentration_pixel',
        'leakage_intensity_width_1',
        'leakage_intensity_width_2',
        'leakage_pixels_width_1',
        'leakage_pixels_width_2',
        'n_islands',
        'intercept',
        'time_gradient',
        'n_pixels',
        'wl',
        'r',
    ])

    nodes_keys = get_dataset_keys(args.input_file)
    print(nodes_keys)
    if args.noimage:
        nodes_keys.remove(dl1_images_lstcam_key)

    auto_merge_h5files([args.input_file], args.output_file, nodes_keys=nodes_keys)

    with tables.open_file(args.input_file, mode='r') as input:
        image_table = input.root[dl1_images_lstcam_key]
        with tables.open_file(args.output_file, mode='a') as output:

            params = output.root[dl1_params_lstcam_key].read()

            for ii, row in enumerate(image_table):
                if ii % 10000 == 0:
                    print(ii)
                image = row['image']
                peak_time = row['peak_time']

                signal_pixels = tailcuts_clean_with_pedestal_threshold(
                                    camera_geom,
                                    image,
                                    th_ped_interleave,
                                    picture_thresh=pic_th,
                                    boundary_thresh=bound_th,
                                    keep_isolated_pixels=keep_isolated_pixels,
                                    min_number_picture_neighbors=min_number_picture_neighbors,
                                    )
                n_pixels = np.count_nonzero(signal_pixels)
                if n_pixels > 0:
                    num_islands, island_labels = number_of_islands(camera_geom, signal_pixels)
                    n_pixels_on_island = np.bincount(island_labels.astype(np.int))
                    n_pixels_on_island[0] = 0  # first island is no-island and should not be considered
                    max_island_label = np.argmax(n_pixels_on_island)
                    signal_pixels[island_labels != max_island_label] = False

                    hillas = hillas_parameters(camera_geom[signal_pixels], image[signal_pixels])

                    dl1_container.fill_hillas(hillas)
                    dl1_container.set_timing_features(camera_geom[signal_pixels],
                                                      image[signal_pixels],
                                                      peak_time[signal_pixels],
                                                      hillas)

                    dl1_container.set_leakage(camera_geom, image, signal_pixels)
                    dl1_container.set_concentration(camera_geom, image, hillas)
                    dl1_container.n_islands = num_islands
                    dl1_container.wl = dl1_container.width / dl1_container.length
                    dl1_container.n_pixels = n_pixels
                    width = np.rad2deg(np.arctan2(dl1_container.width, foclen))
                    length = np.rad2deg(np.arctan2(dl1_container.length, foclen))
                    dl1_container.width = width
                    dl1_container.length = length
                    dl1_container.r = np.sqrt(dl1_container.x ** 2 + dl1_container.y ** 2)

                else:
                    # for consistency with r0_to_dl1.py:
                    for key in dl1_container.keys():
                        dl1_container[key] = \
                            u.Quantity(0, dl1_container.fields[key].unit)

                    dl1_container.width = u.Quantity(np.nan, u.m)
                    dl1_container.length = u.Quantity(np.nan, u.m)
                    dl1_container.wl  = u.Quantity(np.nan, u.m)

                for p in parameters_to_update:
                    params[ii][p] = u.Quantity(dl1_container[p]).value

            output.root[dl1_params_lstcam_key][:] = params


if __name__ == '__main__':
    main()
