#!/usr/bin/env python3

"""
Read a HDF5 DL1 file, recompute parameters based on calibrated images and 
pulse times and a config file and write a new HDF5 file
Updated parameters are : Hillas paramaters, wl, r, leakage, n_islands, 
intercept, time_gradient

- Input: DL1 data file.
- Output: DL1 data file.

Usage: 

$> python lstchain_mc_dl1ab.py 
--input-file dl1_gamma_20deg_0deg_run8___cta-prod3-lapalma-2147m-LaPalma-FlashCam.simtel.gz

"""

import tables
import numpy as np
import argparse
from lstchain.io.io import dl1_params_lstcam_key, dl1_images_lstcam_key
from ctapipe.image.cleaning import tailcuts_clean, number_of_islands
from ctapipe.image import hillas_parameters
from lstchain.io.config import read_configuration_file, replace_config
from lstchain.io.config import get_standard_config
from ctapipe.instrument import CameraGeometry, OpticsDescription
from lstchain.io.lstcontainers import DL1ParametersContainer
from ctapipe.io.containers import HillasParametersContainer
from astropy.units import Quantity
from distutils.util import strtobool
from lstchain.io import get_dataset_keys, auto_merge_h5files
from astropy.table import Table

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
# Optional arguments
parser.add_argument('--config', '-c', action='store', type=str,
                    dest='config_file',
                    help='Path to a configuration file. If none is given, a standard configuration is applied',
                    default=None
                    )

parser.add_argument('--no-image', action='store', 
                    type=lambda x: bool(strtobool(x)),
                    dest='noimage',
                    help='Boolean. True to remove the images in output file',
                    default=False)

args = parser.parse_args()



def main():
    std_config = get_standard_config()

    if args.config_file is not None:
        config = replace_config(std_config, read_configuration_file(args.config_file))
    else:
        config = std_config

    print(config['tailcut'])

    foclen = OpticsDescription.from_name('LST').equivalent_focal_length
    cam_table = Table.read(args.input_file, path="instrument/telescope/camera/LSTCam")
    camera_geom = CameraGeometry.from_table(cam_table)

    dl1_container = DL1ParametersContainer()
    parameters_to_update = list(HillasParametersContainer().keys())
    parameters_to_update.extend([
        'wl', 'r',
        'leakage1_intensity',
        'leakage2_intensity',
        'leakage1_pixel',
        'leakage2_pixel',
        'concentration_cog',
        'concentration_core',
        'concentration_pixel',
        'n_pixels',
        'n_islands', 'intercept', 'time_gradient'
    ])

    nodes_keys = get_dataset_keys(args.input_file)
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
                pulse_time = row['pulse_time']

                signal_pixels = tailcuts_clean(camera_geom, image, **config['tailcut'])
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
                                                      pulse_time[signal_pixels],
                                                      hillas)

                    dl1_container.set_leakage(camera_geom, image, signal_pixels)
                    dl1_container.set_concentration(camera_geom, image, hillas)
                    dl1_container.n_islands = num_islands
                    dl1_container.wl = dl1_container.width / dl1_container.length
                    dl1_container.n_pixels = n_pixels
                    width = np.rad2deg(np.arctan2(dl1_container.width, foclen))
                    length = np.rad2deg(np.arctan2(dl1_container.length, foclen))
                    dl1_container.width = width.value
                    dl1_container.length = length.value
                    dl1_container.r = np.sqrt(dl1_container.x ** 2 + dl1_container.y ** 2)

                    for p in parameters_to_update:
                        params[ii][p] = Quantity(dl1_container[p]).value
                else:
                    for p in parameters_to_update:
                        params[ii][p] = 0

            output.root[dl1_params_lstcam_key][:] = params


if __name__ == '__main__':
    main()
