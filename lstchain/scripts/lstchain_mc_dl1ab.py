# Read a HDF5 DL1 file, recompute parameters based on calibrated images and pulse times and a config file
# and write a new HDF5 file
# Updated parameters are : Hillas paramaters, wl, r, leakage, n_islands, intercept, time_gradient


import os
import shutil
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

parser = argparse.ArgumentParser(description="Recompute parameters in a DL1 HDF5 file from calibrated images"
                                             "and based on passed config file. The results are written in a new HDF5 "
                                             "file."
                                             "Updated parameters are : Hillas paramaters, wl, r, leakage, "
                                             "n_islands, intercept, time_gradient")

# Required arguments
parser.add_argument('input_file', type=str, help='path to the DL1 file ')

parser.add_argument('output_file', type=str, help='key for the table of new parameters')

parser.add_argument('--config_file', '-conf', action='store', type=str,
                    dest='config_file',
                    help='Path to a configuration file. If none is given, a standard configuration is applied',
                    default=None
                    )

parser.add_argument('--no-image', action='store', type=lambda x: bool(strtobool(x)),
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

    geom = CameraGeometry.from_name('LSTCam-002')
    foclen = OpticsDescription.from_name('LST').equivalent_focal_length
    dl1_container = DL1ParametersContainer()
    parameters_to_update = list(HillasParametersContainer().keys())
    parameters_to_update.extend(['wl', 'r', 'leakage', 'n_islands', 'intercept', 'time_gradient'])

    nodes_keys = get_dataset_keys(args.input_file)
    if args.noimage:
        nodes_keys.remove(dl1_images_lstcam_key)

    auto_merge_h5files([args.input_file], args.output_file, nodes_keys=nodes_keys)

    with tables.open_file(args.input_file, mode='r') as input:
        image_table = input.root[dl1_images_lstcam_key]
        with tables.open_file(args.output_file, mode='a') as output:

            params = output.root[dl1_params_lstcam_key].read()

            for ii, row in enumerate(image_table):
                if ii%10000 == 0:
                    print(ii)
                image = row['image']
                pulse_time = row['pulse_time']
                signal_pixels = tailcuts_clean(geom, image, **config['tailcut'])
                if image[signal_pixels].shape[0] > 0:
                    num_islands, island_labels = number_of_islands(geom, signal_pixels)
                    hillas = hillas_parameters(geom[signal_pixels], image[signal_pixels])

                    dl1_container.fill_hillas(hillas)
                    dl1_container.set_timing_features(geom[signal_pixels],
                                                      image[signal_pixels],
                                                      pulse_time[signal_pixels],
                                                      hillas)
                    dl1_container.set_leakage(geom, image, signal_pixels)
                    dl1_container.n_islands = num_islands
                    dl1_container.wl = dl1_container.width / dl1_container.length
                    width = np.rad2deg(np.arctan2(dl1_container.width, foclen))
                    length = np.rad2deg(np.arctan2(dl1_container.length, foclen))
                    dl1_container.width = width.value
                    dl1_container.length = length.value
                    dl1_container.r = np.sqrt(dl1_container.x**2 + dl1_container.y**2)

                    for p in parameters_to_update:
                        params[ii][p] = Quantity(dl1_container[p]).value
                else:
                    for p in parameters_to_update:
                        params[ii][p] = 0

            output.root[dl1_params_lstcam_key][:] = params


if __name__ == '__main__':
    main()
