# Read DL1 file and recompute parameters

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

parser = argparse.ArgumentParser(description="Update parameters table in a DL1 HDF5 file")


# Required arguments
parser.add_argument('datafile', type=str,
                    help='path to the DL1 file ',
                    )

parser.add_argument('--config_file', '-conf', action='store', type=str,
                    dest='config_file',
                    help='Path to a configuration file. If none is given, a standard configuration is applied',
                    default=None
                    )

# parser.add_argument('--outfile', '-o', action='store', type=str,
#                     dest='outfile',
#                     help='Path to a new DL1 file',
#                     )


args = parser.parse_args()



if __name__ == '__main__':
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

    with tables.open_file(args.datafile, mode='a') as file:
        images = file.root[dl1_images_lstcam_key][:]['image']
        pulse_times = file.root[dl1_images_lstcam_key][:]['pulse_time']
        params = file.root[dl1_params_lstcam_key].read()

        for ii, image in enumerate(images):
            pulse_time = pulse_times[ii]
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

        file.root[dl1_params_lstcam_key][:] = params