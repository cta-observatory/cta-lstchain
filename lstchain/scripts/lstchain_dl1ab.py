#!/usr/bin/env python3

"""
Read a HDF5 DL1 file, recompute parameters based on calibrated images and
pulse times and a config file and write a new HDF5 file
Updated parameters are : Hillas paramaters, wl, r, leakage, n_islands,
intercept, time_gradient
- Input: DL1 data file.
- Output: DL1 data file.
Usage:
$> python lstchain_dl1ab.py
--input-file dl1_gamma_20deg_0deg_run8___cta-prod3-lapalma-2147m-LaPalma-FlashCam.simtel.gz
"""

import argparse
import logging
from distutils.util import strtobool

import astropy.units as u
import numpy as np
import tables
from ctapipe.image import hillas_parameters
from ctapipe.image.cleaning import tailcuts_clean
from ctapipe.image.morphology import number_of_islands
from ctapipe.instrument import SubarrayDescription

from lstchain.calib.camera.pixel_threshold_estimation import get_threshold_from_dl1_file
from lstchain.io import get_dataset_keys, auto_merge_h5files
from lstchain.io.config import get_cleaning_parameters
from lstchain.io.config import get_standard_config
from lstchain.io.config import read_configuration_file, replace_config
from lstchain.io.io import dl1_params_lstcam_key, dl1_images_lstcam_key, read_metadata, write_metadata
from lstchain.io.lstcontainers import DL1ParametersContainer
from lstchain.reco.disp import disp
from lstchain.image.modifier import smear_light_in_pixels, add_noise_in_pixels
from lstchain.image.cleaning import apply_time_delta_cleaning, apply_dynamic_cleaning

log = logging.getLogger(__name__)

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

parser.add_argument('--pedestal-cleaning', action='store',
                    type=lambda x: bool(strtobool(x)),
                    dest='pedestal_cleaning',
                    help='Boolean. True to use pedestal cleaning',
                    default=False)

args = parser.parse_args()


def main():
    std_config = get_standard_config()

    log.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    logging.getLogger().addHandler(handler)

    if args.config_file is not None:
        config = replace_config(std_config, read_configuration_file(args.config_file))
    else:
        config = std_config

    increase_nsb = False
    increase_psf = False
    if "image_modifier" in config:
        imconfig = config["image_modifier"]
        increase_nsb = imconfig["increase_nsb"]
        increase_psf = imconfig["increase_psf"]
        if increase_nsb or increase_psf:
            log.info(f"image_modifier configuration: {imconfig}")
        extra_noise_in_dim_pixels = imconfig["extra_noise_in_dim_pixels"]
        extra_bias_in_dim_pixels = imconfig["extra_bias_in_dim_pixels"]
        transition_charge = imconfig["transition_charge"]
        extra_noise_in_bright_pixels = imconfig["extra_noise_in_bright_pixels"]
        smeared_light_fraction = imconfig["smeared_light_fraction"]
        if (increase_nsb or increase_psf) and args.noimage is False:
            log.info("NOTE: Using the image_modifier options means images will "
                     "not be saved.")
            args.noimage = True

    if args.pedestal_cleaning:
        log.info("Pedestal cleaning")
        clean_method_name = 'tailcuts_clean_with_pedestal_threshold'
        sigma = config[clean_method_name]['sigma']
        pedestal_thresh = get_threshold_from_dl1_file(args.input_file, sigma)
        cleaning_params = get_cleaning_parameters(config, clean_method_name)
        pic_th, boundary_th, isolated_pixels, min_n_neighbors = cleaning_params
        log.info(f"Fraction of pixel cleaning thresholds above picture thr.:"
                 f"{np.sum(pedestal_thresh>pic_th) / len(pedestal_thresh):.3f}")
        picture_th = np.clip(pedestal_thresh, pic_th, None)
        log.info(f"Tailcut clean with pedestal threshold config used:"
                 f"{config['tailcuts_clean_with_pedestal_threshold']}")
    else:
        clean_method_name = 'tailcut'
        cleaning_params = get_cleaning_parameters(config, clean_method_name)
        picture_th, boundary_th, isolated_pixels, min_n_neighbors = cleaning_params
        log.info(f"Tailcut config used: {config['tailcut']}")

    use_dynamic_cleaning = False
    if 'apply' in config['dynamic_cleaning']:
        use_dynamic_cleaning = config['dynamic_cleaning']['apply']

    if use_dynamic_cleaning:
        THRESHOLD_DYNAMIC_CLEANING = config['dynamic_cleaning']['threshold']
        FRACTION_CLEANING_SIZE = config['dynamic_cleaning']['fraction_cleaning_intensity']
        log.info("Using dynamic cleaning for events with average size of the "
            f"3 most brighest pixels > {config['dynamic_cleaning']['threshold']} p.e")
        log.info("Remove from image pixels which have charge below "
                 f"= {config['dynamic_cleaning']['fraction_cleaning_intensity']} * average size")
    
    use_only_main_island = True
    if "use_only_main_island" in config[clean_method_name]:
        use_only_main_island = config[clean_method_name]["use_only_main_island"]

    delta_time = None
    if "delta_time" in config[clean_method_name]:
        delta_time = config[clean_method_name]["delta_time"]

    subarray_info = SubarrayDescription.from_hdf(args.input_file)
    tel_id = config["allowed_tels"][0] if "allowed_tels" in config else 1
    focal_length = subarray_info.tel[tel_id].optics.equivalent_focal_length
    camera_geom = subarray_info.tel[tel_id].camera.geometry

    dl1_container = DL1ParametersContainer()
    parameters_to_update = [
        'intensity',
        'x',
        'y',
        'r',
        'phi',
        'length',
        'width',
        'psi',
        'skewness',
        'kurtosis',
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
        'log_intensity'
    ]

    nodes_keys = get_dataset_keys(args.input_file)
    if args.noimage:
        nodes_keys.remove(dl1_images_lstcam_key)

    auto_merge_h5files([args.input_file], args.output_file, nodes_keys=nodes_keys)
    metadata = read_metadata(args.input_file)

    with tables.open_file(args.input_file, mode='r') as input:
        image_table = input.root[dl1_images_lstcam_key]
        dl1_params_input = input.root[dl1_params_lstcam_key].colnames
        disp_params = {'disp_dx', 'disp_dy', 'disp_norm', 'disp_angle', 'disp_sign'}
        if set(dl1_params_input).intersection(disp_params):
            parameters_to_update.extend(disp_params)
        uncertainty_params = {'width_uncertainty', 'length_uncertainty'}
        if set(dl1_params_input).intersection(uncertainty_params):
            parameters_to_update.extend(uncertainty_params)

        if increase_nsb:
            rng = np.random.default_rng(
                    input.root.dl1.event.subarray.trigger.col('obs_id')[0])

        with tables.open_file(args.output_file, mode='a') as output:
            params = output.root[dl1_params_lstcam_key].read()

            for ii, row in enumerate(image_table):

                dl1_container.reset()

                image = row['image']
                peak_time = row['peak_time']

                if increase_nsb:
                    # Add noise in pixels, to adjust MC to data noise levels.
                    # TO BE DONE: in case of "pedestal cleaning" (not used now
                    # in MC) we should recalculate picture_th above!
                    image = add_noise_in_pixels(rng, image,
                                                extra_noise_in_dim_pixels,
                                                extra_bias_in_dim_pixels,
                                                transition_charge,
                                                extra_noise_in_bright_pixels)
                if increase_psf:
                    image = smear_light_in_pixels(image,
                                                  camera_geom,
                                                  smeared_light_fraction)
            
                

                signal_pixels = tailcuts_clean(camera_geom,
                                               image,
                                               picture_th,
                                               boundary_th,
                                               isolated_pixels,
                                               min_n_neighbors)
    

                n_pixels = np.count_nonzero(signal_pixels)
                if n_pixels > 0:
                    num_islands, island_labels = number_of_islands(camera_geom, signal_pixels)
                    n_pixels_on_island = np.bincount(island_labels.astype(np.int64))
                    n_pixels_on_island[0] = 0  # first island is no-island and should not be considered
                    max_island_label = np.argmax(n_pixels_on_island)
                    if use_only_main_island:
                        signal_pixels[island_labels != max_island_label] = False

                    # if delta_time has been set, we require at least one
                    # neighbor within delta_time to accept a pixel in the image:
                    if delta_time is not None:
                        cleaned_pixel_times = peak_time
                        # makes sure only signal pixels are used in the time
                        # check:
                        cleaned_pixel_times[~signal_pixels] = np.nan
                        new_mask = apply_time_delta_cleaning(camera_geom,
                                                             signal_pixels,
                                                             cleaned_pixel_times,
                                                             1, delta_time)
                        signal_pixels = new_mask

                    if use_dynamic_cleaning:
                        new_mask = apply_dynamic_cleaning(image,
                                                          signal_pixels,
                                                          THRESHOLD_DYNAMIC_CLEANING,
                                                          FRACTION_CLEANING_SIZE)
                        signal_pixels = new_mask

                    # count the surviving pixels
                    n_pixels = np.count_nonzero(signal_pixels)

                    if n_pixels > 0:
                        hillas = hillas_parameters(camera_geom[signal_pixels],
                                                   image[signal_pixels])

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
                        width = np.rad2deg(np.arctan2(dl1_container.width, focal_length))
                        width_uncertainty = np.rad2deg(np.arctan2(dl1_container.width_uncertainty, focal_length))
                        length = np.rad2deg(np.arctan2(dl1_container.length, focal_length))
                        length_uncertainty = np.rad2deg(np.arctan2(dl1_container.length_uncertainty, focal_length))
                        dl1_container.width = width
                        dl1_container.width_uncertainty = width_uncertainty
                        dl1_container.length = length
                        dl1_container.length_uncertainty = length_uncertainty
                        dl1_container.log_intensity = np.log10(dl1_container.intensity)

                if set(dl1_params_input).intersection(disp_params):
                    disp_dx, disp_dy, disp_norm, disp_angle, disp_sign = disp(
                        dl1_container['x'].to_value(u.m),
                        dl1_container['y'].to_value(u.m),
                        params['src_x'][ii],
                        params['src_y'][ii]
                    )

                    dl1_container['disp_dx'] = disp_dx
                    dl1_container['disp_dy'] = disp_dy
                    dl1_container['disp_norm'] = disp_norm
                    dl1_container['disp_angle'] = disp_angle
                    dl1_container['disp_sign'] = disp_sign

                for p in parameters_to_update:

                    params[ii][p] = u.Quantity(dl1_container[p]).value

            output.root[dl1_params_lstcam_key][:] = params

    write_metadata(metadata, args.output_file)


if __name__ == '__main__':
    main()
