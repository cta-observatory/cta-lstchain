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
import sys
import argparse
import logging
from pathlib import Path

import astropy.units as u
import numpy as np
import tables
from ctapipe.image import (
    tailcuts_clean,
    number_of_islands,
    apply_time_delta_cleaning,
)
from ctapipe.instrument import SubarrayDescription

from lstchain.calib.camera.pixel_threshold_estimation import get_threshold_from_dl1_file
from lstchain.image.cleaning import apply_dynamic_cleaning
from lstchain.image.modifier import random_psf_smearer, set_numba_seed, add_noise_in_pixels
from lstchain.io import get_dataset_keys, copy_h5_nodes, HDF5_ZSTD_FILTERS, add_source_filenames

from lstchain.io.config import (
    get_cleaning_parameters,
    get_standard_config,
    read_configuration_file,
    replace_config,
)
from lstchain.io.io import (
    dl1_images_lstcam_key,
    dl1_params_lstcam_key,
    global_metadata, 
    write_metadata,
)
from lstchain.io.lstcontainers import DL1ParametersContainer
from lstchain.reco.disp import disp
from lstchain.reco.r0_to_dl1 import parametrize_image

log = logging.getLogger(__name__)

parser = argparse.ArgumentParser(
    description="Recompute DL1b parameters from a DL1a file"
)

# Required arguments
parser.add_argument(
    '-f', '--input-file',
    required=True,
    help='path to the DL1a file ',
)

parser.add_argument(
    '-o', '--output-file',
    required=True,
    help='key for the table of new parameters',
)
# Optional arguments
parser.add_argument(
    '-c', '--config',
    dest='config_file',
    help='Path to a configuration file. If none is given, a standard configuration is applied',
)

parser.add_argument(
    '--no-image', action='store_true',
    help='Pass this argument to avoid writing the images in the new DL1 files. Beware, if `increase_nsb` or `increase_psf` are True in the config, the images will not be written.',
)

parser.add_argument(
    '--no-pedestal-cleaning', action='store_false',
    dest='pedestal_cleaning',
    help='Disable pedestal cleaning. This is also done automatically for simulations.',
)


def main():
    args = parser.parse_args()

    log.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    logging.getLogger().addHandler(handler)

    if Path(args.output_file).exists():
        log.critical(f'Output file {args.output_file} already exists')
        sys.exit(1)

    std_config = get_standard_config()
    if args.config_file is not None:
        config = replace_config(std_config, read_configuration_file(args.config_file))
    else:
        config = std_config

    with tables.open_file(args.input_file, 'r') as f:
        is_simulation = 'simulation' in f.root

    imconfig = config.get('image_modifier', {})
    increase_nsb = imconfig.get("increase_nsb", False)
    increase_psf = imconfig.get("increase_psf", False)

    if increase_nsb or increase_psf:
        log.info(f"image_modifier configuration: {imconfig}")
        log.info("NOTE: Using the image_modifier options means images will "
                 "not be saved.")
        args.no_image = True
    if increase_nsb:
        extra_noise_in_dim_pixels = imconfig["extra_noise_in_dim_pixels"]
        extra_bias_in_dim_pixels = imconfig["extra_bias_in_dim_pixels"]
        transition_charge = imconfig["transition_charge"]
        extra_noise_in_bright_pixels = imconfig["extra_noise_in_bright_pixels"]
    if increase_psf:
        smeared_light_fraction = imconfig["smeared_light_fraction"]

    args.pedestal_cleaning = False if is_simulation else args.pedestal_cleaning

    if args.pedestal_cleaning:
        log.info("Pedestal cleaning")
        clean_method_name = 'tailcuts_clean_with_pedestal_threshold'
        sigma = config[clean_method_name]['sigma']
        pedestal_thresh = get_threshold_from_dl1_file(args.input_file, sigma)
        cleaning_params = get_cleaning_parameters(config, clean_method_name)
        pic_th, boundary_th, isolated_pixels, min_n_neighbors = cleaning_params
        log.info(f"Fraction of pixel cleaning thresholds above picture thr.:"
                 f"{np.sum(pedestal_thresh > pic_th) / len(pedestal_thresh):.3f}")
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
    optics = subarray_info.tel[tel_id].optics
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
        'log_intensity',
        'sin_az_tel'
    ]

    nodes_keys = get_dataset_keys(args.input_file)
    if args.no_image:
        nodes_keys.remove(dl1_images_lstcam_key)

    metadata = global_metadata()

    with tables.open_file(args.input_file, mode='r') as infile:
        image_table = infile.root[dl1_images_lstcam_key]
        dl1_params_input = infile.root[dl1_params_lstcam_key].colnames
        disp_params = {'disp_dx', 'disp_dy', 'disp_norm', 'disp_angle', 'disp_sign'}
        if set(dl1_params_input).intersection(disp_params):
            parameters_to_update.extend(disp_params)
        uncertainty_params = {'width_uncertainty', 'length_uncertainty'}
        if set(dl1_params_input).intersection(uncertainty_params):
            parameters_to_update.extend(uncertainty_params)

        if increase_nsb:
            rng = np.random.default_rng(
                    infile.root.dl1.event.subarray.trigger.col('obs_id')[0])

        if increase_psf:
            set_numba_seed(infile.root.dl1.event.subarray.trigger.col('obs_id')[0])

        image_mask_save = not args.no_image and 'image_mask' in infile.root[dl1_images_lstcam_key].colnames

        with tables.open_file(args.output_file, mode='a', filters=HDF5_ZSTD_FILTERS) as outfile:
            copy_h5_nodes(infile, outfile, nodes=nodes_keys)
            add_source_filenames(outfile, [args.input_file])

            params_node = outfile.root[dl1_params_lstcam_key]
            params = params_node.read()

            new_params = set(parameters_to_update) - set(params_node.colnames)
            if new_params:
                log.warning(f"Parameters not in original DL1 file {args.input_file} that can't be recomputed:"
                            f"{new_params}")
            parameters_to_update = list(set(parameters_to_update) & set(params_node.colnames))
            if image_mask_save:
                image_mask = outfile.root[dl1_images_lstcam_key].col('image_mask')

            # need container to use lstchain.io.add_global_metadata and lstchain.io.add_config_metadata
            for k, item in metadata.as_dict().items():
                outfile.root[dl1_params_lstcam_key].attrs[k] = item
            outfile.root[dl1_params_lstcam_key].attrs["config"] = str(config)

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
                    image = random_psf_smearer(image, smeared_light_fraction,
                                               camera_geom.neighbor_matrix_sparse.indices,
                                               camera_geom.neighbor_matrix_sparse.indptr)

                signal_pixels = tailcuts_clean(camera_geom,
                                               image,
                                               picture_th,
                                               boundary_th,
                                               isolated_pixels,
                                               min_n_neighbors)

                n_pixels = np.count_nonzero(signal_pixels)

                if n_pixels > 0:

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

                    # count a number of islands after all of the image cleaning steps
                    num_islands, island_labels = number_of_islands(camera_geom, signal_pixels)
                    dl1_container.n_islands = num_islands

                    n_pixels_on_island = np.bincount(island_labels.astype(np.int64))
                    n_pixels_on_island[0] = 0  # first island is no-island and should not be considered
                    max_island_label = np.argmax(n_pixels_on_island)

                    if use_only_main_island:
                        signal_pixels[island_labels != max_island_label] = False

                    # count the surviving pixels
                    n_pixels = np.count_nonzero(signal_pixels)
                    dl1_container.n_pixels = n_pixels

                    if n_pixels > 0:
                        parametrize_image(
                            image=image,
                            peak_time=peak_time,
                            signal_pixels=signal_pixels,
                            camera_geometry=camera_geom,
                            focal_length=optics.equivalent_focal_length,
                            dl1_container=dl1_container,
                        )

                if set(dl1_params_input).intersection(disp_params):
                    disp_dx, disp_dy, disp_norm, disp_angle, disp_sign = disp(
                        dl1_container['x'].to_value(u.m),
                        dl1_container['y'].to_value(u.m),
                        params['src_x'][ii],
                        params['src_y'][ii],
                        dl1_container['psi'].to_value(u.rad)
                    )

                    dl1_container['disp_dx'] = disp_dx
                    dl1_container['disp_dy'] = disp_dy
                    dl1_container['disp_norm'] = disp_norm
                    dl1_container['disp_angle'] = disp_angle
                    dl1_container['disp_sign'] = disp_sign

                dl1_container['sin_az_tel'] = np.sin(params['az_tel'][ii])
                    
                for p in parameters_to_update:
                    params[ii][p] = u.Quantity(dl1_container[p]).value

                if image_mask_save:
                    image_mask[ii] = signal_pixels

            outfile.root[dl1_params_lstcam_key][:] = params
            if image_mask_save:
                outfile.root[dl1_images_lstcam_key].modify_column(colname='image_mask', column=image_mask)

        write_metadata(metadata, args.output_file)


if __name__ == '__main__':
    main()
