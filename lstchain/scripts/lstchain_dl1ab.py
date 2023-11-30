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
import json

import astropy.units as u
import numpy as np
import tables
from ctapipe.io import read_table, write_table
from ctapipe.image import (
    tailcuts_clean,
    number_of_islands,
    apply_time_delta_cleaning,
)
from ctapipe.instrument import SubarrayDescription
from ctapipe_io_lst import constants

from lstchain.calib.camera.pixel_threshold_estimation import get_threshold_from_dl1_file
from lstchain.image.cleaning import apply_dynamic_cleaning
from lstchain.image.modifier import random_psf_smearer, set_numba_seed, add_noise_in_pixels
from lstchain.io import get_dataset_keys, copy_h5_nodes, HDF5_ZSTD_FILTERS, add_source_filenames, add_config_metadata

from lstchain.io.config import (
    get_cleaning_parameters,
    get_standard_config,
    read_configuration_file,
    replace_config,
    includes_image_modification,
)
from lstchain.io.io import (
    dl1_images_lstcam_key,
    dl1_params_lstcam_key,
    global_metadata,
    write_metadata,
    dl1_mon_tel_catB_ped_key,
    dl1_mon_tel_catB_flat_key,
    dl1_mon_tel_catB_cal_key
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
    '--catB-calibration-file',
    type=Path,
    help='path to the Cat-B calibration file ',
)

parser.add_argument(
    '--max-unusable-pixels',
    type=int,
    default=70,
    help='Maximum accepted number of unusable pixels. Default: 70 (= 10 modules)',
)

parser.add_argument(
    '-c', '--config',
    dest='config_file',
    help='Path to a configuration file. If none is given, a standard configuration is applied',
)

parser.add_argument(
    '--no-image', action='store_true',
    help='Pass this argument to avoid writing the images in the new DL1 files.',
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

    # read Cat-B calibration data if available
    catB_calib = None
    if args.catB_calibration_file is not None:
        if not args.catB_calibration_file.exists():
            log.critical(f"Calibration file {args.catB_calibration_file} not found")
            sys.exit(1)

        log.info(f"Cat-B calbration file: {args.catB_calibration_file}")
        catB_calib = read_table(args.catB_calibration_file, "/tel_1/calibration")

        # add the calibration index
        catB_calib['calibration_id'] = np.arange(len(catB_calib))
        catB_calib['pedestal_id'] = np.arange(len(catB_calib))
        catB_calib['flatfield_id'] = np.arange(len(catB_calib))

        catB_pedestal = read_table(args.catB_calibration_file, "/tel_1/pedestal")
        catB_pedestal['pedestal_id'] = np.arange(len(catB_pedestal))

        catB_flatfield = read_table(args.catB_calibration_file, "/tel_1/flatfield")
        catB_flatfield['pedestal_id'] = np.arange(len(catB_flatfield))

        catB_calib_time = np.array(catB_calib["time_min"])
        catB_dc_to_pe = np.array(catB_calib["dc_to_pe"])

        catB_pedestal_per_sample = np.array(catB_calib["pedestal_per_sample"])

        catB_time_correction = np.array(catB_calib["time_correction"])
        catB_unusable_pixels = np.array(catB_calib["unusable_pixels"])

        # add good time interval column (gti)
        catB_calib['gti'] = np.max(np.sum(catB_unusable_pixels, axis=2),axis=1) < args.max_unusable_pixels

        pixel_index = np.arange(constants.N_PIXELS)


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
        if not args.no_image:
            log.info("Modified images are saved in the output file.")
 
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
        log.info(f"Fraction of Cat_A pixel cleaning thresholds above Cat_A picture thr.:"
                 f"{np.sum(pedestal_thresh > pic_th) / len(pedestal_thresh):.3f}")
        picture_th = np.clip(pedestal_thresh, pic_th, None)
        log.info(f"Tailcut clean with pedestal threshold config used:"
                 f"{config['tailcuts_clean_with_pedestal_threshold']}")
        
        if args.catB_calibration_file is not None:
            catB_pedestal_mean = np.array(catB_pedestal["charge_mean"])
            catB_pedestal_std= np.array(catB_pedestal["charge_std"])
            catB_threshold_clean_pe = catB_pedestal_mean + sigma * catB_pedestal_std


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
    parameters_to_update = {
        'intensity': np.float64,
        'x': np.float32,
        'y': np.float32,
        'r': np.float32,
        'phi': np.float32,
        'length': np.float32,
        'width': np.float32,
        'psi': np.float32,
        'skewness': np.float32,
        'kurtosis': np.float32,
        'concentration_cog': np.float32,
        'concentration_core': np.float32,
        'concentration_pixel': np.float32,
        'leakage_intensity_width_1': np.float32,
        'leakage_intensity_width_2': np.float32,
        'leakage_pixels_width_1': np.float32,
        'leakage_pixels_width_2': np.float32,
        'n_islands': np.int32,
        'intercept': np.float64,
        'time_gradient': np.float64,
        'n_pixels': np.int32,
        'wl': np.float32,
        'log_intensity': np.float64,
        'sin_az_tel': np.float32,
    }

    if catB_calib:
        parameters_to_update["calibration_id"] = np.int32

    nodes_keys = get_dataset_keys(args.input_file)
    if args.no_image:
        nodes_keys.remove(dl1_images_lstcam_key)

    metadata = global_metadata()

    with tables.open_file(args.input_file, mode='r') as infile:
        image_table = read_table(infile, dl1_images_lstcam_key)
        # if the image modifier has been used to produce these images, stop here
        config_from_image_table = json.loads(image_table.meta['config'])
        if includes_image_modification(config_from_image_table) and includes_image_modification(config):
            log.critical(f"\nThe image modifier has already been used to produce the images in file {args.input_file}.\n"
                        "Re-applying the image modifier is not a good practice, start again from unmodified images please.")
            sys.exit(1)

        images = image_table['image']
        params = read_table(infile, dl1_params_lstcam_key)
        dl1_params_input = params.colnames

        disp_params = {'disp_dx': np.float32,
                       'disp_dy': np.float32,
                       'disp_norm': np.float32,
                       'disp_angle': np.float32,
                       'disp_sign': np.int32
                       }
        if set(dl1_params_input).intersection(disp_params):
            parameters_to_update.update(disp_params)

        uncertainty_params = {'width_uncertainty': np.float32,
                              'length_uncertainty': np.float32,
                              }
        if set(dl1_params_input).intersection(uncertainty_params):
            parameters_to_update.update(uncertainty_params)

        if catB_calib:
            trigger_times = params['trigger_time']

        if increase_nsb:
            rng = np.random.default_rng(infile.root.dl1.event.subarray.trigger.col('obs_id')[0])

        if increase_psf:
            set_numba_seed(infile.root.dl1.event.subarray.trigger.col('obs_id')[0])

        new_params = set(parameters_to_update.keys()) - set(params.colnames)
        for p in new_params:
            params[p] = np.empty(len(params), dtype=parameters_to_update[p])

        with tables.open_file(args.output_file, mode='a', filters=HDF5_ZSTD_FILTERS) as outfile:
            copy_h5_nodes(infile, outfile, nodes=nodes_keys)
            add_source_filenames(outfile, [args.input_file])

            # need container to use lstchain.io.add_global_metadata and lstchain.io.add_config_metadata
            for k, item in metadata.as_dict().items():
                outfile.root[dl1_params_lstcam_key].attrs[k] = item
            outfile.root[dl1_params_lstcam_key].attrs["config"] = str(config)

            for ii, row in enumerate(image_table):

                dl1_container.reset()

                image = row['image']
                peak_time = row['peak_time']

                if catB_calib:
                    selected_gain = row['selected_gain_channel']

                    # search right Cat-B calibration and update the index
                    calib_idx = np.searchsorted(catB_calib_time, trigger_times[ii])
                    if calib_idx > 0:
                        calib_idx -= 1

                    dl1_container.calibration_id = calib_idx

                    dc_to_pe = catB_dc_to_pe[calib_idx][selected_gain, pixel_index]
                    time_correction = catB_time_correction[calib_idx][selected_gain, pixel_index]
                    unusable_pixels = catB_unusable_pixels[calib_idx][selected_gain, pixel_index]

                    n_samples = config['LocalPeakWindowSum']['window_width']
                    pedestal = catB_pedestal_per_sample[calib_idx][selected_gain,pixel_index] * n_samples

                    # calibrate charge
                    image = (image - pedestal) * dc_to_pe

                    # put to zero charge unusable pixels in order not to select them in the cleaning
                    image[unusable_pixels] = 0

                    # time flafielding
                    peak_time = peak_time + time_correction

                    # store it to save it later
                    image_table['image'][ii] = image
                    image_table['peak_time'][ii] = peak_time

                    # use CatB pedestals to estimate the picture threshold 
                    # as defined in the config file
                    if args.pedestal_cleaning:
                        threshold_clean_pe = catB_threshold_clean_pe[calib_idx][selected_gain, pixel_index]
                        threshold_clean_pe[unusable_pixels] = pic_th
                        picture_th = np.clip(threshold_clean_pe, pic_th, None)

                if increase_nsb:
                    # Add noise in pixels, to adjust MC to data noise levels.
                    # TO BE DONE: in case of "pedestal cleaning" (not used now
                    # in MC) we should recalculate picture_th above!
                    image = add_noise_in_pixels(rng, 
                                                image,
                                                extra_noise_in_dim_pixels,
                                                extra_bias_in_dim_pixels,
                                                transition_charge,
                                                extra_noise_in_bright_pixels)
                if increase_psf:
                    image = random_psf_smearer(image, 
                                               smeared_light_fraction,
                                               camera_geom.neighbor_matrix_sparse.indices,
                                               camera_geom.neighbor_matrix_sparse.indptr)

                signal_pixels = tailcuts_clean(camera_geom,
                                               image,
                                               picture_th,
                                               boundary_th,
                                               isolated_pixels,
                                               min_n_neighbors,
                                               )

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
                                                             1,
                                                             delta_time)
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
                    # first island is no-island and should not be considered
                    n_pixels_on_island[0] = 0
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

                images[ii] = image

                if 'image_mask' in image_table.colnames:
                    image_table['image_mask'][ii] = signal_pixels


            add_config_metadata(image_table, config)
            if not args.no_image:
                write_table(image_table, outfile, dl1_images_lstcam_key, overwrite=True, filters=HDF5_ZSTD_FILTERS)

            add_config_metadata(params, config)
            write_table(params, outfile, dl1_params_lstcam_key, overwrite=True, filters=HDF5_ZSTD_FILTERS)

            # write a cat-B calibrations in DL1b
            if catB_calib:
                write_table(catB_calib, outfile, dl1_mon_tel_catB_cal_key)
                write_table(catB_pedestal, outfile, dl1_mon_tel_catB_ped_key)
                write_table(catB_flatfield, outfile, dl1_mon_tel_catB_flat_key)

        write_metadata(metadata, args.output_file)


if __name__ == '__main__':
    main()
