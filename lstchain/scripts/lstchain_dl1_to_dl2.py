#!/usr/bin/env python3

"""
Pipeline for the reconstruction of Energy, disp and gamma/hadron
separation of events stored in a DL1 file.

- Input: DL1 files and trained Random Forests.
- Output: DL2 data file.

Usage:

$> python lstchain_dl1_to_dl2.py
--input-file dl1_LST-1.Run02033.0137.h5
--path-models ./trained_models

"""

import argparse
import os

import numpy as np
import pandas as pd
from ctapipe.instrument import SubarrayDescription
from tables import open_file

from lstchain.io import (
    get_dataset_keys,
    get_srcdep_params,
    global_metadata,
    read_configuration_file,
    replace_config,
    standard_config,
    write_dl2_dataframe,
    write_metadata,
)
from lstchain.io.io import (
    dl1_images_lstcam_key,
    dl1_params_lstcam_key,
    dl1_params_src_dep_lstcam_key,
    dl1_likelihood_params_lstcam_key,
    dl2_params_src_dep_lstcam_key,
    dl2_likelihood_params_lstcam_key,
    write_dataframe,
)
from lstchain.reco import dl1_to_dl2
from lstchain.reco.utils import filter_events, impute_pointing, add_delta_t_key

parser = argparse.ArgumentParser(description="DL1 to DL2")

# Required arguments
parser.add_argument('--input-file', '-f', type=str,
                    dest='input_file',
                    help='path to a DL1 HDF5 file',
                    default=None, required=True)

parser.add_argument('--path-models', '-p', action='store', type=str,
                    dest='path_models',
                    help='Path where to find the trained RF',
                    default='./trained_models')

# Optional arguments
parser.add_argument('--output-dir', '-o', action='store', type=str,
                    dest='output_dir',
                    help='Path where to store the reco dl2 events',
                    default='./dl2_data')

parser.add_argument('--config', '-c', action='store', type=str,
                    dest='config_file',
                    help='Path to a configuration file. If none is given, a standard configuration is applied',
                    default=None, required=False)


def main():
    args = parser.parse_args()

    custom_config = {}
    if args.config_file is not None:
        try:
            custom_config = read_configuration_file(os.path.abspath(args.config_file))
        except("Custom configuration could not be loaded !!!"):
            pass

    config = replace_config(standard_config, custom_config)

    data = pd.read_hdf(args.input_file, key=dl1_params_lstcam_key)

    if 'lh_fit_config' in config.keys():
        lhfit_data = pd.read_hdf(args.input_file, key=dl1_likelihood_params_lstcam_key)
        if np.all(lhfit_data['obs_id'] == data['obs_id']) & np.all(lhfit_data['event_id'] == data['event_id']):
            lhfit_data.drop({'obs_id', 'event_id'}, axis=1, inplace=True)
        lhfit_keys = lhfit_data.keys()
        data = pd.concat([data, lhfit_data], axis=1)

    # if real data, add deltat t to dataframe keys
    data = add_delta_t_key(data)

    # Dealing with pointing missing values. This happened when `ucts_time` was invalid.
    if 'alt_tel' in data.columns and 'az_tel' in data.columns \
            and (np.isnan(data.alt_tel).any() or np.isnan(data.az_tel).any()):
        # make sure there is a least one good pointing value to interp from.
        if np.isfinite(data.alt_tel).any() and np.isfinite(data.az_tel).any():
            data = impute_pointing(data)
        else:
            data.alt_tel = - np.pi / 2.
            data.az_tel = - np.pi / 2.

    # Get trained RF path for reconstruction:
    file_reg_energy = os.path.join(args.path_models, 'reg_energy.sav')
    file_cls_gh = os.path.join(args.path_models, 'cls_gh.sav')
    if config['disp_method'] == 'disp_vector':
        file_disp_vector = os.path.join(args.path_models, 'reg_disp_vector.sav')
    elif config['disp_method'] == 'disp_norm_sign':
        file_disp_norm = os.path.join(args.path_models, 'reg_disp_norm.sav')
        file_disp_sign = os.path.join(args.path_models, 'cls_disp_sign.sav')

    subarray_info = SubarrayDescription.from_hdf(args.input_file)
    tel_id = config["allowed_tels"][0] if "allowed_tels" in config else 1
    focal_length = subarray_info.tel[tel_id].optics.equivalent_focal_length

    # Apply the models to the data

    # Source-independent analysis
    if not config['source_dependent']:
        data = filter_events(data,
                             filters=config["events_filters"],
                             finite_params=config['energy_regression_features']
                                           + config['disp_regression_features']
                                           + config['particle_classification_features']
                                           + config['disp_classification_features'],
                             )

        if config['disp_method'] == 'disp_vector':
            dl2 = dl1_to_dl2.apply_models(data, file_cls_gh, file_reg_energy, reg_disp_vector=file_disp_vector,
                                          focal_length=focal_length,
                                          custom_config=config)
        elif config['disp_method'] == 'disp_norm_sign':
            dl2 = dl1_to_dl2.apply_models(data, file_cls_gh, file_reg_energy, reg_disp_norm=file_disp_norm,
                                          cls_disp_sign=file_disp_sign,
                                          focal_length=focal_length, custom_config=config)

    # Source-dependent analysis
    if config['source_dependent']:

        # if source-dependent parameters are already in dl1 data, just read those data.
        if dl1_params_src_dep_lstcam_key in get_dataset_keys(args.input_file):
            data_srcdep = get_srcdep_params(args.input_file)

        # if not, source-dependent parameters are added now
        else:
            data_srcdep = pd.concat(dl1_to_dl2.get_source_dependent_parameters(
                data, config, focal_length=focal_length), axis=1)

        dl2_srcdep_dict = {}
        srcindep_keys = data.keys()
        srcdep_assumed_positions = data_srcdep.columns.levels[0]

        for i, k in enumerate(srcdep_assumed_positions):
            data_with_srcdep_param = pd.concat([data, data_srcdep[k]], axis=1)
            data_with_srcdep_param = filter_events(data_with_srcdep_param,
                                                   filters=config["events_filters"],
                                                   finite_params=config['energy_regression_features']
                                                                 + config['disp_regression_features']
                                                                 + config['particle_classification_features']
                                                                 + config['disp_classification_features'],
                                                   )

            if config['disp_method'] == 'disp_vector':
                dl2_df = dl1_to_dl2.apply_models(data_with_srcdep_param, file_cls_gh, file_reg_energy,
                                                 reg_disp_vector=file_disp_vector,
                                                 focal_length=focal_length, custom_config=config)
            elif config['disp_method'] == 'disp_norm_sign':
                dl2_df = dl1_to_dl2.apply_models(data_with_srcdep_param, file_cls_gh, file_reg_energy,
                                                 reg_disp_norm=file_disp_norm, cls_disp_sign=file_disp_sign,
                                                 focal_length=focal_length, custom_config=config)

            dl2_srcdep = dl2_df.drop(srcindep_keys, axis=1)
            dl2_srcdep_dict[k] = dl2_srcdep

            if i == 0:
                dl2_srcindep = dl2_df[srcindep_keys]

    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, os.path.basename(args.input_file).replace('dl1', 'dl2', 1))

    if os.path.exists(output_file):
        raise IOError(output_file + ' exists, exiting.')

    dl1_keys = get_dataset_keys(args.input_file)

    if dl1_images_lstcam_key in dl1_keys:
        dl1_keys.remove(dl1_images_lstcam_key)

    if dl1_params_lstcam_key in dl1_keys:
        dl1_keys.remove(dl1_params_lstcam_key)

    if dl1_params_src_dep_lstcam_key in dl1_keys:
        dl1_keys.remove(dl1_params_src_dep_lstcam_key)

    if dl1_likelihood_params_lstcam_key in dl1_keys:
        dl1_keys.remove(dl1_likelihood_params_lstcam_key)

    metadata = global_metadata()
    write_metadata(metadata, output_file)

    with open_file(args.input_file, 'r') as h5in:
        with open_file(output_file, 'a') as h5out:

            # Write the selected DL1 info
            for k in dl1_keys:
                if not k.startswith('/'):
                    k = '/' + k

                path, name = k.rsplit('/', 1)
                if path not in h5out:
                    grouppath, groupname = path.rsplit('/', 1)
                    g = h5out.create_group(
                        grouppath, groupname, createparents=True
                    )
                else:
                    g = h5out.get_node(path)

                h5in.copy_node(k, g, overwrite=True)

    # need container to use lstchain.io.add_global_metadata and lstchain.io.add_config_metadata
    if not config['source_dependent']:
        if 'lh_fit_config' not in config.keys():
            write_dl2_dataframe(dl2, output_file, config=config, meta=metadata)
        else:
            dl2_onlylhfit = dl2[lhfit_keys]
            dl2.drop(lhfit_keys, axis=1, inplace=True)
            write_dl2_dataframe(dl2, output_file, config=config, meta=metadata)
            write_dataframe(dl2_onlylhfit, output_file, dl2_likelihood_params_lstcam_key, config=config, meta=metadata)

    else:
        write_dl2_dataframe(dl2_srcindep, output_file, config=config, meta=metadata)
        write_dataframe(pd.concat(dl2_srcdep_dict, axis=1), output_file, dl2_params_src_dep_lstcam_key, config=config,
                        meta=metadata)


if __name__ == '__main__':
    main()
