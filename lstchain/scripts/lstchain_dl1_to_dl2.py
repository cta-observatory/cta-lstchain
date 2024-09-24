#!/usr/bin/env python3
__doc__ = """
Run the DL1 to DL2 step: Pipeline for the reconstruction of Energy, disp and gamma/hadron
separation of events stored in a DL1 file. It takes DL1 file(s) and trained Random Forests as input 
and outputs DL2 data file(s).
Run lstchain_dl1_to_dl2 --help to see the options.
"""

import argparse
from pathlib import Path
import joblib
import logging
import numpy as np
import pandas as pd
import astropy.units as u
from astropy.coordinates import Angle
from ctapipe.instrument import SubarrayDescription
from ctapipe_io_lst import OPTICS
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

logger = logging.getLogger(__name__)
parser = argparse.ArgumentParser(description=__doc__)

# Required arguments
parser.add_argument('--input-files', '-f',
                    type=Path,
                    nargs='+',
                    dest='input_files',
                    help='Path (or list of paths) to a DL1 HDF5 file',
                    required=True)

parser.add_argument('--path-models', '-p',
                    action='store',
                    type=Path,
                    dest='path_models',
                    help='Path where to find the trained RF',
                    default='./trained_models')

# Optional arguments
parser.add_argument('--output-dir', '-o',
                    action='store',
                    type=Path,
                    dest='output_dir',
                    help='Path where to store the reco dl2 events',
                    default='./dl2_data')

parser.add_argument('--config', '-c',
                    action='store',
                    type=Path,
                    dest='config_file',
                    help='Path to a configuration file. If none is given, a standard configuration is applied',
                    default=None,
                    required=False)


def apply_to_file(filename, models_dict, output_dir, config):

    data = pd.read_hdf(filename, key=dl1_params_lstcam_key)

    if 'lh_fit_config' in config.keys():
        lhfit_data = pd.read_hdf(filename, key=dl1_likelihood_params_lstcam_key)
        if np.all(lhfit_data['obs_id'] == data['obs_id']) & np.all(lhfit_data['event_id'] == data['event_id']):
            lhfit_data.drop({'obs_id', 'event_id'}, axis=1, inplace=True)
        lhfit_keys = lhfit_data.keys()
        data = pd.concat([data, lhfit_data], axis=1)

    # if real data, add deltat t to dataframe keys
    data = add_delta_t_key(data)

    # Dealing with pointing missing values. This happened when `ucts_time` was invalid.
    if 'alt_tel' in data.columns and 'az_tel' in data.columns \
            and (np.isnan(data.alt_tel).any() or np.isnan(data.az_tel).any()):
        # make sure there is at least one good pointing value to interp from.
        if np.isfinite(data.alt_tel).any() and np.isfinite(data.az_tel).any():
            data = impute_pointing(data)
        else:
            data.alt_tel = - np.pi / 2.
            data.az_tel = - np.pi / 2.

    try:
        subarray_info = SubarrayDescription.from_hdf(filename)
        tel_id = config["allowed_tels"][0] if "allowed_tels" in config else 1
        effective_focal_length = subarray_info.tel[tel_id].optics.effective_focal_length
    except OSError:
        print("subarray table is not readable because of the version incompatibility.")
        print("The effective focal length for the standard LST optics will be used.")
        effective_focal_length = OPTICS.effective_focal_length
        
    # Normalize all azimuth angles to the range [0, 360) degrees 
    data.az_tel = Angle(data.az_tel, u.rad).wrap_at(360 * u.deg).rad

    # Dealing with `sin_az_tel` missing data because of the former version of lstchain
    if 'sin_az_tel' not in data.columns:
        data['sin_az_tel'] = np.sin(data.az_tel)


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
            dl2 = dl1_to_dl2.apply_models(data,
                                          models_dict['cls_gh'],
                                          models_dict['reg_energy'],
                                          reg_disp_vector=models_dict['reg_disp_vector'],
                                          effective_focal_length=effective_focal_length,
                                          custom_config=config)
        elif config['disp_method'] == 'disp_norm_sign':
            dl2 = dl1_to_dl2.apply_models(data,
                                          models_dict['cls_gh'],
                                          models_dict['reg_energy'],
                                          reg_disp_norm=models_dict['reg_disp_norm'],
                                          cls_disp_sign=models_dict['cls_disp_sign'],
                                          effective_focal_length=effective_focal_length,
                                          custom_config=config)

    # Source-dependent analysis
    if config['source_dependent']:
        # if source-dependent parameters are already in dl1 data, just read those data.
        if dl1_params_src_dep_lstcam_key in get_dataset_keys(filename):
            data_srcdep = get_srcdep_params(filename)

        # if not, source-dependent parameters are added now
        else:
            data_srcdep = pd.concat(dl1_to_dl2.get_source_dependent_parameters(
                data, config, effective_focal_length=effective_focal_length), axis=1)

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
                dl2 = dl1_to_dl2.apply_models(data_with_srcdep_param,
                                                 models_dict['cls_gh'],
                                                 models_dict['reg_energy'],
                                                 reg_disp_vector=models_dict['reg_disp_vector'],
                                                 effective_focal_length=effective_focal_length,
                                                 custom_config=config)
            elif config['disp_method'] == 'disp_norm_sign':
                dl2 = dl1_to_dl2.apply_models(data_with_srcdep_param,
                                                 models_dict['cls_gh'],
                                                 models_dict['reg_energy'],
                                                 reg_disp_norm=models_dict['reg_disp_norm'],
                                                 cls_disp_sign=models_dict['cls_disp_sign'],
                                                 effective_focal_length=effective_focal_length,
                                                 custom_config=config)

            dl2_srcdep = dl2.drop(srcindep_keys, axis=1)
            dl2_srcdep_dict[k] = dl2_srcdep

            if i == 0:
                dl2_srcindep = dl2[srcindep_keys]

    # do not write file if empty
    if len(dl2) == 0:
        logger.warning("No dl2 output file written.")
        return

    output_dir.mkdir(exist_ok=True)
    output_file = output_dir.joinpath(filename.name.replace('dl1', 'dl2', 1))

    if output_file.exists():
        raise IOError(str(output_file) + ' exists, exiting.')

    dl1_keys = get_dataset_keys(filename)

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

    with open_file(filename, 'r') as h5in:
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
        if 'lh_fit_config' not in config.keys():
            write_dl2_dataframe(dl2_srcindep, output_file, config=config, meta=metadata)
        else:
            dl2_onlylhfit = dl2_srcindep[lhfit_keys]
            dl2_srcindep.drop(lhfit_keys, axis=1, inplace=True)
            write_dl2_dataframe(dl2_srcindep, output_file, config=config, meta=metadata)
            write_dataframe(dl2_onlylhfit, output_file, dl2_likelihood_params_lstcam_key, config=config, meta=metadata)
        write_dataframe(pd.concat(dl2_srcdep_dict, axis=1), output_file, dl2_params_src_dep_lstcam_key, config=config,
                        meta=metadata)


def main():
    args = parser.parse_args()

    custom_config = {}
    if args.config_file is not None:
        try:
            custom_config = read_configuration_file(args.config_file.absolute())
        except("Custom configuration could not be loaded !!!"):
            pass

    config = replace_config(standard_config, custom_config)

    models_keys = ['reg_energy', 'cls_gh']

    if config['disp_method'] == 'disp_vector':
        models_keys.append('reg_disp_vector')
    elif config['disp_method'] == 'disp_norm_sign':
        models_keys.extend(['reg_disp_norm', 'cls_disp_sign'])

    models_dict = {}
    for models_key in models_keys:
        models_path = Path(args.path_models, f'{models_key}.sav')

        # For a single input file, each model is loaded just before it is used
        if len(args.input_files)==1:
            models_dict[models_key] = models_path
        # For multiple input files, all the models are loaded only once here 
        else:
            models_dict[models_key] = joblib.load(models_path)

    for filename in args.input_files:
        apply_to_file(filename, models_dict, args.output_dir, config)



if __name__ == '__main__':
    main()
