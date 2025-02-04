#!/usr/bin/env python3
__doc__ = """
Run the DL1 to DL2 step: Pipeline for the reconstruction of Energy, disp and gamma/hadron
separation of events stored in a DL1 file. It takes DL1 file(s) and trained Random Forests as input 
and outputs DL2 data file(s).
Run lstchain_dl1_to_dl2 --help to see the options.
"""

from pathlib import Path
import joblib
import logging
import numpy as np
import pandas as pd
import astropy.units as u
from astropy.table import Table
from astropy.coordinates import Angle
from ctapipe.instrument import SubarrayDescription
from ctapipe_io_lst import OPTICS
from tables import open_file
from ctapipe.core import Tool, ToolConfigurationError, traits, Provenance

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
from lstchain.io.provenance import write_provenance
from lstchain.reco import dl1_to_dl2
from lstchain.reco.utils import filter_events, impute_pointing, add_delta_t_key

logger = logging.getLogger(__name__)


__all__ = ["DL1ToDL2Tool"]


def dl2_filename(dl1_filename):
    """
    Create the name of the DL2 file from the DL1 file name.

    Parameters:
    -----------
    dl1_filename : str
        Name of the DL1 file

    Returns:
    --------
    str
        Name of the DL2 file
    """
    return dl1_filename.replace('dl1', 'dl2', 1)

class DL1ToDL2Tool(Tool):
    name = "lstchain_dl1_to_dl2"
    description = __doc__

    input_files = traits.List(
        traits.Path,
        help="Path (or list of paths) to a DL1 HDF5 file",
    ).tag(config=True)

    path_models = traits.Path(
        help="Path where to find the trained RF",
        default_value='./trained_models',
    ).tag(config=True)

    output_dir = traits.Path(
        help="Path where to store the reco dl2 events",
        default_value='./dl2_data',
    ).tag(config=True)

    config_file = traits.Path(
        allow_none=True,
        help="Path to a configuration file. If none is given, a standard configuration is applied",
        default_value=None,
    ).tag(config=True)

    aliases = {
        ("f", "input-files"): "DL1ToDL2Tool.input_files",
        ("p", "path-models"): "DL1ToDL2Tool.path_models",
        ("o", "output-dir"): "DL1ToDL2Tool.output_dir",
        ("c", "config"): "DL1ToDL2Tool.config_file",
    }

    def setup(self):
         
        # Check if input files are provided
        if not self.input_files:
            raise ToolConfigurationError("No input files provided. Use --input-files to specify.")

        # Additional setup logic can go here
        self.log.info(f"Input files: {self.input_files}")
        self.log.info(f"Path to models: {self.path_models}")
        self.log.info(f"Output directory: {self.output_dir}")

    def start(self):
            
        custom_config = {}
        if self.config_file is not None:
            try:
                custom_config = read_configuration_file(self.config_file.absolute())
            except Exception as e:
                self.log.error(f"Custom configuration could not be loaded: {e}")
                return

        config = replace_config(standard_config, custom_config)

        models_keys = ['reg_energy', 'cls_gh']

        if config['disp_method'] == 'disp_vector':
            models_keys.append('reg_disp_vector')
        elif config['disp_method'] == 'disp_norm_sign':
            models_keys.extend(['reg_disp_norm', 'cls_disp_sign'])

        models_dict = {}
        for models_key in models_keys:
            models_path = Path(self.path_models, f'{models_key}.sav')

            if len(self.input_files) == 1:
                models_dict[models_key] = models_path
            else:
                models_dict[models_key] = joblib.load(models_path)
                
        output_dir = Path(self.output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        
        for input_dl1file in self.input_files:
            dl2_output_file = output_dir.joinpath(dl2_filename(input_dl1file.name))

            p = Provenance()
            p.add_input_file(input_dl1file, role='dl1 input file')
            p.add_output_file(dl2_output_file, role='dl2 output file')
            p.add_input_file(self.path_models, role='trained model directory')

            # Remove previous file if overwrite option is used:
            if self.overwrite:
                dl2_output_file.unlink(missing_ok=True)

            if dl2_output_file.exists():
                raise IOError(str(dl2_output_file) + ' exists, exiting.')

            write_provenance(dl2_output_file, 'dl1_to_dl2')
            
            apply_to_file(input_dl1file, models_dict, dl2_output_file, config, 
                          self.path_models)
                

def apply_to_file(filename, models_dict, dl2_output_file, config, models_path):
    """
    Applies models to the data in the specified file and writes the output to a new file in the output directory.

    Parameters:
    - filename (Path or str): The path to the input file.
    - models_dict (dict): A dictionary containing the models to be applied.
    - dl2_output_file (Path or str): The path for the output DL2 file.
    - config (dict): The configuration dictionary containing parameters for the processing.
    - models_path (Path or str): The path to the directory containing the trained models.
    """

    data = pd.read_hdf(filename, key=dl1_params_lstcam_key)

    # Read in the settings for the interpolation of Random Forest predictions
    # in cos(zd). If activated this avoids the jumps of performance produced
    # by the discrete set of pointings in the RF training sample.

    interpolate_rf = None     # Default, no interpolation
    training_pointings = None # Default, no interpolation
    if 'random_forest_zd_interpolation' in config:
        zdinter = config['random_forest_zd_interpolation']
        interpolate_energy = zdinter.get('interpolate_energy', False)
        interpolate_gammaness = zdinter.get('interpolate_gammaness', False)
        interpolate_direction = zdinter.get('interpolate_direction', False)
        interpolate_rf = {'energy_regression': interpolate_energy,
                          'particle_classification': interpolate_gammaness,
                          'disp': interpolate_direction
                          }
        if True in interpolate_rf.values():
            logger.info('Cos(zenith) interpolation will be used in:')
            if interpolate_energy:
                logger.info('   energy reconstruction Random Forest')
            if interpolate_gammaness:
                logger.info('   g/h classification Random Forest')
            if interpolate_direction:
                logger.info('   direction reconstruction Random Forest')

            # Obtain the training pointings, needed for the RF interpolation:
            training_pointings_path = Path(models_path, 'training_dirs.ecsv')
            if training_pointings_path.is_file():
                training_pointings = Table.read(training_pointings_path)
                logger.info('RF training pointings:')
                logger.info(training_pointings)
            else:
                logger.warning(f'{training_pointings_path} not found!')
                logger.warning('Switching off RF interpolation with zenith!')
                interpolate_rf = None

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
                                          custom_config=config,
                                          interpolate_rf=interpolate_rf,
                                          training_pointings=training_pointings)
        elif config['disp_method'] == 'disp_norm_sign':
            dl2 = dl1_to_dl2.apply_models(data,
                                          models_dict['cls_gh'],
                                          models_dict['reg_energy'],
                                          reg_disp_norm=models_dict['reg_disp_norm'],
                                          cls_disp_sign=models_dict['cls_disp_sign'],
                                          effective_focal_length=effective_focal_length,
                                          custom_config=config,
                                          interpolate_rf=interpolate_rf,
                                          training_pointings=training_pointings)

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
                                                 custom_config=config,
                                                 interpolate_rf=interpolate_rf,
                                                 training_pointings=training_pointings)
            elif config['disp_method'] == 'disp_norm_sign':
                dl2 = dl1_to_dl2.apply_models(data_with_srcdep_param,
                                                 models_dict['cls_gh'],
                                                 models_dict['reg_energy'],
                                                 reg_disp_norm=models_dict['reg_disp_norm'],
                                                 cls_disp_sign=models_dict['cls_disp_sign'],
                                                 effective_focal_length=effective_focal_length,
                                                 custom_config=config,
                                                 interpolate_rf=interpolate_rf,
                                                 training_pointings=training_pointings)

            dl2_srcdep = dl2.drop(srcindep_keys, axis=1)
            dl2_srcdep_dict[k] = dl2_srcdep

            if i == 0:
                dl2_srcindep = dl2[srcindep_keys]

    # do not write file if empty
    if len(dl2) == 0:
        logger.warning("No dl2 output file written.")
        return

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
    write_metadata(metadata, dl2_output_file)

    with open_file(filename, 'r') as h5in:
        with open_file(dl2_output_file, 'a') as h5out:

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
            write_dl2_dataframe(dl2, dl2_output_file, config=config, meta=metadata)
        else:
            dl2_onlylhfit = dl2[lhfit_keys]
            dl2.drop(lhfit_keys, axis=1, inplace=True)
            write_dl2_dataframe(dl2, dl2_output_file, config=config, meta=metadata)
            write_dataframe(dl2_onlylhfit, dl2_output_file, dl2_likelihood_params_lstcam_key, config=config, meta=metadata)
    else:
        if 'lh_fit_config' not in config.keys():
            write_dl2_dataframe(dl2_srcindep, dl2_output_file, config=config, meta=metadata)
        else:
            dl2_onlylhfit = dl2_srcindep[lhfit_keys]
            dl2_srcindep.drop(lhfit_keys, axis=1, inplace=True)
            write_dl2_dataframe(dl2_srcindep, dl2_output_file, config=config, meta=metadata)
            write_dataframe(dl2_onlylhfit, dl2_output_file, dl2_likelihood_params_lstcam_key, config=config, meta=metadata)
        write_dataframe(pd.concat(dl2_srcdep_dict, axis=1), dl2_output_file, dl2_params_src_dep_lstcam_key, config=config,
                        meta=metadata)
        
    return dl2_output_file
        

def main():

    tool = DL1ToDL2Tool()
    tool.run()


if __name__ == '__main__':
    main()

