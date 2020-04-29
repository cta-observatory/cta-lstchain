#!/usr/bin/env python3

"""
Pipeline for the reconstruction of Energy, disp and gamma/hadron
separation of events stored in a simtelarray file.

- Input: DL1 files and trained Random Forests.
- Output: DL2 data file.

Usage:

$> python lstchain_dl1_to_dl2.py 
--input-file dl1_LST-1.Run02033.0137.h5
--path-models ./trained_models

"""

from lstchain.reco import dl1_to_dl2
from sklearn.externals import joblib
import argparse
import os
import shutil
import pandas as pd

from lstchain.reco.utils import filter_events, impute_pointing
from lstchain.io import read_configuration_file, standard_config, replace_config
from lstchain.io import write_dl2_dataframe
from lstchain.io.io import dl1_params_lstcam_key, dl1_params_src_dep_lstcam_key
import numpy as np
import astropy.units as u


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


parser.add_argument('--config-file', '-c', action='store', type=str,
                    dest='config_file',
                    help='Path to a configuration file. If none is given, a standard configuration is applied',
                    default=None, required=False)



args = parser.parse_args()

def main():

    custom_config = {}
    if args.config_file is not None:
        try:
            custom_config = read_configuration_file(os.path.abspath(args.config_file))
        except("Custom configuration could not be loaded !!!"):
            pass

    config = replace_config(standard_config, custom_config)

    data = pd.read_hdf(args.input_file, key=dl1_params_lstcam_key)

    if config['source_dependent']:
        data = pd.concat([data, pd.read_hdf(data, key=dl1_params_src_dep_lstcam_key)], axis=1)
  
    # Dealing with pointing missing values. This happened when `ucts_time` was invalid.
    if 'alt_tel' in data.columns and 'az_tel' in data.columns \
            and (np.isnan(data.alt_tel).any() or np.isnan(data.az_tel).any()):
        # make sure there is a least one good pointing value to interp from.
        if np.isfinite(data.alt_tel).any() and np.isfinite(data.az_tel).any():
            data = impute_pointing(data)
        else:
            data.alt_tel = - np.pi/2.
            data.az_tel = - np.pi/2.
    data = filter_events(data, filters=config["events_filters"])

    #Load the trained RF for reconstruction:
    fileE = args.path_models + "/reg_energy.sav"
    fileD = args.path_models + "/reg_disp_vector.sav"
    fileH = args.path_models + "/cls_gh.sav"
    
    reg_energy = joblib.load(fileE)
    reg_disp_vector = joblib.load(fileD)
    cls_gh = joblib.load(fileH)
    
    #Apply the models to the data

    dl2 = dl1_to_dl2.apply_models(data, cls_gh, reg_energy, reg_disp_vector, custom_config=config)

    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, os.path.basename(args.input_file).replace('dl1','dl2'))


    shutil.copyfile(args.input_file, output_file)
    write_dl2_dataframe(dl2.astype(float), output_file)


if __name__ == '__main__':
    main()
