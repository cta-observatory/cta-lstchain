#!/usr/bin/env python3

"""
Script to add the source dependent parameters to a DL1 file.

Input: DL1 data file. Source dependent parameters will be added to this file. 

Usage: 

$> python lstchain_add_source_dependent_parameters.py 
--input-file dl1_LST-1.Run02033.0137.h5 

"""

import os
import argparse
import pandas as pd
import numpy as np
from lstchain.reco.dl1_to_dl2 import get_source_dependent_parameters
from lstchain.reco.utils import radec_to_camera
from lstchain.io import read_configuration_file, get_standard_config
from lstchain.io.io import(
    dl1_params_lstcam_key,
    dl1_params_src_dep_on_lstcam_key,
    dl1_params_src_dep_off_lstcam_key,
    write_dataframe,
)


parser = argparse.ArgumentParser(description="Add the source dependent parameters to a DL1 file")

# Required arguments
parser.add_argument('--input-file', '-f', type=str,
                    dest='input_file',
                    help='path to a DL1 HDF5 file',
                    )

# Optional arguments
parser.add_argument('--config', '-c', action='store', type=str,
                    dest='config_file',
                    help='Path to a configuration file for source dependent analysis',
                    default=None
                    )

args = parser.parse_args()

def main():

    dl1_filename = os.path.abspath(args.input_file)

    config = get_standard_config()
    if args.config_file is not None:
        try:
            config = read_configuration_file(os.path.abspath(args.config_file))
        except("Custom configuration could not be loaded !!!"):                                                                                            
            pass

    dl1_params = pd.read_hdf(dl1_filename, key=dl1_params_lstcam_key)
 
    if config['observation_mode']=='on':
        src_dep_df = get_source_dependent_parameters(dl1_params, config)
        write_dataframe(src_dep_df, dl1_filename, dl1_params_src_dep_on_lstcam_key)

    if config['observation_mode']=='wobble':
        # ON position
        src_dep_df = get_source_dependent_parameters(dl1_params, config, wobble_angle = 0.)
        write_dataframe(src_dep_df, dl1_filename, dl1_params_src_dep_on_lstcam_key)
        
        # OFF position
        for ioff in range(config['n_off_wobble']):
            wobble_angle = 2 * np.pi / (config['n_off_wobble'] + 1) * (ioff + 1)
            src_dep_df = get_source_dependent_parameters(dl1_params, config, wobble_angle = wobble_angle)
            write_dataframe(src_dep_df, dl1_filename,  dl1_params_src_dep_off_lstcam_key.replace('*', '{:.0f}'.format(np.rad2deg(wobble_angle))))
 

if __name__ == '__main__':
    main()
 
