#!/usr/bin/env python3

"""
Script to add the source dependent parameters to a DL1 file.

Input: DL1 data file. Source dependent parameters will be added to this file. 

Usage: 

$> python lstchain_add_source_dependent_parameters.py 
--input-file dl1_LST-1.Run02033.0137.h5 
--config lstchain_src_dep_config.json

"""

import os
import argparse
import pandas as pd
from ctapipe.instrument import SubarrayDescription
from lstchain.reco.dl1_to_dl2 import get_source_dependent_parameters
from lstchain.io import read_configuration_file, get_standard_config
from lstchain.io.io import(
    dl1_params_lstcam_key,
    dl1_params_src_dep_lstcam_key,
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
    subarray_info = SubarrayDescription.from_hdf(dl1_filename)
    tel_id = config["allowed_tels"][0] if "allowed_tels" in config else 1
    focal_length = subarray_info.tel[tel_id].optics.equivalent_focal_length
 
    src_dep_df = pd.concat(get_source_dependent_parameters(dl1_params, config, focal_length=focal_length), axis=1)

    write_dataframe(src_dep_df, dl1_filename, dl1_params_src_dep_lstcam_key)

if __name__ == '__main__':
    main()

