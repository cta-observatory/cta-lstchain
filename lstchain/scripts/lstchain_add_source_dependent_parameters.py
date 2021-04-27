#!/usr/bin/env python3

"""
Script to add the source dependent parameters to a DL1 file.

Input: DL1 data file. Source dependent parameters will be added to this file. 

Usage: 

$> lstchain_add_source_dependent_parameters
--input-file dl1_LST-1.Run02033.0137.h5 
--config lstchain_src_dep_config.json

"""

import os
import argparse
import pandas as pd
from tables import open_file
from distutils.util import strtobool

from ctapipe.instrument import SubarrayDescription
from lstchain.reco.dl1_to_dl2 import get_source_dependent_parameters
from lstchain.io import read_configuration_file, get_standard_config, get_dataset_keys
from lstchain.io.io import(
    dl1_params_lstcam_key,
    dl1_images_lstcam_key,
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
parser.add_argument('--output-dir', '-o', action='store', type=str,
                    dest='output_dir',
                    help='Path where to store the dl1 files with source-dependent parameters',
                    default='./dl1_data_with_srcdep')

parser.add_argument('--no-image', action='store', type=lambda x: bool(strtobool(x)),
                    dest='noimage',
                    help='Boolean. True to remove the images',
                    default=True)

parser.add_argument('--overwrite', action='store', type=lambda x: bool(strtobool(x)),
                    dest='overwrite',
                    help='Boolean. True to overwrite the original dl1 file',
                    default=False)

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


    if args.overwrite:
        write_dataframe(src_dep_df, dl1_filename, dl1_params_src_dep_lstcam_key)

    else:

        os.makedirs(args.output_dir, exist_ok=True)
        output_file = os.path.join(args.output_dir, os.path.basename(args.input_file))

        if os.path.exists(output_file):
            raise IOError(output_file + ' exists, exiting.')

        dl1_keys = get_dataset_keys(args.input_file)

        if args.noimage and dl1_images_lstcam_key in dl1_keys:
            dl1_keys.remove(dl1_images_lstcam_key)
            
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

        write_dataframe(src_dep_df, output_file, dl1_params_src_dep_lstcam_key)

if __name__ == '__main__':
    main()

