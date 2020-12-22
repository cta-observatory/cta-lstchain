#!/usr/bin/env python3

"""
- Input: DL1 file including DL1a
- Output: DL1 file after the reprocess with a new configulation.
Usage:
$> python lstchain_data_reprocess_dl1.py 
--input-file dl1_LST-1.Run02033.0137.h5
--output-file out.h5
--config config.json
--no-image True
"""

import astropy.units as u
import logging
import tables
import numpy as np
import pandas as pd
import argparse
from tables import open_file
import argparse
from distutils.util import strtobool
import os

from ctapipe.containers import EventIndexContainer
from ctapipe.io import HDF5TableWriter
from ctapipe_io_lst.containers import LSTDataContainer
from lstchain.io import standard_config, replace_config, get_dataset_keys, read_configuration_file
from lstchain.io.io import dl1_params_lstcam_key, dl1_images_lstcam_key
from lstchain.io import DL1ParametersContainer
from lstchain.io.subarray import SubarrayDescription
from lstchain.reco.r0_to_dl1 import get_dl1


logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description="DL1 reprocess")

# Required arguments
parser.add_argument('--input-file', '-f', type=str,
                    dest='input_file',
                    help='path to a DL1 HDF5 file',
                    default=None, required=True)

parser.add_argument('--output-file', '-o', type=str,
                     dest='output_file',
                     help='Path where to store the new dl1 events',
                     default='test_dl1.h5')

parser.add_argument('--config', '-c', action='store', type=str,
                    dest='config_file',
                    help='Path to a configuration file. If none is given, a standard configuration is applied',
                    default=None, required=False)

parser.add_argument('--no-image', action='store', type=lambda x: bool(strtobool(x)),
                    dest='noimage',
                    help='Boolean. True to remove the images',
                    default=True)

args = parser.parse_args()

def main():

    custom_config = {}
    if args.config_file is not None:
        try:
            custom_config = read_configuration_file(os.path.abspath(args.config_file))
        except("Custom configuration could not be loaded !!!"):
            pass

    config = replace_config(standard_config, custom_config)
    
    input_file = args.input_file
    output_file = args.output_file

    filters = tables.Filters(
        complevel=5,            # enable compression, with level 0=disabled, 9=max
        complib='blosc:zstd',   # compression using blosc
        fletcher32=True,        # attach a checksum to each chunk for error correction
        bitshuffle=False,       # for BLOSC, shuffle bits for better compression
    )

    subarray = SubarrayDescription.from_hdf(input_file)
    tel_id=1
    tel_name = str(subarray.tel[tel_id])[4:]

    is_simu = 'simulation/run_config' in get_dataset_keys(input_file)
    
    with open_file(input_file, 'r') as h5in:    
        with HDF5TableWriter(
                filename=output_file,
                group_name='dl1/event',
                mode='a',
                filters=filters,
                add_prefix=True,
                overwrite=True,
        ) as h5out:
            
            data = h5in.root
        
            # Read image parameters
            image_group = data.dl1.event.telescope.image.LST_LSTCam
        
            image = np.array([x['image'] for x in image_group.iterrows()])
            peak_time = np.array([x['peak_time'] for x in image_group.iterrows()])
            tel_id = np.array([x['tel_id'] for x in image_group.iterrows()])
        
            # Read DL1 parameters
            dl1_group = data.dl1.event.telescope.parameters.LST_LSTCam

            dl1_container = DL1ParametersContainer() 
            index_container = EventIndexContainer()
        
            for i, x in enumerate(dl1_group.iterrows()):

                if i % 100 == 0:
                    logger.info(i)
                    
                # prepare container
                event = LSTDataContainer()
            
                event.dl1.tel[tel_id[i]].image = image[i]
                event.dl1.tel[tel_id[i]].peak_time = peak_time[i]
            
                if is_simu and config['mc_image_scaling_factor'] != 1:
                    event.dl1.tel[tel_id[i]].image *= config['mc_image_scaling_factor']
            
                dl1_container.reset()
                
            
                # get dl1 parameters after image cleaning
                get_dl1(event,
                    subarray,
                    tel_id[i],
                    dl1_container=dl1_container,
                    custom_config=config,
                    use_main_island=True)
            
                # other dl1 parameters            
                if is_simu:
                    dl1_container.calibration_id = x['calibration_id']
                    dl1_container.mc_energy = u.Quantity(x['mc_energy'], u.TeV)
                    dl1_container.log_mc_energy = x['log_mc_energy']
                    dl1_container.mc_alt = u.Quantity(x['mc_alt'], u.rad)
                    dl1_container.mc_az = u.Quantity(x['mc_az'], u.rad)
                    dl1_container.mc_core_x = u.Quantity(x['mc_core_x'], u.m)
                    dl1_container.mc_core_y = u.Quantity(x['mc_core_y'], u.m)
                    dl1_container.mc_h_first_int = u.Quantity(x['mc_h_first_int'], u.m)
                    dl1_container.mc_type = x['mc_type']
                    dl1_container.mc_az_tel = u.Quantity(x['mc_az_tel'], u.rad)
                    dl1_container.mc_alt_tel = u.Quantity(x['mc_alt_tel'], u.rad)
                    dl1_container.mc_x_max = u.Quantity(x['mc_x_max'], u.g / u.cm / u.cm)
                    dl1_container.mc_core_distance = u.Quantity(x['mc_core_distance'], u.m)
                    dl1_container.tel_id = x['tel_id']
                    dl1_container.tel_pos_x = x['tel_pos_x']
                    dl1_container.tel_pos_y = x['tel_pos_y']
                    dl1_container.tel_pos_z = x['tel_pos_z']
                    dl1_container.trigger_type = x['trigger_type']
                    dl1_container.disp_dx = u.Quantity(x['disp_dx'], u.m)
                    dl1_container.disp_dy = u.Quantity(x['disp_dy'], u.m)
                    dl1_container.disp_norm = u.Quantity(x['disp_norm'], u.m)
                    dl1_container.disp_angle = u.Quantity(x['disp_angle'], u.rad)
                    dl1_container.disp_sign = x['disp_sign']
                    dl1_container.src_x = u.Quantity(x['src_x'], u.m)
                    dl1_container.src_y = u.Quantity(x['src_y'], u.m)
                    
                else:
                    dl1_container.alt_tel = u.Quantity(x['alt_tel'], u.rad)
                    dl1_container.az_tel = u.Quantity(x['az_tel'], u.rad)
                    dl1_container.calibration_id = x['calibration_id']
                    dl1_container.dragon_time = x['dragon_time']
                    dl1_container.ucts_time = x['ucts_time']
                    dl1_container.tib_time = x['tib_time']
                    dl1_container.tel_id = x['tel_id']
                    dl1_container.tel_pos_x = x['tel_pos_x']
                    dl1_container.tel_pos_y = x['tel_pos_y']
                    dl1_container.tel_pos_z = x['tel_pos_z']
                    dl1_container.trigger_type = x['trigger_type']
                    dl1_container.ucts_trigger_type = x['ucts_trigger_type']
                    dl1_container.trigger_time = x['trigger_time']

                dl1_container.prefix = ''
                
                # index container
                index_container.reset()
                index_container = EventIndexContainer()
                index_container.obs_id = x['obs_id']
                index_container.event_id = x['event_id']          

                h5out.write(table_name=f'telescope/parameters/{tel_name}', containers=[dl1_container, index_container])
                              
        
        with open_file(output_file, 'a') as h5out:
        
            keys = get_dataset_keys(input_file)
 
            if args.noimage:
                if dl1_images_lstcam_key in keys:
                    keys.remove(dl1_images_lstcam_key)
                
            if dl1_params_lstcam_key in keys:
                keys.remove(dl1_params_lstcam_key)
            
            # Write the selected DL1 info
            for k in keys:
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


if __name__ == '__main__':
    main()
