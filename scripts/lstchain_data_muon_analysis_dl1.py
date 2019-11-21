import argparse
import sys, os
import numpy as np
from ctapipe.image.muon.features import ring_containment 
from ctapipe.image.muon.features import ring_completeness
from ctapipe.image.muon.features import npix_above_threshold
from ctapipe.image.muon.features import npix_composing_ring
from ctapipe.image.muon.muon_integrator import MuonLineIntegrate
from ctapipe.image.cleaning import tailcuts_clean
from ctapipe.instrument import CameraGeometry
from ctapipe.io.hdf5tableio import HDF5TableReader
from astropy import units as u

from lstchain.image.muon import analyze_muon_event
from lstchain.io.io import dl1_params_lstcam_key, dl1_images_lstcam_key

from astropy.table import Table
import pandas as pd
import tables

'''
Script to perform the analysis of muon events.
To run it, type:

python lstchain_data_muon_analysis_dl1.py 
--input_file dl1_Run01566_0322.h5 
--output_file Data_table.fits 
'''

parser = argparse.ArgumentParser()

# Required arguments
parser.add_argument("--input_file", help = "Path to DL1a data file (containing charge information).",
                    type = str, default = "")

parser.add_argument("--output_file", help = "Path to create the output fits table with muon parameters",
                    type = str)

# Optional argument
parser.add_argument("--plot_rings", help = "Plot figures of the stored rings", 
                    default = False, action='store_true')

parser.add_argument("--plots_path", help = "Path to the plots",
                    default = None, type = str)

args = parser.parse_args()


def main():

    print("input file: {}".format(args.input_file))
    print("output file: {}".format(args.output_file))

    # Camera geometry
    geom = CameraGeometry.from_name("LSTCam-002")

    # Definition of the output parameters for the table
    output_parameters = {'event_id': [],
                         'ring_size': [],
                         'size_outside': [],
                         'ring_radius': [],
                         'ring_width': [],
                         'good_ring': [],
                         'muon_efficiency': [],
                         'ring_containment': [],
                         'ring_completeness': [],
                         'ring_pixel_completeness': [],
                         'impact_parameter': [],
                         'impact_x_array': [],
                         'impact_y_array': [],
                         }
    
    # image = pd.read_hdf(args.input_file, key = dl1_image_lstcam_key)
    # For some reason the call above is not working, this is a fix using
    # tables for the time being
    file = tables.open_file(args.input_file).root 
    group = file.dl1.event.telescope.image.LST_LSTCam  
    images = [x['image'] for x in group.iterrows()]
    
    parameters = pd.read_hdf(args.input_file, key = dl1_params_lstcam_key)
    telescope_description = pd.read_hdf(args.input_file, 
                                        key='instrument/telescope/optics')
    equivalent_focal_length = telescope_description['equivalent_focal_length'].values * u.m
    mirror_area = telescope_description['mirror_area'].values * pow(u.m,2)

    # File open
    num_muons = 0
    for image, event_id in zip(images, parameters['event_id']):

        print("Event {}. Number of pixels above 10 phe: {}".format(event_id,
                                                                  np.size(image[0][image[0] > 10.])))
        if((np.size(image[0][image[0]>10.]) > 300) or (np.size(image[0][image[0]>10.]) < 50)):
            continue

        muonintensityparam, size_outside_ring, muonringparam, good_ring = \
            analyze_muon_event(event_id, image, geom, equivalent_focal_length, 
                               mirror_area, args.plot_rings, args.plots_path)
        #if not (good_ring):
        #    continue
        print("Number of muons found {}, EventID {}".format(num_muons, event_id))

        num_muons = num_muons + 1

        output_parameters['event_id'].append(
        event_id)
        output_parameters['ring_size'].append(
        muonintensityparam.ring_size)
        output_parameters['size_outside'].append(
        size_outside_ring)
        output_parameters['ring_radius'].append(
        muonringparam.ring_radius.value)
        output_parameters['ring_width'].append(
        muonintensityparam.ring_width.value)
        output_parameters['good_ring'].append(
        good_ring)
        output_parameters['muon_efficiency'].append(
        muonintensityparam.optical_efficiency_muon)
        output_parameters['ring_containment'].append(
        muonringparam.ring_containment)
        output_parameters['ring_completeness'].append(
        muonintensityparam.ring_completeness)
        output_parameters['ring_pixel_completeness'].append(
        muonintensityparam.ring_pix_completeness)
        output_parameters['impact_parameter'].append(
        muonintensityparam.impact_parameter.value)
        output_parameters['impact_x_array'].append(
        muonintensityparam.impact_parameter_pos_x.value)
        output_parameters['impact_y_array'].append(
        muonintensityparam.impact_parameter_pos_y.value)

    table = Table(output_parameters)
    if os.path.exists(args.output_file):
            os.remove(args.output_file)
    table.write(args.output_file, format='fits')


if __name__ == '__main__':
    main()
