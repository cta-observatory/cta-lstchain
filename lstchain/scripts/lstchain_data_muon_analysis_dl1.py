import argparse
import sys, os
import glob
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

from lstchain.image.muon import analyze_muon_event, muon_filter, tag_pix_thr, fill_muon_event
from lstchain.io.io import dl1_params_lstcam_key, dl1_images_lstcam_key
from lstchain.visualization import plot_calib

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
                    type = str, required=True)

parser.add_argument("--calib_file", help = "Path to corresponding calibration file (containing bad pixel information).",
                    type = str, required=True)

parser.add_argument("--output_file", help = "Path to create the output fits table with muon parameters",
                    type = str, required=True)

# Optional argument
parser.add_argument("--plot_rings", help = "Plot figures of the stored rings", 
                    default = False, action='store_true')

parser.add_argument("--plots_path", help = "Path to the plots",
                    default = None, type = str)

parser.add_argument("--max_muons", help = "Maximum number of processed muon ring candidates",
                    default = np.nan, type = int)

args = parser.parse_args()


def main():

    print("input files: {}".format(args.input_file))
    print("calib file: {}".format(args.calib_file))
    print("output file: {}".format(args.output_file))
    
    max_muons = args.max_muons

    # Camera geometry
    geom = CameraGeometry.from_name("LSTCam-004")

    # Definition of the output parameters for the table
    output_parameters = {'event_id': [],
                         'event_time': [],
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
                         'radial_stdev' : [],                  # Standard deviation of (cleaned) light distribution along ring radius
                         'radial_skewness' : [],               # Skewness of (cleaned) light distribution along ring radius
                         'radial_excess_kurtosis' : [],        # Excess kurtosis of (cleaned) light distribution along ring radius
                         'num_pixels_in_ring' : [],            # pixels inside the integration area around the ring
                         'mean_pixel_charge_around_ring' : []  # Average pixel charge in pixels surrounding the outer part of the ring 
                         }

    plot_calib.read_file(args.calib_file)
    bad_pixels = plot_calib.calib_data.unusable_pixels[0]
    print(f"Found a total of {np.sum(bad_pixels)} bad pixels.")

    # image = pd.read_hdf(args.input_file, key = dl1_image_lstcam_key)
    # The call above does not work, because of the file's vector columns (pixel-wise charges & times)
    # So we use tables for the time being.

    print(glob.glob(args.input_file))

    filenames = glob.glob(args.input_file)
    filenames.sort()
    
    num_muons = 0
    
    for filename in filenames:
        print('Opening file', filename)

        with tables.open_file(filename) as file:
            
            # unfortunately pandas.read_hdf does not seem compatible with "with... as..." statements
            parameters = pd.read_hdf(filename, key = dl1_params_lstcam_key)
            telescope_description = pd.read_hdf(filename, key='instrument/telescope/optics')

            group = file.root.dl1.event.telescope.image.LST_LSTCam  
            images = [x['image'] for x in group.iterrows()]
        
            equivalent_focal_length = telescope_description['equivalent_focal_length'].values * u.m
            mirror_area = telescope_description['mirror_area'].values * pow(u.m,2)
            
            for full_image, event_id, dragon_time in zip(images, parameters['event_id'], parameters['dragon_time']):
                image = full_image*(~bad_pixels)
                #print("Event {}. Number of pixels above 10 phe: {}".format(event_id,
                #                                                          np.size(image[image > 10.])))
                #if((np.size(image[image > 10.]) > 300) or (np.size(image[image > 10.]) < 50)):
                #    continue
                if not tag_pix_thr(image): #default skips pedestal and calibration events
                    continue
                
                #if not muon_filter(image) #default values apply no filtering. This filter is rather useless for biased extractors anyway
                #    continue
            
                muonintensityparam, size_outside_ring, muonringparam, good_ring, \
                    radial_distribution, mean_pixel_charge_around_ring = analyze_muon_event(event_id, image, geom, equivalent_focal_length, 
                                                                                            mirror_area, args.plot_rings, args.plots_path)
        
                if good_ring:
                    num_muons = num_muons + 1
                    print("Number of good muon rings found {}, EventID {}".format(num_muons, event_id))

                # write ring data, including also "not-so-good" rings, in case we want to reconsider ring selections!:

                fill_muon_event(output_parameters, good_ring, event_id, dragon_time, muonintensityparam, muonringparam,
                                radial_distribution, size_outside_ring, mean_pixel_charge_around_ring)
                    
                if num_muons == max_muons:
                    break

            if num_muons == max_muons:
                break

    table = Table(output_parameters)
    if os.path.exists(args.output_file):
            os.remove(args.output_file)
    table.write(args.output_file, format='fits')

if __name__ == '__main__':
    main()
