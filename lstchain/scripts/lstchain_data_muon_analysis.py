import argparse
import sys, os

from ctapipe.instrument import CameraGeometry
from ctapipe.io import event_source
import numpy as np

from lstchain.calib.camera.r0 import LSTR0Corrections
from lstchain.image.muon import analyze_muon_event, muon_filter, tag_pix_thr
from lstchain.calib.camera.calibrator import LSTCameraCalibrator
from traitlets.config.loader import Config
from astropy.table import Table


'''
Script to perform the analysis of muon events.
To run it, type:

python lstchain_data_muon_analysis.py 
--input_file LST-1.4.Run00442.0001.fits.fz 
--output_file Data_table.fits --pedestal_file pedestal_file_run446_0000.fits 
--calibration_file calibration.hdf5
--max_events 1000
--tel_id 0

'''

parser = argparse.ArgumentParser()

# Required arguments
parser.add_argument("--input_file", help = "Path to fits.fz data file.",
                    type = str, default = "")

parser.add_argument("--output_file", help = "Path to create the output fits table with muon parameters",
                    type = str)

parser.add_argument("--pedestal_file", help = "Path to the pedestal file",
                    type = str)

parser.add_argument("--calibration_file", help = "Path to the file containing the calibration constants",
                    type = str)

parser.add_argument("--time_calibration_file", help = "Path to the calibration file with time corrections ",
                    type = str, default='')

# Optional argument
parser.add_argument("--max_events", help = "Maximum numbers of events to read. "
                                         "Default = 100",
                    type = int, default = 100)
parser.add_argument("--gain_threshold", help = "Gain threshold in ADC. "
                                               "Default = 3500", type = int, default=3500)

parser.add_argument("--plot_rings", help = "Plot figures of the stored rings", 
                    default = False, action='store_true')

parser.add_argument("--plots_path", help = "Path to the plots",
                    default = None, type = str)
parser.add_argument("--tel_id", help = "telescope id. "
                                       "Default = 1",type = int, default = 1)


args = parser.parse_args()


def main():

    print("input file: {}".format(args.input_file))
    print("output file: {}".format(args.output_file))
    print("pedestal file: {}".format(args.pedestal_file))
    print("calibration file: {}".format(args.calibration_file))
    print("time calibration file: {}".format(args.time_calibration_file))
    print("max events: {}".format(args.max_events))

    # Camera geometry
    #geom = CameraGeometry.from_name("LSTCam-003")

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

    tel_id = args.tel_id

    # Low level calibration
    r0calib = LSTR0Corrections(
        pedestal_path = args.pedestal_file, tel_id=tel_id)

    # high level calibration and gain selection
    charge_config = Config({
        "LocalPeakWindowSum": {
            "window_shift": 4,
            "window_width": 8
        }
    })

    r1_dl1_calibrator = LSTCameraCalibrator(calibration_path=args.calibration_file,
                                            time_calibration_path=args.time_calibration_file,
                                            image_extractor="LocalPeakWindowSum",
                                            config=charge_config,
                                            gain_threshold=args.gain_threshold,
                                            allowed_tels=[tel_id])


    # Maximum number of events
    if args.max_events:
        max_events = args.max_events
    else:
        max_events = None

    # File open
    num_muons = 0
    source = event_source(input_url = args.input_file, max_events = max_events)

    # geometry
    subarray = source.subarray(tel_id)
    telescope_description = subarray.tel[tel_id]
    equivalent_focal_length = telescope_description.optics.equivalent_focal_length
    mirror_area = telescope_description.optics.mirror_area.to("m2")
    geom = telescope_description.camera


    for event in source:
        event_id = event.lst.tel[tel_id].evt.event_id

        # drs4 calibration
        r0calib.calibrate(event)

        # r1 calibration
        r1_dl1_calibrator(event)
        image = event.dl1.tel[tel_id].image

        if not tag_pix_thr(image): #default skipps pedestal and calibration events
            continue

        if not muon_filter(image,800,5000): #default values apply no filtering
            continue


        print("--> Event {}. Number of pixels above 10 phe: {}".format(event_id,
                                                                   np.size(image[image > 10.])))


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
