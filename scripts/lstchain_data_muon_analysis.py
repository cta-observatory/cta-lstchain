import argparse
import sys, os
import numpy as np
from ctapipe.image.extractor import LocalPeakWindowSum
from ctapipe.image.muon.features import ring_containment 
from ctapipe.image.muon.features import ring_completeness
from ctapipe.image.muon.features import npix_above_threshold
from ctapipe.image.muon.features import npix_composing_ring
from ctapipe.image.muon.muon_integrator import MuonLineIntegrate
from ctapipe.image.cleaning import tailcuts_clean
from ctapipe.instrument import CameraGeometry
from ctapipe.io.hdf5tableio import HDF5TableReader
from ctapipe.io.containers import FlatFieldContainer, WaveformCalibrationContainer, PedestalContainer
from ctapipe.io import event_source
from ctapipe.io import EventSeeker

from lstchain.calib.camera.r0 import LSTR0Corrections
from lstchain.image.muon import analyze_muon_event

from astropy.table import Table


'''
Script to perform the analysis of muon events.
To run it, type:

python lstchain_data_muon_analysis.py --input_file LST-1.4.Run00442.0001.fits.fz \
--output_file Data_table.fits --max_events 1000

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

# Optional argument
parser.add_argument("--max_events", help = "Maximum numbers of events to read."
                                         "Default = 100",
                    type = int, default = 100)

parser.add_argument("--plot_rings", help = "Plot figures of the stored rings", 
                    default = False, action='store_true')

parser.add_argument("--plots_path", help = "Path to the plots",
                    default = None, type = str)



args = parser.parse_args()

if __name__ == '__main__':
    print("input file: {}".format(args.input_file))
    print("output file: {}".format(args.output_file))
    print("pedestal file: {}".format(args.pedestal_file))
    print("calibration file: {}".format(args.calibration_file))
    print("max events: {}".format(args.max_events))
    
    #if sys.argv[1:]:
    #    run_name = sys.argv[1]
        #max_events = sys.argv[2]
    #    max_events = 10000
    #else:
    #    run_name = "/fefs/onsite/data/20190527/LST-1.4.Run00442.0001.fits.fz"
    #    max_events = None

    r0calib = LSTR0Corrections(
        #pedestal_path="/Users/rlopezcoto/Desktop/Data_LST_local/20190604/Data/pedestal_file_run446_0000.fits",
        pedestal_path = args.pedestal_file,
        r1_sample_start=2,r1_sample_end=38)


    ff_data = FlatFieldContainer()
    cal_data =  WaveformCalibrationContainer()
    ped_data =  PedestalContainer()

    dc_to_pe = []
    ped_median = []

    #with HDF5TableReader('/Users/rlopezcoto/Desktop/Data_LST_local/20190604/Calibration/calibration.hdf5') as h5_table:
    with HDF5TableReader(args.calibration_file) as h5_table:
        assert h5_table._h5file.isopen == True
        for cont in h5_table.read('/tel_0/pedestal', ped_data):
                ped_median = cont.charge_median
                
        for calib in h5_table.read('/tel_0/calibration', cal_data):
                dc_to_pe = calib['dc_to_pe']
    h5_table.close()

    geom = CameraGeometry.from_name("LSTCam-002")
    
    source = event_source(input_url = args.input_file, max_events = args.max_events)
    output_parameters = {'event_id': [],
                         'ring_size': [],
                         'size_outside': [],
                         'ring_radius': [],
                         'ring_width': [],
                         'muon_efficiency': [],
                         'ring_containment': [],
                         'ring_completeness': [],
                         'ring_pixel_completeness': [],
                         'impact_parameter': [],
                         'impact_x_array': [],
                         'impact_y_array': [],
                         'good_ring': [],
                         }

    num_muons = 0
    for event in source:

        event_id = event.lst.tel[0].evt.event_id
        teldes = event.inst.subarray.tel[0]
        r0calib.calibrate(event)
        pedcorrectedsamples = event.r1.tel[0].waveform
        integrator = LocalPeakWindowSum(window_shift=4, window_width=9)
        integration, pulse_time = integrator(pedcorrectedsamples)
        image = (integration - ped_median)*dc_to_pe

        print("Event {}. Number of pixels above 10 phe: {}".format(event_id,
                                                                  np.size(image[0][image[0] > 10.])))
        if((np.size(image[0][image[0]>10.]) > 300) or (np.size(image[0][image[0]>10.]) < 50)):
            continue

        muonintensityparam, size_outside_ring, muonringparam, good_ring = \
            analyze_muon_event(event_id, image, geom, teldes, 
                               args.plot_rings, args.plots_path)
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
        output_parameters['good_ring'].append(
        good_ring)

    table = Table(output_parameters)
    if os.path.exists(args.output_file):
            os.remove(args.output_file)
    table.write(args.output_file, format='fits')
