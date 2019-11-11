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
import matplotlib.pyplot as plt


if __name__ == '__main__':
    
    if sys.argv[1:]:
        run_name = sys.argv[1]
        #max_events = sys.argv[2]
        max_events = 10000
    else:
        run_name = "/fefs/onsite/data/20190527/LST-1.4.Run00442.0001.fits.fz"
        max_events = None

    r0calib = LSTR0Corrections(
        #pedestal_path="Data/pedestal_run436.fits",
        #pedestal_path="./Calibration/pedestal_file_run446_0000.fits",
        pedestal_path="/Users/rlopezcoto/Desktop/Data_LST_local/20190604/Data/pedestal_file_run446_0000.fits",
        r1_sample_start=2,r1_sample_end=38)


    ff_data = FlatFieldContainer()
    cal_data =  WaveformCalibrationContainer()
    ped_data =  PedestalContainer()

    dc_to_pe = []
    ped_median = []

    with HDF5TableReader('/Users/rlopezcoto/Desktop/Data_LST_local/20190604/Calibration/calibration.hdf5') as h5_table:
        assert h5_table._h5file.isopen == True
        for cont in h5_table.read('/tel_0/pedestal', ped_data):
                ped_median = cont.charge_median
                
        for calib in h5_table.read('/tel_0/calibration', cal_data):
                dc_to_pe = calib['dc_to_pe']
    h5_table.close()

    geom = CameraGeometry.from_name("LSTCam-002")
    
    source = event_source(input_url = run_name, max_events = None)
    output_parameters = {'MuonEff': [],
                         'ImpactP': [],
                         'RingWidth': [],
                         'RingCont': [],
                         'RingComp': [],
                         'RingPixComp': [],
                         'Impact_x_arr': [],
                         'Impact_y_arr': [],
                         'RingSize': [],
                         'RingRadius': [],
                         'Event_id': []}

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

        plot_ring = False

        muonintensityparam, muonringparam, impact_condition = \
            analyze_muon_event(event_id, image, geom, teldes, plot_ring)
        #if not (impact_condition):
        #    continue
        print("Number of muons found, EventID", num_muons, event_id)

        num_muons = num_muons + 1

        output_parameters['MuonEff'].append(
        muonintensityparam.optical_efficiency_muon)
        output_parameters['ImpactP'].append(
        muonintensityparam.impact_parameter.value)
        output_parameters['RingWidth'].append(
        muonintensityparam.ring_width.value)
        output_parameters['RingCont'].append(
        muonringparam.ring_containment)
        output_parameters['RingComp'].append(
        muonintensityparam.ring_completeness)
        output_parameters['RingPixComp'].append(
        muonintensityparam.ring_pix_completeness)
        output_parameters['Impact_x_arr'].append(
        muonintensityparam.impact_parameter_pos_x.value)
        output_parameters['Impact_y_arr'].append(
        muonintensityparam.impact_parameter_pos_y.value)
        output_parameters['RingSize'].append(
        muonintensityparam.ring_size)
        output_parameters['RingRadius'].append(
        muonringparam.ring_radius.value)
        output_parameters['Event_id'].append(
        event_id)

    table = Table(output_parameters)
    outname = "./Data_table.fits"
    if os.path.exists(outname):
            os.remove(outname)
    table.write(outname,format='fits')
