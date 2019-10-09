#!/bin/env python
import sys, os
import numpy as np
from ctapipe.image.extractor import LocalPeakWindowSum
from ctapipe.image.muon.features import ring_completeness
from ctapipe.image.muon.features import npix_above_threshold
from ctapipe.image.muon.features import npix_composing_ring
from ctapipe.image.muon.muon_integrator import MuonLineIntegrate
from ctapipe.image.cleaning import tailcuts_clean
from ctapipe.instrument import CameraGeometry

from astropy.coordinates import Angle, SkyCoord, AltAz
from ctapipe.image.muon.muon_ring_finder import ChaudhuriKunduRingFitter
from ctapipe.coordinates import CameraFrame, NominalFrame
from astropy import units as u

from ctapipe.io.hdf5tableio import HDF5TableWriter, HDF5TableReader
from ctapipe.io.containers import FlatFieldContainer, WaveformCalibrationContainer, PedestalContainer
from ctapipe.io import event_source
from ctapipe.io import EventSeeker

from lstchain.calib.camera.r0 import LSTR0Corrections
from astropy.table import Table

def get_xy():
    x, y = geom.pix_x, geom.pix_y

    telescope_pointing = SkyCoord(
            alt=70 * u.deg,
            az=0 * u.deg,
            frame=AltAz()
        )

    camera_coord = SkyCoord(
            x=x, y=y,
            frame=CameraFrame(
                focal_length=teldes.optics.equivalent_focal_length,
                rotation=geom.pix_rotation,
                telescope_pointing=telescope_pointing
            )
    )
    nom_coord = camera_coord.transform_to(
            NominalFrame(origin=telescope_pointing)
        )


    x = nom_coord.delta_az.to(u.deg)
    y = nom_coord.delta_alt.to(u.deg)

    return x,y

def fit_muon(x,y,phe):

    muonring = ChaudhuriKunduRingFitter(None)
    clean_mask = tailcuts_clean(geom, phe, picture_thresh=tailcuts[0],
                                    boundary_thresh=tailcuts[1])
    image = phe * clean_mask
    muonringparam = muonring.fit(x, y, image)

    dist = np.sqrt(np.power(x - muonringparam.ring_center_x, 2)
                       + np.power(y - muonringparam.ring_center_y, 2))
    ring_dist = np.abs(dist - muonringparam.ring_radius)
    muonringparam = muonring.fit(
            x, y, image * (ring_dist < muonringparam.ring_radius * 0.4)
        )

    dist = np.sqrt(np.power(x - muonringparam.ring_center_x, 2) +
                       np.power(y - muonringparam.ring_center_y, 2))
    ring_dist = np.abs(dist - muonringparam.ring_radius)

    muonringparam = muonring.fit(
            x, y, image * (ring_dist < muonringparam.ring_radius * 0.4)
        )
    
    return muonringparam, clean_mask, dist, image

def analyze_muon_event(phe):

    x, y = get_xy()
    muonringparam, clean_mask, dist, image = fit_muon(x,y,phe[0])

    mir_rad = np.sqrt(teldes.optics.mirror_area.to("m2") / np.pi)
    dist_mask = np.abs(dist - muonringparam.ring_radius
                    ) < muonringparam.ring_radius * 0.4
    pix_im = phe[0] * dist_mask
    nom_dist = np.sqrt(np.power(muonringparam.ring_center_x,2) 
                    + np.power(muonringparam.ring_center_y, 2))

    ctel = MuonLineIntegrate(
                mir_rad, hole_radius = 0.308 * u.m,
                pixel_width=0.1 * u.deg,
                sct_flag=False,
                secondary_radius = 0. * u.m
            )

    muonintensityoutput = ctel.fit_muon(muonringparam.ring_center_x,
                                    muonringparam.ring_center_y,
                                    muonringparam.ring_radius,
                                    x[dist_mask], y[dist_mask],
                                    phe[0][dist_mask])
    muonintensityoutput.mask = dist_mask
    idx_ring = np.nonzero(pix_im)
    muonintensityoutput.ring_completeness = ring_completeness(
                    x[idx_ring], y[idx_ring], pix_im[idx_ring],
                    muonringparam.ring_radius,
                    muonringparam.ring_center_x,
                    muonringparam.ring_center_y,
                    threshold=30,
                    bins=30)
    muonintensityoutput.ring_size = np.sum(pix_im)
    dist_ringwidth_mask = np.abs(dist - muonringparam.ring_radius
                                             ) < (muonintensityoutput.ring_width)
    pix_ringwidth_im = phe[0] * dist_ringwidth_mask
    idx_ringwidth = np.nonzero(pix_ringwidth_im)

    muonintensityoutput.ring_pix_completeness = npix_above_threshold(
                    pix_ringwidth_im[idx_ringwidth], tailcuts[0]) / len(
                    pix_im[idx_ringwidth])

    print("Impact parameter = %s"
                             "ring_width=%s, ring radius=%s, ring completeness=%s"% (
                             muonintensityoutput.impact_parameter,
                             muonintensityoutput.ring_width,
                             muonringparam.ring_radius,
                             muonintensityoutput.ring_completeness))

    altaz = AltAz(alt = 70 * u.deg, az = 0 * u.deg)
    flen = event.inst.subarray.tel[0].optics.equivalent_focal_length
    ring_nominal = SkyCoord(
                delta_az=muonringparam.ring_center_x,
                delta_alt=muonringparam.ring_center_y,
                frame=NominalFrame(origin=altaz)
            )

    ring_camcoord = ring_nominal.transform_to(CameraFrame(
                focal_length=flen,
                rotation=geom.pix_rotation,
                telescope_pointing=altaz))
    centroid = (ring_camcoord.x.value, ring_camcoord.y.value)
    radius = muonringparam.ring_radius
    width = muonintensityoutput.ring_width
    ringrad_camcoord = 2 * radius.to(u.rad) * flen
    ringwidthfrac = width / radius
    ringrad_inner = ringrad_camcoord * (1. - ringwidthfrac)
    ringrad_outer = ringrad_camcoord * (1. + ringwidthfrac)

    return centroid, ringrad_camcoord, ringrad_inner, ringrad_outer, muonintensityoutput, muonringparam, clean_mask

if __name__ == '__main__':
    
    if sys.argv[1:]:
        run_name = sys.argv[1]
        max_events = sys.argv[2]
    else:
        run_name = "/fefs/onsite/data/20190527/LST-1.4.Run00442.0001.fits.fz"
        max_events = None

    r0calib = LSTR0Corrections(
        #pedestal_path="Data/pedestal_run436.fits",
        #pedestal_path="./Calibration/pedestal_file_run446_0000.fits",
        pedestal_path="./Calibration/pedestal_run582_00.fits",
        r1_sample_start=2,r1_sample_end=38)


    ff_data = FlatFieldContainer()
    cal_data =  WaveformCalibrationContainer()
    ped_data =  PedestalContainer()

    dc_to_pe = []
    ped_median = []

    with HDF5TableReader('./Calibration/calibration.hdf5') as h5_table:
        assert h5_table._h5file.isopen == True
        for cont in h5_table.read('/tel_0/pedestal', ped_data):
                ped_median = cont.charge_median
                #print(ped_median)
                #cont.meta()
                
        for calib in h5_table.read('/tel_0/calibration', cal_data):
                #print(calib.as_dict())
                #print(calib)
                dc_to_pe = calib['dc_to_pe']
    h5_table.close()

    geom = CameraGeometry.from_name("LSTCam-002")
    tailcuts = [3, 6]
    
    N_modules = 7*265
    
    source = event_source(input_url=run_name,max_events=max_events)
    tab = {'Event_id': [],
           'Size': [],
           'RingComp': []}
    i = 0
    for event in source:
        std_signal = np.zeros(1855)
        for pixel in range(0, N_modules):
            std_signal[pixel] = np.max(event.r0.tel[0].waveform[0, pixel, 2:38])   
        if((np.size(std_signal[std_signal>800.]) > 300) or (np.size(std_signal[std_signal>800.]) < 80)):
            continue
       
        i += 1
        print(i)
        r0calib.calibrate(event)
        pedcorrectedsamples = event.r1.tel[0].waveform
        integrator = LocalPeakWindowSum(window_shift=4, window_width=9)
        integration, pulse_time = integrator(pedcorrectedsamples)
        phe = (integration - ped_median)*dc_to_pe
        teldes = event.inst.subarray.tel[0]
        centroid, ringrad_camcoord, ringrad_inner, ringrad_outer, muonintensityoutput, muonringparam, clean_mask = analyze_muon_event(phe)
        rc = muonintensityoutput.ring_completeness

        tab['Event_id'].append(event.lst.tel[0].evt.event_id)
        tab['Size'].append(phe[0].sum())
        tab['RingComp'].append(rc)

    table = Table(tab)
    outname = "./Data_table.fits"
    if os.path.exists(outname):
            os.remove(outname)
    table.write(outname,format='fits')








