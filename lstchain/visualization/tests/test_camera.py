import astropy.units as u
import numpy as np
from ctapipe.instrument import CameraGeometry
from ctapipe.visualization import CameraDisplay

from lstchain.reco.disp import disp_parameters_event
from lstchain.visualization.camera import overlay_source, overlay_disp_vector, display_dl1_event


def test_overlay_source():
    geom = CameraGeometry.from_name('LSTCam')
    image = np.random.rand(geom.n_pixels)
    display = CameraDisplay(geom, image)
    overlay_source(display, 0.1 * u.m, 0.3 * u.m)


def test_overlay_disp_vector():
    from ctapipe.image import hillas_parameters

    geom = CameraGeometry.from_name('LSTCam')
    image = np.random.rand(geom.n_pixels)
    display = CameraDisplay(geom, image)
    hillas = hillas_parameters(geom, image)
    disp = disp_parameters_event(hillas, 0.1 * u.m, 0.3 * u.m)
    overlay_disp_vector(display, disp, hillas)


def test_display_dl1_event(mc_gamma_testfile):
    from ctapipe.io import EventSource, EventSeeker
    from ctapipe.calib import CameraCalibrator

    source = EventSource(mc_gamma_testfile, back_seekable=True)
    seeker = EventSeeker(source)
    event = seeker.get_event_index(11)  # event 11 has telescopes 1 and 4 with data
    CameraCalibrator(subarray=source.subarray)(event)
    display_dl1_event(event, source.subarray.tel[1].camera.geometry, tel_id=1)
    display_dl1_event(event, source.subarray.tel[4].camera.geometry, tel_id=4)
