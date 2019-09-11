import numpy as np
from lstchain.visualization.camera import overlay_source, overlay_disp_vector
from lstchain.reco.disp import disp_parameters_event
from ctapipe.visualization import CameraDisplay
from ctapipe.instrument import CameraGeometry
import astropy.units as u

def test_overlay_disp_vector():

    from ctapipe.image import hillas_parameters

    geom = CameraGeometry.from_name('LSTCam')
    image = np.random.rand(geom.n_pixels)
    display = CameraDisplay(geom, image)
    hillas = hillas_parameters(geom, image)
    disp = disp_parameters_event(hillas, 0.1*u.m, 0.3*u.m)
    overlay_disp_vector(display, disp, hillas)


def test_overlay_source():
    geom = CameraGeometry.from_name('LSTCam')
    image = np.random.rand(geom.n_pixels)
    display = CameraDisplay(geom, image)
    overlay_source(display, 0.1, 0.3)