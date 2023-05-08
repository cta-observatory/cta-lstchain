import astropy.units as u
from ctapipe.containers import CameraHillasParametersContainer
import numpy as np


def test_disp():
    from lstchain.reco.disp import disp_parameters_event

    hillas = CameraHillasParametersContainer(
        x=0.5 * u.m,
        y=1.0 * u.m,
        width=0.1 * u.m,
        length=0.3 * u.m,
        psi=np.arctan(0.5) * u.rad
    )

    # with the angle above, this source has alpha=0
    source_pos_x = 0.7 * u.m
    source_pos_y = 1.1 * u.m

    disp = disp_parameters_event(hillas, source_pos_x, source_pos_y)
    assert u.isclose(disp.norm, np.sqrt(0.05) * u.m)
    assert u.isclose(disp.miss, 0 * u.m, atol=1e-6 * u.m)

    assert u.isclose(disp.dx, 0.2 * u.m)
    assert u.isclose(disp.dy, 0.1 * u.m)
    assert u.isclose(disp.angle, hillas.psi)
    assert disp.sign == 1.0

    # with the angle above, this source has alpha=0
    source_pos_x = 0.3 * u.m
    source_pos_y = 0.9 * u.m

    disp = disp_parameters_event(hillas, source_pos_x, source_pos_y)
    assert disp.sign == -1.0
    assert u.isclose(disp.norm, np.sqrt(0.05) * u.m)
    assert u.isclose(disp.miss, 0 * u.m, atol=1e-6 * u.m)

    assert u.isclose(disp.dx, -0.2 * u.m)
    assert u.isclose(disp.dy, -0.1 * u.m)
    assert u.isclose(disp.angle, hillas.psi)
