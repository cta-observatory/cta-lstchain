import numpy as np
from ..reco.disp import disp_vector
import astropy.units as u
import matplotlib.pyplot as plt
from ctapipe.visualization import CameraDisplay
from ctapipe.instrument import CameraGeometry

__all__ = [
    'overlay_disp_vector',
    'overlay_hillas_major_axis',
    'overlay_source',
    'display_dl1_event',
]

def display_dl1_event(event, camera_geometry, tel_id=1, axes=None, **kwargs):
    """
    Display a DL1 event (image and pulse time map) side by side

    Parameters
    ----------
    event: ctapipe event
    tel_id: int
    axes: list of `matplotlib.pyplot.axes` of shape (2,) or None
    kwargs: kwargs for `ctapipe.visualization.CameraDisplay`

    Returns
    -------
    axes: `matplotlib.pyplot.axes`
    """

    if axes is None:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    image = event.dl1.tel[tel_id].image
    peak_time = event.dl1.tel[tel_id].peak_time

    if image is None or peak_time is None:
        raise Exception(f"There is no calibrated image or pulse time map for telescope {tel_id}")

    d1 = CameraDisplay(camera_geometry, image, ax=axes[0], **kwargs)
    d1.add_colorbar(ax=axes[0])
    d2 = CameraDisplay(camera_geometry, peak_time, ax=axes[1], **kwargs)
    d2.add_colorbar(ax=axes[1])

    return axes


def overlay_source(display, source_pos_x, source_pos_y, **kwargs):
    """
    Display the source (event) position in the camera

    Parameters
    ----------
    display: `ctapipe.visualization.CameraDisplay`
    source_pos_x: `astropy.units.Quantity`
    source_pos_y: `astropy.units.Quantity`
    kwargs: args for `matplotlib.pyplot.scatter`

    Returns
    -------
    `matplotlib.pyplot.axes`
    """
    kwargs['marker'] = 'x' if 'marker' not in kwargs else kwargs['marker']
    kwargs['color'] = 'red' if 'color' not in kwargs else kwargs['color']
    display.axes.scatter(source_pos_x, source_pos_y, **kwargs)


def overlay_disp_vector(display, disp, hillas, **kwargs):
    """
    Overlay disp vector on a CameraDisplay

    Parameters
    ----------
    display: `ctapipe.visualization.CameraDisplay`
    disp: `DispContainer`
    hillas: `ctapipe.containers.HillasParametersContainer`
    kwargs: args for `matplotlib.pyplot.quiver`

    """
    assert np.isfinite([hillas.x.value, hillas.y.value]).all()
    if not np.isfinite([disp.dx.value, disp.dy.value]).all():
        disp_vector(disp)

    display.axes.quiver(hillas.x, hillas.y,
                        disp.dx, disp.dy,
                        units='xy', scale=1*u.m,
                        angles='xy',
                        **kwargs,
                        )

    display.axes.quiver(hillas.x.value, hillas.y.value, disp.dx.value, disp.dy.value, units='xy', scale=1)


def overlay_hillas_major_axis(display, hillas, **kwargs):
    """
    Overlay hillas ellipse major axis on a CameraDisplay.

    Parameters
    ----------
    display: `ctapipe.visualization.CameraDisplay`
    hillas: `ctapipe.containers.HillaParametersContainer`
    kwargs: args for `matplotlib.pyplot.plot`

    """
    kwargs['color'] = 'black' if 'color' not in kwargs else kwargs['color']

    length = hillas.length * 2
    x = -length + 2 * length * np.arange(10) / 10
    display.axes.plot(hillas.x + x * np.cos(hillas.psi.to(u.rad).value),
                      hillas.y + x * np.sin(hillas.psi.to(u.rad).value),
                      **kwargs,
                      )

