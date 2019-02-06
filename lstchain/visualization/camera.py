import numpy as np
from ..reco.utils import disp_vector
import astropy.units as u


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
    hillas: `ctapipe.io.containers.HillasParametersContainer`
    kwargs: args for `matplotlib.pyplot.quiver`

    """
    assert np.isfinite([hillas.x.value, hillas.y.value]).all()
    if not np.isfinite([disp.dx.value, disp.dy.value]).all():
        disp_vector(disp)

    display.axes.quiver(hillas.x, hillas.y,
                        disp.dx, disp.dy,
                        units='xy', scale=1*u.m,
                        **kwargs,
                        )


def overlay_hillas_major_axis(display, hillas, **kwargs):
    """
    Overlay hillas ellipse major axis on a CameraDisplay.

    Parameters
    ----------
    display: `ctapipe.visualization.CameraDisplay`
    hillas: `ctapipe.io.containers.HillaParametersContainer`
    kwargs: args for `matplotlib.pyplot.plot`

    """
    kwargs['color'] = 'black' if 'color' not in kwargs else kwargs['color']

    length = hillas.length * 2
    x = -length + 2 * length * np.arange(10) / 10
    display.axes.plot(hillas.x + x * np.cos(hillas.psi.to(u.rad).value),
                      hillas.y + x * np.sin(hillas.psi.to(u.rad).value),
                      **kwargs,
                      )

