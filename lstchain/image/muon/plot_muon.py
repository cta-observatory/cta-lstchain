from ctapipe.visualization import CameraDisplay

__all__ = ['plot_muon_event']


def plot_muon_event(ax, geom, image, centroid, ringrad_camcoord,
                    ringrad_inner, ringrad_outer, event_id):
    """
    Function to plot single muon events

    Parameters
    ----------
    image
    ax : `matplotlib.pyplot.axis`
    geom : CameraGeometry
    centroid : `float`
        Centroid of the muon ring
    ringrad_camcoord : `float`
        Ring radius in camera coordinates
    ringrad_inner : `float`
        Inner ring radius in camera coordinates
    ringrad_outer : `float`
        Outer ring radius in camera coordinates
    event_id : `int`
        ID of the analyzed event

    Returns
    -------
    ax : `matplotlib.pyplot.axis`

    """

    disp0 = CameraDisplay(geom, ax=ax)
    disp0.image = image
    disp0.cmap = 'viridis'
    disp0.add_colorbar(ax=ax)
    disp0.add_ellipse(centroid, ringrad_camcoord.value * 2,
                      ringrad_camcoord.value * 2, 0., 0., color="red")
    disp0.add_ellipse(centroid, ringrad_inner.value * 2,
                      ringrad_inner.value * 2, 0., 0.,
                      color="magenta")
    disp0.add_ellipse(centroid, ringrad_outer.value * 2,
                      ringrad_outer.value * 2, 0., 0.,
                      color="magenta")
    ax.set_title(f"Event {event_id}")

    return ax
