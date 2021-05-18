from ctapipe.visualization import CameraDisplay

__all__ = ['plot_muon_event',
           ]

def plot_muon_event(ax, geom, image, centroid, ringrad_camcoord, 
                    ringrad_inner, ringrad_outer, event_id):
    """
    

    Paramenters
    ---------
    ax:               `matplotlib.pyplot.axis`
    geom:             CameraGeometry  
    centroid:         `float` centroid of the muon ring
    ringrad_camcoord: `float` ring radius in camera coordinates
    ringrad_inner:    `float` inner ring radius in camera coordinates
    ringrad_outer:    `float` outer ring radius in camera coordinates
    event_id:         `int` id of the analyzed event

    Returns
    ---------
    ax:               `matplotlib.pyplot.axis`
    """

    disp0 = CameraDisplay(geom, ax=ax)
    disp0.image = image
    disp0.cmap = 'viridis'
    disp0.add_colorbar(ax=ax)
    disp0.add_ellipse(centroid, ringrad_camcoord.value,
                  ringrad_camcoord.value, 0., 0., color="red")
    disp0.add_ellipse(centroid, ringrad_inner.value,
                                    ringrad_inner.value, 0., 0.,
                                    color="magenta")
    disp0.add_ellipse(centroid, ringrad_outer.value,
                                    ringrad_outer.value, 0., 0.,
                                    color="magenta")
    ax.set_title(f"Event {event_id}")

    return ax
