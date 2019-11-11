from ctapipe.visualization import CameraDisplay

def plot_event(ax, geom, phe, centroid, ringrad_camcoord, ringrad_inner, ringrad_outer, event_id):
    disp0 = CameraDisplay(geom, ax=ax)
    disp0.image = phe[0]
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
