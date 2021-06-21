'''
COPIED FROM CTAPIPE master before release of 0.11 to avoid the
bad performance of the version in ctapipe 0.10.5.

TODO: Remove when switching to ctapipe > 0.10.5
'''
import numpy as np


def apply_time_delta_cleaning(
    geom, mask, arrival_times, min_number_neighbors, time_limit
):
    """
    Identify all pixels from selection that have less than N
    neighbors that arrived within a given timeframe.

    Parameters
    ----------
    geom: `ctapipe.instrument.CameraGeometry`
        Camera geometry information
    mask: array, boolean
        boolean mask of *clean* pixels before time_delta_cleaning
    arrival_times: array
        pixel timing information
    min_number_neighbors: int
        Threshold to determine if a pixel survives cleaning steps.
        These steps include checks of neighbor arrival time and value
    time_limit: int or float
        arrival time limit for neighboring pixels

    Returns
    -------

    A boolean mask of *clean* pixels.
    """
    pixels_to_keep = mask.copy()
    time_diffs = np.abs(arrival_times[mask, None] - arrival_times)
    # neighboring pixels arriving in the time limit and previously selected
    valid_neighbors = (time_diffs < time_limit) & geom.neighbor_matrix[mask] & mask
    enough_neighbors = np.count_nonzero(valid_neighbors, axis=1) >= min_number_neighbors
    pixels_to_keep[pixels_to_keep] &= enough_neighbors
    return pixels_to_keep
