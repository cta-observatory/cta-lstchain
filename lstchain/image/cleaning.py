"""
Image Cleaning Algorithms
"""

__all__ = [
    "time_constrained_clean",
]

import numpy as np
from scipy.sparse.csgraph import connected_components

def select_core(
    geom,
    image,
    picture_thresh=7,
    keep_isolated_pixels=False,
    min_number_picture_neighbors=0,
):

    """
    Clean an image by selecting only pixels that pass a certain p.e. threshold

    Parameters
    ----------
    geom: `ctapipe.instrument.CameraGeometry`
        Camera geometry information
    image: array
        pixel values
    picture_thresh: float or array
        threshold above which all pixels are retained
    keep_isolated_pixels: bool
        If True, pixels above the picture threshold will be included always,
        if not they are only included if a neighbor is in the picture or
        boundary
    min_number_picture_neighbors: int
        A picture pixel survives cleaning only if it has at least this number
        of picture neighbors. This has no effect in case keep_isolated_pixels is True

    Returns
    -------

    A boolean mask of *clean* pixels.  To get a zero-suppressed image and pixel
    list, use `image[mask], geom.pix_id[mask]`, or to keep the same
    image size and just set unclean pixels to 0 or similar, use
    `image[~mask] = 0`

    """
    pixels_above_picture = image >= picture_thresh

    if keep_isolated_pixels or min_number_picture_neighbors == 0:
        pixels_in_picture = pixels_above_picture
    else:
        # Require at least min_number_picture_neighbors. Otherwise, the pixel
        #  is not selected
        number_of_neighbors_above_picture = geom.neighbor_matrix_sparse.dot(
            pixels_above_picture.view(np.byte)
        )
        pixels_in_picture = pixels_above_picture & (
            number_of_neighbors_above_picture >= min_number_picture_neighbors
        )
    return pixels_in_picture


def select_timing_close_to_average(
    geom, mask_core, image, arrival_times, time_limit=4.5
):
    pixels_to_remove = []
    mask_core = mask_core.copy()
    time_ave = np.average(arrival_times[np.where(mask_core)[0]], weights=image[np.where(mask_core)[0]])
    for pixel in np.where(mask_core)[0]:
        time_diff = np.abs(arrival_times[pixel] - time_ave)
        if time_diff > time_limit:
            pixels_to_remove.append(pixel)
    mask_core[pixels_to_remove] = False
    return mask_core


def select_boundary(
    geom, mask_core, image, boundary_thresh=5
):
    pixels_above_boundary = image >= boundary_thresh
    pixels_with_picture_neighbors = geom.neighbor_matrix_sparse.dot(mask_core)
    return ( pixels_above_boundary & pixels_with_picture_neighbors ) & np.invert(mask_core)


def select_timing_close_to_core(
    geom, mask_core, mask_boundary, arrival_times, time_limit=1.5
):
    pixels_to_remove = []
    mask_boundary = mask_boundary.copy()
    for pixel in np.where(mask_boundary)[0]:
        neighbors_core = np.where(geom.neighbor_matrix[pixel] & mask_core)[0]  
        time_diff = np.abs(arrival_times[neighbors_core] - arrival_times[pixel])
        if sum(time_diff < time_limit) == 0:
            pixels_to_remove.append(pixel)
    mask_boundary[pixels_to_remove] = False
    return mask_boundary


def time_constrained_clean(
    geom, image, arrival_times, picture_thresh=7, boundary_thresh=5, time_limit_core=4.5, time_limit_boundary=1.5, keep_isolated_pixels=False, min_number_picture_neighbors=0
):

    # find core pixels that pass a picture threshold
    mask_core = select_core(geom, image, picture_thresh, keep_isolated_pixels, min_number_picture_neighbors)

    # keep core pixels whose arrival times are within a certain time limit of the average
    if sum(mask_core) > 0:
        mask_core = select_timing_close_to_average(geom, mask_core, image, arrival_times, time_limit_core)
    
    # find boundary pixels that pass a boundary threshold
    mask_boundary = select_boundary(geom, mask_core, image, boundary_thresh)

    # keep boundary pixels whose arrival times are within a certain time limit of the neighboring core pixels
    mask_boundary = select_timing_close_to_core(geom, mask_core, mask_boundary, arrival_times, time_limit_boundary)
        
    return mask_core | mask_boundary
