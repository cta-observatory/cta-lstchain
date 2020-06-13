import tables
import numpy as np

def tailcuts_clean_with_pedestal_threshold(
    geom,
    image,
    interleaved_pedestal_thresh,
    picture_thresh=7,
    boundary_thresh=5,
    keep_isolated_pixels=False,
    min_number_picture_neighbors=0,
):

    """ Modified tailcuts_clean method from ctapipe.
    Clean an image by selection pixels that pass a two-threshold
    tail-cuts procedure.  The picture and boundary thresholds are
    defined with respect to the pedestal dispersion. The picture threshold have
    additional condition: picture threshold > threshold from interleaved
    pedestal events. All pixels that have a signal higher than the picture
    threshold will be retained, along with all those above the boundary
    threshold that are neighbors of a picture pixel.

    To include extra neighbor rows of pixels beyond what are accepted, use the
    `ctapipe.image.dilate` function.
    Parameters
    ----------
    geom: `ctapipe.instrument.CameraGeometry`
        Camera geometry information
    image: array
        pixel values
    interleaved_pedestal_thresh: array
        additional picture threshold calculated using interleaved events
    picture_thresh: float or array
        threshold above which all pixels are retained
    boundary_thresh: float or array
        threshold above which pixels are retained if they have a neighbor
        already above the picture_thresh
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

    pixels_above_picture = np.logical_and(image>= picture_thresh,
                                          image >= interleaved_pedestal_thresh)

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

    # by broadcasting together pixels_in_picture (1d) with the neighbor
    # matrix (2d), we find all pixels that are above the boundary threshold
    # AND have any neighbor that is in the picture
    pixels_above_boundary = image >= boundary_thresh
    pixels_with_picture_neighbors = geom.neighbor_matrix_sparse.dot(pixels_in_picture)
    if keep_isolated_pixels:
        return (
            pixels_above_boundary & pixels_with_picture_neighbors
        ) | pixels_in_picture
    else:
        pixels_with_boundary_neighbors = geom.neighbor_matrix_sparse.dot(
            pixels_above_boundary
        )
        return (pixels_above_boundary & pixels_with_picture_neighbors) | (
            pixels_in_picture & pixels_with_boundary_neighbors
)



def get_bias_and_rms(dl1_file):
    """
    Function to extract bias and rms from interleaved events from dl1 file.
    Parameters
    ----------
    input_filename: str
        path to dl1 file
    Returns
    -------
    bias, rms: np.ndarray, np.ndarray
        bias and rms in p.e.
    """
    f = tables.open_file(dl1_file)
    ped = f.root['/dl1/event/telescope/monitoring/pedestal']
    ped_charge_mean = np.array(ped.cols.charge_mean)
    ped_charge_rms = np.array(ped.cols.charge_std)
    calib = f.root['/dl1/event/telescope/monitoring/calibration']
    dc_to_pe = np.array(calib.cols.dc_to_pe)
    ped_charge_mean_pe = ped_charge_mean*dc_to_pe
    ped_charge_rms_pe = ped_charge_rms*dc_to_pe
    f.close()
    return ped_charge_mean_pe, ped_charge_rms_pe

def get_threshold(ped_mean_pe, ped_rms_pe, sigma_clean):
    """
    Function to calculate picture threshold from interleaved pedestal events.
    Parameters
    ----------
    ped_mean_pe: np.ndarray
        pedestal charge mean from interleaved pedestal events
    ped_rms_pe: np.ndarray
        pedestal charge rms from interleaved pedestal events
    sigma_clean: float
        cleaning level
    Returns
    -------
    picture_thresh: np.ndarray
        picture threshold calculated using interleaved pedestal events
    """
    threshold_clean_pe = ped_mean_pe + sigma_clean*ped_rms_pe
    return threshold_clean_pe
