import h5py
import tables
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

from ctapipe.instrument import CameraGeometry
from ctapipe.visualization import CameraDisplay


def tailcuts_clean_with_pedestal_threshold(
    geom,
    image,
    interleaved_pedestal_thresh,
    picture_thresh=7,
    boundary_thresh=5,
    keep_isolated_pixels=False,
    min_number_picture_neighbors=0,
):

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
    "return bias (mean) and rms from pedestal events in pe"

    f = tables.open_file(dl1_file)
    ped = f.root['/dl1/event/telescope/monitoring/pedestal']
    print("n events = {}".format(np.array(ped.cols.n_events)))
    ped_charge_mean = np.array(ped.cols.charge_mean)
    ped_charge_rms = np.array(ped.cols.charge_std)
    calib = f.root['/dl1/event/telescope/monitoring/calibration']
    dc_to_pe = np.array(calib.cols.dc_to_pe)
    ped_charge_mean_pe = ped_charge_mean*dc_to_pe
    ped_charge_rms_pe = ped_charge_rms*dc_to_pe
    f.close()
    return ped_charge_mean_pe, ped_charge_rms_pe

def get_threshold(ped_mean_pe, ped_rms_pe, sigma_clean):
    threshold_clean_pe = ped_mean_pe + sigma_clean*ped_rms_pe
    return threshold_clean_pe
