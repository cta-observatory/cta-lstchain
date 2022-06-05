
import numpy as np
from ctapipe.image import number_of_islands
from ctapipe_io_lst.constants import N_PIXELS, PIXEL_INDEX

__all__ = [
    'get_bad_pixel_id_and_weight'
    'interpolate_bad_pixels'
]

def get_bad_pixel_id_and_weight(
        camera_geom, monitoring_table, apply_weight = True, power = 2
):
    """
    Get bad pixel id list and weight factors
    Parameters
    ----------
    camera_geom: 
          camera geometory
    monitoring_table:
          monitoring table
    apply_weight:
          apply weight based on distance to bad pixels if True
    power:
          power of weight factor if `apply_weight` = True 
    Returns
    -------
    bad_pixel_ids: `np.ndarray`
          pixel ids of bad pixels
    bad_pixel_neighbors_by_island: `np.ndarray`
          weight factors to compute average values
    """

    unusable = monitoring_table['unusable_pixels']
    # Locate pixels with HG declared unusable either in original calibration or                                                                                         
    # in interleaved events:                                                                                                                                            
    bad_pixels = unusable[0][0]  # original calibration                                                                                                                 
    for tf in unusable[1:][0]:   # calibrations with interleaveds                                                                                                       
        bad_pixels = np.logical_or(bad_pixels, tf)
        
    bad_pixel_ids = PIXEL_INDEX[bad_pixels]
    
    # Label islands containing bad pixels and neighbors
    bad_pixels_and_neighbors = bad_pixels | camera_geom.neighbor_matrix_sparse.dot(bad_pixels)
    _, bad_pixel_island_labels = number_of_islands(camera_geom, bad_pixels_and_neighbors)


    weight_factors = np.zeros([len(bad_pixel_ids), N_PIXELS])

    pix_x = camera_geom.pix_x.to_value("m")
    pix_y = camera_geom.pix_y.to_value("m")

    for i, bad_pixel_id in enumerate(bad_pixel_ids):

        bad_pixel_island_id = bad_pixel_island_labels[bad_pixel_id]
        good_pixels_on_island = (bad_pixel_island_labels == bad_pixel_island_id) & ~bad_pixels
        weight_factors[i][good_pixels_on_island] = 1

        if apply_weight == True:
            dist_to_bad_pixel = np.sqrt(
                (pix_x - pix_x[bad_pixel_id])**2 + (pix_y - pix_y[bad_pixel_id])**2
            )
            
            weight_factors[i] *= np.divide(1, dist_to_bad_pixel, 
                                           out = np.zeros(dist_to_bad_pixel.shape),
                                           where = (dist_to_bad_pixel != 0)
            )
            weight_factors[i] **= power

    return bad_pixel_ids, weight_factors


def interpolate_bad_pixels(
        image, peak_time, bad_pixel_ids, weight_factors
):
    """
    Interpolate bad pixels using the average values of nighboring pixels
    Parameters
    ----------
    image: `np.ndarray`
          Pixel charges
    peak_time: `np.ndarray`
          Pixel peak time
    bad_pixel_ids: `np.ndarray`
          pixel ids of bad pixels
    weight_factors: `np.ndarray`
          weight factors to compute average values
    """
    
    for i, bad_pixel_id in enumerate(bad_pixel_ids):
        image[bad_pixel_id] = np.average(image, weights = weight_factors[i])
        peak_time[bad_pixel_id] = np.average(image, weights = weight_factors[i])
