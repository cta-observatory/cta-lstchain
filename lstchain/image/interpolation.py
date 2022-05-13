
import numpy as np
from ctapipe.image import number_of_islands

__all__ = [
    'get_bad_pixel_and_neighbors_by_island',
    'interpolate_bad_pixels'
]


def get_bad_pixel_and_neighbors_by_island(camera_geom, monitoring_table):
    """
    Get bad pixel and neighboring pixels mask
    Parameters
    ----------
    camera_geom: 
          camera geometory
    monitoring_table: `int`   
          monitoring table
    Returns
    -------
    bad_pixel_by_island: `np.ndarray`
    
    bad_pixel_neighbors_by_island: `np.ndarray`
    
    """

    unusable = monitoring_table['unusable_pixels']
    # Locate pixels with HG declared unusable either in original calibration or                                                                                         
    # in interleaved events:                                                                                                                                            
    bad_pixels = unusable[0][0]  # original calibration                                                                                                                 
    for tf in unusable[1:][0]:   # calibrations with interleaveds                                                                                                       
        bad_pixels = np.logical_or(bad_pixels, tf)
        
    bad_pixels_neighbors = bad_pixels | camera_geom.neighbor_matrix_sparse.dot(bad_pixels)
    
    num_bad_pixel_islands, bad_pixel_island_labels = number_of_islands(camera_geom, bad_pixels_neighbors)

    bad_pixel_by_island = []
    bad_pixel_neighbors_by_island = []

    for island_index in range(1, num_bad_pixel_islands+1):
        
        bad_pixel_island = (bad_pixel_island_labels == island_index)
        bad_pixel_by_island.append(bad_pixel_island & bad_pixels)
        bad_pixel_neighbors_by_island.append(bad_pixel_island & ~bad_pixels)
        
    return bad_pixel_by_island, bad_pixel_neighbors_by_island


def interpolate_bad_pixels(
        image, peak_time, bad_pixel_by_island, bad_pixel_neighbors_by_island
):
    """
    Interpolate bad pixels using the average values of nighboring pixels
    Parameters
    ----------
    image: `np.ndarray`
          Pixel charges
    peak_time: `np.ndarray`
          Pixel peak time
    bad_pixel_by_island: `np.ndarray`
          mask of bad pixels
    bad_pixel_neighbors_by_island: `np.ndarray`
          mask of bad pixel neighbors
    """
    
    for bad_pixel, bad_pixel_neighbors in zip(
            bad_pixel_by_island, bad_pixel_neighbors_by_island
    ):
            
        image[bad_pixel] = np.average(image[bad_pixel_neighbors])
        peak_time[bad_pixel] = np.average(peak_time[bad_pixel_neighbors])

    
