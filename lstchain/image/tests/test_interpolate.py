
import numpy as np
from astropy.table import Table
from ctapipe.instrument import CameraGeometry
from lstchain.image.interpolation import get_bad_pixel_and_neighbors_by_island, interpolate_bad_pixels

def test_interpolation():

    ngains = 2
    npixels = 1855
    image = np.arange(npixels)
    peak_time = np.arange(npixels)
    
    unusable_pixels = np.zeros([2, ngains, npixels], dtype=bool)
    unusable_pixels[0, 0][100] = True
    unusable_pixels[1, 0, 1000] = True
    unusable_pixels[1, 0, 1001] = True
    unusable_pixels[1, 0, 1854] = True

    image[[100, 1000, 1001, 1854]] = 0
    peak_time[[100, 1000, 1001, 1854]] = 20

    monitoring_table = Table([unusable_pixels], names=({'unusable_pixels'}))
    camera_geom = CameraGeometry.from_name('LSTCam-003')

    bad_pixel_by_island, bad_pixel_neighbors_by_island = get_bad_pixel_and_neighbors_by_island(
        camera_geom, monitoring_table
    )
    
    interpolate_bad_pixels(
        image, peak_time, bad_pixel_by_island, bad_pixel_neighbors_by_island
    )

    assert all(image == peak_time)
    assert all(image[[100, 1000, 1001, 1854]] == np.array([102, 1002, 1002, 1844]))
