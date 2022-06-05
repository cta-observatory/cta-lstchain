
import numpy as np
from astropy.table import Table
from ctapipe.instrument import CameraGeometry
from lstchain.image.bad_pixel_interpolation import(
    get_bad_pixel_id_and_weight, 
    bad_pixel_interpolation
)
from ctapipe_io_lst.constants import N_GAINS, N_PIXELS
import copy

def test_interpolation():

    camera_geom = CameraGeometry.from_name('LSTCam-003')
    pix_x = camera_geom.pix_x.to_value("m")

    image = copy.copy(pix_x) + 10
    peak_time = copy.copy(pix_x) + 10
    
    unusable_pixels = np.zeros([2, N_GAINS, N_PIXELS], dtype=bool)
    
    unusable_pixels[0, 0, :7] = True
    unusable_pixels[1, 0, [1000, 1001, 1854]] = True

    image[unusable_pixels[0,0] | unusable_pixels[1,0]] = -10
    peak_time[unusable_pixels[0,0] | unusable_pixels[1,0]] = -10

    monitoring_table = Table([unusable_pixels], names=({'unusable_pixels'}))
    camera_geom = CameraGeometry.from_name('LSTCam-003')

    bad_pixel_ids, weight_factors = get_bad_pixel_id_and_weight(camera_geom, monitoring_table)

    bad_pixel_interpolation(
         image, peak_time, bad_pixel_ids, weight_factors
    )


    assert all(image == peak_time)
    
    np.testing.assert_allclose(
        image,
        pix_x + 10,
        rtol=1e-2,    
    )

