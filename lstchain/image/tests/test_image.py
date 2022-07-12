from lstchain.image.cleaning import (
    apply_dynamic_cleaning,
    get_only_main_island,
    lst_image_cleaning,
)
from ctapipe.image import number_of_islands, dilate
from ctapipe.instrument import CameraGeometry
from traitlets.config import Config
import numpy as np

def test_dynamic_cleaning():

    npixels = 1855
    image = np.linspace(0, npixels-1, npixels)
    signal_pixels = np.array(npixels*[True])
    fraction = 0.03
    mean3 = np.mean(image[-3:])
    mask = apply_dynamic_cleaning(image, signal_pixels, 100, fraction)
    assert(mask.sum() == np.sum(image>fraction*mean3))
    mask = apply_dynamic_cleaning(image, signal_pixels,
                                  np.max(image),
                                  fraction)
    assert(mask.sum() == signal_pixels.sum())

def test_get_only_main_island():

    # Creating a mask of pixels with two islands:
    # Bigger island around pixel#39,40,41 and smaller one around pixel#42
    geom = CameraGeometry.from_name("LSTCam")
    mask = np.array(1855*[False])

    num_pix = [39, 40, 41, 42]
    for pix in num_pix:
        some_neighs = geom.neighbors[pix][0:6]
        mask[pix] = True
        mask[some_neighs] = True

    num_islands, island_labels = number_of_islands(geom, mask)
    # Number of islands should be 2 here
    assert(num_islands == 2)

    signal_pixels = get_only_main_island(island_labels, mask)
    num_island, _ = number_of_islands(geom, signal_pixels)
    
    assert(num_island == 1)
    assert(signal_pixels.sum() == 13)

def test_lst_image_cleaning():

    geom = CameraGeometry.from_name("LSTCam")
    n_pixels = 1855
    mask = np.array(1855*[False])
    image = np.zeros((1855,))
    arrival_times = np.zeros((1855,))

    num_pix = [37, 38, 39, 40, 41, 42, 43, 44]
    for pix in num_pix:
        mask[pix] = True
        image[pix] = 50

    mask_dilate_1 = dilate(geom, mask)
    mask_dilate_2 = dilate(geom, mask_dilate_1)
    image[mask_dilate_1 ^ mask] = 10
    arrival_times[mask_dilate_1] = 15
    arrival_times[mask_dilate_2 ^ mask_dilate_1] = 10

    signal_pixels, num_islands, n_pixels = lst_image_cleaning(
        geom=geom,
        image=image,
        arrival_times=arrival_times,
        signal_pixels=mask_dilate_2,
        delta_time=2,
        use_dynamic_cleaning=True,
        fraction_dynamic=0.3,
        threshold_dynamic=40,
        use_only_main_island=True
    )
    
    assert(num_islands == 2)
    assert(signal_pixels.sum() == 5)
    assert(n_pixels == 63)
