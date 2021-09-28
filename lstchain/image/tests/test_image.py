from lstchain.image.cleaning import apply_dynamic_cleaning
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
