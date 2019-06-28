import numpy as np
from lstchain.calib.camera.calib import gain_selection
from astropy.utils import deprecated


@deprecated('28/06/2019', message='gain selection is now performed at <= R1 calibration level')
def test_gain_selection():
    """
    test gain selection
    """
    # Let's generate a fake waveform from a camera of 3 samples and 10 pixels
    n_samples = 3
    w1 = np.transpose([np.concatenate([np.ones(5), 3 * np.ones(5)]) for i in range(n_samples)])
    w2 = np.transpose([10 * np.ones(10) for i in range(n_samples)])
    waveform = np.array([w1, w2])
    image = waveform.mean(axis=2)

    threshold = 2
    combined_image, combined_peakpos = gain_selection(waveform, image, image, threshold)

    # with a threshold of 2, the 5 first pixels should be selected in the first channel and 5 others in the second \
    # channel

    np.testing.assert_array_equal(combined_image, np.array([1, 1, 1, 1, 1, 10, 10, 10, 10, 10]))