#!/usr/bin/env python
import numpy as np
from ctapipe.io import event_source
from ctapipe.utils import get_dataset_path
from ctapipe.calib import CameraCalibrator
from ctapipe.image.cleaning import tailcuts_clean
from lstchain.io import get_standard_config

from lstchain.reco.volume_reducer import (get_volume_reduction_method,
                                          check_and_apply_volume_reduction,
                                          zero_suppression_tailcut_dilation)


def test_get_volume_reduction_method():
    config = get_standard_config()
    config['volume_reducer']['algorithm'] = 'zero_suppression_tailcut_dilation'
    algo = get_volume_reduction_method(config)
    algo = globals()[algo]
    assert algo is zero_suppression_tailcut_dilation


def test_check_and_apply_volume_reduction():
    source = event_source(get_dataset_path('gamma_test.simtel.gz'))
    ev = next(iter(source))
    cal = CameraCalibrator()
    config = get_standard_config()
    config['volume_reducer']['algorithm'] = 'zero_suppression_tailcut_dilation'

    cal(ev)
    check_and_apply_volume_reduction(ev, config)

    for tel_id in ev.r0.tels_with_data:
        assert 0 in ev.dl1.tel[tel_id].image
        assert 0 in ev.dl1.tel[tel_id].pulse_time
        assert 0 in ev.dl0.tel[tel_id].waveform


def test_zero_suppression_tailcut_dilation():
    source = event_source(get_dataset_path('gamma_test.simtel.gz'))
    for i, event in enumerate(source):
        for tel_id in list(event.r0.tels_with_data):
            if tel_id <= 4:
                break
            else:
                continue
        break

    cal = CameraCalibrator()
    cal(event)
    camera = event.inst.subarray.tel[tel_id].camera
    imag = event.dl1.tel[tel_id].image

    pixels_zero_supp = zero_suppression_tailcut_dilation(camera, imag)
    pixels_tailcut = tailcuts_clean(camera, imag,
                                    picture_thresh=8,
                                    boundary_thresh=4,
                                    keep_isolated_pixels=True,
                                    min_number_picture_neighbors=0
                                    )

    reduced_imag = np.copy(imag)
    cleaned_imag = np.copy(imag)

    reduced_imag[~pixels_zero_supp] = 0
    cleaned_imag[~pixels_tailcut] = 0

    pixels_cleaned_after_reduction = tailcuts_clean(camera, reduced_imag,
                                                    picture_thresh=8,
                                                    boundary_thresh=4,
                                                    keep_isolated_pixels=True,
                                                    min_number_picture_neighbors=0
                                                    )

    reduced_imag[~pixels_cleaned_after_reduction] = 0

    # Check that a (reduced & cleaned) and a cleaned image give the same result
    assert (reduced_imag == cleaned_imag).all()
    # Check that the anti mask of a volume reduced imaged is just filled by zeros.
    assert (reduced_imag[~pixels_zero_supp] == 0).all()
