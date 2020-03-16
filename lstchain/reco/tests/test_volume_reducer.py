#!/usr/bin/env python
import numpy as np
from ctapipe.io import event_source
from ctapipe.utils import get_dataset_path
from ctapipe.calib import CameraCalibrator
from lstchain.io import get_standard_config

from lstchain.reco.volume_reducer import (get_volume_reduction_method,
                                          apply_volume_reduction,
                                          zero_suppression_tailcut_dilation)


def test_get_volume_reduction_method():
    config = get_standard_config()
    config['volume_reducer']['algorithm'] = 'zero_suppression_tailcut_dilation'
    algo = get_volume_reduction_method(config)
    algo = globals()[algo]
    assert algo is zero_suppression_tailcut_dilation


def test_apply_volume_reduction():
    source = event_source(get_dataset_path('gamma_test.simtel.gz'))
    ev = next(iter(source))
    cal = CameraCalibrator()
    config = get_standard_config()
    config['volume_reducer']['algorithm'] = 'zero_suppression_tailcut_dilation'

    cal(ev)
    algo = get_volume_reduction_method(config)
    apply_volume_reduction(ev, algo, config)

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
        if tel_id <= 4:
            break

    cal = CameraCalibrator()
    cal(event)
    camera = event.inst.subarray.tel[tel_id].camera
    imag = event.dl1.tel[tel_id].image

    pixels_to_keep = zero_suppression_tailcut_dilation(camera, imag)
    reduced_imag = np.copy(imag)
    reduced_imag[~pixels_to_keep] = 0

    assert reduced_imag[pixels_to_keep].all() is imag[pixels_to_keep].all()
    assert (reduced_imag[~pixels_to_keep] == 0).all()
