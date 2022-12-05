import os
from ctapipe.containers import ArrayEventContainer
import numpy as np
from lstchain.reco.r0_to_dl1 import r0_to_dl1, rescale_dl1_charge
from lstchain.io import standard_config
from copy import copy, deepcopy


def test_rescale_dl1_charge():
    event = ArrayEventContainer()
    tel_ids = [1, 3]
    images = {}
    for tel_id in tel_ids:
        images[tel_id] = np.random.rand(1855)
        event.dl1.tel[tel_id].image = copy(images[tel_id])

    rescaling_factor = np.random.rand() * 10
    rescale_dl1_charge(event, rescaling_factor)

    for tel_id in tel_ids:
        np.testing.assert_allclose(event.dl1.tel[tel_id].image, images[tel_id]*rescaling_factor)


def test_r0_to_dl1_nsb_tuning(tmp_path, mc_gamma_testfile):
    config = deepcopy(standard_config)
    config['source_config']['EventSource']['allowed_tels'] = [1]
    config['waveform_nsb_tuning']['nsb_tuning'] = True
    config['waveform_nsb_tuning']['spe_location'] = os.path.join(os.path.dirname(__file__),
                                                                 '../../data/SinglePhE_ResponseInPhE_expo2Gaus.dat')
    r0_to_dl1(mc_gamma_testfile, custom_config=config, output_filename=tmp_path / "tmp.h5")
