from ctapipe.containers import ArrayEventContainer
import numpy as np
from lstchain.reco.r0_to_dl1 import rescale_dl1_charge
from copy import copy

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
