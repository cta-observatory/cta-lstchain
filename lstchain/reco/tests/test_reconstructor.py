import lstchain.reco.reconstructor as reco
from lstchain.io import standard_config
from copy import deepcopy
from traitlets.config import Config
from ctapipe.io import EventSource


def test_TimeWaveformFitter_print(mc_gamma_testfile):
    config = deepcopy(standard_config)
    source = EventSource(input_url=mc_gamma_testfile,
                         config=Config(config["source_config"]))
    subarray = source.subarray
    fitter = reco.TimeWaveformFitter(subarray=subarray)
    print(fitter)
