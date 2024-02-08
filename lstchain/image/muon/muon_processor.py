from ctapipe.containers import ArrayEventContainer
from ctapipe.image.muon import MuonProcessor
from ctapipe.core.traits import Bool


class LSTMuonProcessor(MuonProcessor):
    """
    LST specific muon processor
    """

    write_muon_events = Bool(
        help="Write promising muon events to an extra .h5 file",
        default_value=False,
    ).tag(config=True)

    def __init__(self, subarray, **kwargs):
        super().__init__(subarray, **kwargs)

    def __call__(self, event: ArrayEventContainer) -> None:
        pass
