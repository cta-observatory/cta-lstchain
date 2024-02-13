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

    output_path = Path(
        default_value=None,
        allow_none=True,
        exists=False,
        directory_ok=False,
        help="Path to output file.",
    ).tag(config=True)

    def __init__(self, subarray, parent=None, config=None, **kwargs):
        super().__init__(subarray=subarray, parent=parent, config=config, **kwargs)

        self.subarray = subarray

    def __call__(self, event: ArrayEventContainer) -> None:
        pass
