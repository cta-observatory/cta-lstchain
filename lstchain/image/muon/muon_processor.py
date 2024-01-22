from ctapipe.image.muon import MuonProcessor


class LSTMuonProcessor(MuonProcessor):
    def __init__(self, subarray, **kwargs):
        super().__init__(subarray, **kwargs)
