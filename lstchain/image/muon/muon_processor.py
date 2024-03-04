from ctapipe.calib import CameraCalibrator
from ctapipe.image.muon import MuonProcessor
from ctapipe.core.traits import Bool, Path
from ctapipe.io import DataWriter


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

    def __init__(self, event_source, subarray, parent=None, config=None, **kwargs):
        super().__init__(subarray=subarray, parent=parent, config=config, **kwargs)

        self.subarray = subarray
        self.muon_writer = None
        if self.write_muon_events and self.output_path is not None:
            self.muon_writer = DataWriter(
                parent=self, event_source=event_source, output_path=self.output_path
            )

        self.calibrator = CameraCalibrator(parent=self, subarray=subarray)

    def _process_telescope_event(self, event, tel_id):
        # Mainly copy from ctapipe and adding a recalibration step with a different
        # extractor used for muon analysis (GlobalPeakWindowSum)
        pass
