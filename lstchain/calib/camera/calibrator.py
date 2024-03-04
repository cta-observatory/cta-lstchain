from ctapipe.calib import CameraCalibrator
from ctapipe.containers import ArrayEventContainer
from ctapipe.core.traits import Path, Bool
from ctapipe.io import read_table

from . import get_calibration_id


class LSTCameraCalibrator(CameraCalibrator):
    """
    LST specific CameraCalibrator which handles the Cat-B calibrations.
    """

    apply_cat_B_calibrations = Bool(
        help="Apply cat-B calibrations",
        default_value=False,
    ).tag(config=True)

    cat_b_calibrations_path = Path(
        default_value=None,
        allow_none=True,
        exists=True,
        directory_ok=False,
        help="Path to cat-B calibrations file",
    ).tag(config=True)

    def __init__(self, subarray, config=None, parent=None, **kwargs):
        super().__init__(subarray=subarray, config=config, parent=parent, **kwargs)

        self.cat_b_calibrations = read_table(self.cat_b_calibrations_path, "")

    def _fill_dl1_camera_calibration_container(self, event, tel_id):
        waveforms = event.dl0.tel[tel_id].waveform
        if self._check_dl0_empty(waveforms):
            return

        elif not self.apply_cat_B_calibrations or self.cat_b_calibrations_path is None:
            # self.log.warning
            return

        calibration_id = get_calibration_id(self.cat_b_calibrations, event, tel_id)
        selected_gain_channel = event.dl0.tel[tel_id].selected_gain_channel

        event.calibration.tel[tel_id].dl1.time_shift -= self.cat_b_calibrations[
            "time_correction"
        ][calibration_id][selected_gain_channel]
        event.calibration.tel[tel_id].dl1.pedestal_offset = self.cat_b_calibrations[
            "pedestal_per_sample"
        ][calibration_id][selected_gain_channel]
        event.calibration.tel[tel_id].dl1.relative_factor = self.cat_b_calibrations[
            "dc_to_pe"
        ][calibration_id][selected_gain_channel]

        # unusable pixel
        pass

    def _cat_b_calibrate_dl1(self, event, tel_id):
        """
        If waveforms are missing in the input files and applying cat-B calibrations (dl1ab).
        """
        waveforms = event.dl0.tel[tel_id].waveform
        if not self._check_dl0_empty(waveforms):
            return

        elif not self.apply_cat_B_calibrations or self.cat_b_calibrations_path is None:
            # self.log.warning
            return

        calibration_id = get_calibration_id(self.cat_b_calibrations, event, tel_id)
        selected_gain_channel = event.dl0.tel[tel_id].selected_gain_channel
        # apply_cat_B_calibrations on image and peak_time
        #
        pass

    def __call__(self, event: ArrayEventContainer):
        tel = event.r1.tel or event.dl0.tel or event.dl1.tel
        for tel_id in tel.keys():
            self._calibrate_dl0(event, tel_id)
            self._fill_dl1_camera_calibration_container(event, tel_id)
            self._calibrate_dl1(event, tel_id)
            self._cat_b_calibrate_dl1(event, tel_id)
