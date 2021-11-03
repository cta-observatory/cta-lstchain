import numpy as np
import os
from pkg_resources import resource_filename
from ctapipe.core.traits import Path, List, Int, Unicode
from ctapipe.calib.camera import CameraCalibrator
from ctapipe.image.reducer import DataVolumeReducer
from ctapipe.image.extractor import ImageExtractor
from ctapipe.io.hdf5tableio import HDF5TableReader
from ctapipe.containers import MonitoringContainer
from ctapipe.calib.camera import gainselection
from lstchain.calib.camera.pulse_time_correction import PulseTimeCorrection
from lstchain.calib.camera.time_sampling_correction import TimeSamplingCorrection

__all__ = ['LSTCameraCalibrator']


class LSTCameraCalibrator(CameraCalibrator):
    """
    Calibrator to handle the LST camera calibration chain, in order to fill
    the DL1 data level in the event container.
    """
    extractor_product = Unicode(
        'LocalPeakWindowSum',
        help='Name of the charge extractor to be used'
    ).tag(config=True)

    reducer_product = Unicode(
        'NullDataVolumeReducer',
        help='Name of the DataVolumeReducer to use'
    ).tag(config=True)

    calibration_path = Path(
        exists=True, directory_ok=False,
        help='Path to LST calibration file'
    ).tag(config=True)

    time_calibration_path = Path(
        exists=True, directory_ok=False,
        help='Path to drs4 time calibration file'
    ).tag(config=True)

    time_sampling_correction_path = Path(
        exists=True, directory_ok=False,
        help='Path to time sampling correction file',
        allow_none = True,
    ).tag(config=True)

    allowed_tels = List(
        [1],
        help='List of telescope to be calibrated'
    ).tag(config=True)

    gain_threshold = Int(
        4094,
        allow_none=True,
        help='Threshold for the gain selection in ADC'
    ).tag(config=True)

    charge_scale = List(
        [1,1],
        help='Multiplicative correction factor for charge estimation [HG,LG]'
    ).tag(config=True)



    def __init__(self, subarray, **kwargs):
        """
        Parameters
        ----------

        reducer_product : ctapipe.image.reducer.DataVolumeReducer
            The DataVolumeReducer to use. If None, then
            NullDataVolumeReducer will be used by default, and waveforms
            will not be reduced.
        extractor_product : ctapipe.image.extractor.ImageExtractor
            The ImageExtractor to use. If None, then LocalPeakWindowSum
            will be used by default.
        calibration_path :
            Path to LST calibration file to get the pedestal and flat-field corrections


        kwargs
        """
        super().__init__(subarray, **kwargs)

        # load the waveform charge extractor
        self.image_extractor = ImageExtractor.from_name(
            self.extractor_product,
            subarray = self.subarray,
            config = self.config
        )
        self.log.info(f"extractor {self.extractor_product}")

        print("EXTRACTOR", self.image_extractor)

        self.data_volume_reducer = DataVolumeReducer.from_name(
            self.reducer_product,
            subarray=self.subarray,
            config = self.config
        )
        self.log.info(f" {self.reducer_product}")


        # declare gain selector if the threshold is defined
        if self.gain_threshold:
            self.gain_selector = gainselection.ThresholdGainSelector(
                threshold=self.gain_threshold
            )

        # declare time calibrator if correction file exist

        if os.path.exists(self.time_calibration_path):

            self.time_corrector = PulseTimeCorrection(
                calib_file_path=self.time_calibration_path
            )
        else:
            raise IOError(f"Time calibration file {self.time_calibration_path} not found!")

        # declare the charge sampling corrector
        if self.time_sampling_correction_path is not None:
            # search the file in resources if not found
            if not os.path.exists(self.time_sampling_correction_path):
                self.time_sampling_correction_path = resource_filename('lstchain',
                                                                       f"resources/{self.time_sampling_correction_path}")

            if os.path.exists(self.time_sampling_correction_path):
                self.time_sampling_corrector = TimeSamplingCorrection(
                    time_sampling_correction_path=self.time_sampling_correction_path
                )
            else:
                raise IOError(f"Sampling correction file {self.time_sampling_correction_path} not found!")
        else:
            self.time_sampling_corrector = None

        # calibration data container
        self.mon_data = MonitoringContainer()

        # initialize the MonitoringContainer() for the moment it reads it from a hdf5 file
        self._initialize_correction()


        self.log.info(f"Global charge scale {self.charge_scale}")


    def _initialize_correction(self):
        """
        Read the correction from hdf5 calibration file
        """

        self.log.info(f"read {self.calibration_path}")

        try:
            with HDF5TableReader(self.calibration_path) as h5_table:
                for telid in self.allowed_tels:
                    # read the calibration data
                    table = '/tel_' + str(telid) + '/calibration'
                    next(h5_table.read(table, self.mon_data.tel[telid].calibration))

                    # read pedestal data
                    table = '/tel_' + str(telid) + '/pedestal'
                    next(h5_table.read(table, self.mon_data.tel[telid].pedestal))

                    # read flat-field data
                    table = '/tel_' + str(telid) + '/flatfield'
                    next(h5_table.read(table, self.mon_data.tel[telid].flatfield))

                    # read the pixel_status container
                    table = '/tel_' + str(telid) + '/pixel_status'
                    next(h5_table.read(table, self.mon_data.tel[telid].pixel_status))
        except Exception:
            self.log.exception(
                f"Problem in reading calibration file {self.calibration_path}"
            )
            raise


    def _calibrate_dl0(self, event, telid):
        """
        create dl0 level, with gain-selected and calibrated waveform
        """
        waveforms = event.r1.tel[telid].waveform

        if self._check_r1_empty(waveforms):
            return

        # if not already done, initialize the event monitoring containers
        if event.mon.tel[telid].calibration.dc_to_pe is None:
            event.mon.tel[telid].calibration = self.mon_data.tel[telid].calibration
            event.mon.tel[telid].flatfield = self.mon_data.tel[telid].flatfield
            event.mon.tel[telid].pedestal = self.mon_data.tel[telid].pedestal
            event.mon.tel[telid].pixel_status = self.mon_data.tel[telid].pixel_status

        #
        # subtract the pedestal per sample and multiply for the calibration coefficients
        #
        calibrated_waveform = (
                (waveforms - self.mon_data.tel[telid].calibration.pedestal_per_sample[:, :, np.newaxis])
                * self.mon_data.tel[telid].calibration.dc_to_pe[:, :, np.newaxis]).astype(np.float32)

        # If requested, perform gain selection (this will be done by the EvB in future)
        # find the gain selection mask
        if waveforms.ndim == 3:

            # if threshold defined, perform gain selection
            if self.gain_threshold:
                gain_mask = self.gain_selector(waveforms)

                # select the samples
                calibrated_waveform = calibrated_waveform[gain_mask, np.arange(waveforms.shape[1])]

            else:
            # keep both HG and LG
                gain_mask = np.zeros((waveforms.shape[0], waveforms.shape[1]), dtype=np.int64)
                gain_mask[1] = 1
        else:
            # gain selection already performed in EvB: (0=HG, 1=LG)
            gain_mask = event.lst.tel[telid].evt.pixel_status >> 2 & 1

        # remember the calibrated and gain selected waveform
        # (this should be the r1 waveform to be compliant with ctapipe (?))
        event.dl0.tel[telid].waveform = calibrated_waveform

        # remember which channel has been selected
        event.r1.tel[telid].selected_gain_channel = gain_mask
        event.dl0.tel[telid].selected_gain_channel = gain_mask


    def _calibrate_dl1(self, event, telid):
        """
        create calibrated dl1 image and calibrate it
        """

        n_pixels = self.subarray.tels[telid].camera.geometry.n_pixels

        # copy the waveform be cause I do not want to change it
        waveforms = np.copy(event.dl0.tel[telid].waveform)
        gain_mask = event.dl0.tel[telid].selected_gain_channel

        if self._check_dl0_empty(waveforms):
            return

        # In case of no gain selection the selected gain channels are  [0,0,..][1,1,..]
        no_gain_selection = np.zeros((waveforms.shape[0], waveforms.shape[1]), dtype=np.int64)
        no_gain_selection[1] = 1


        # correct the dl0 waveform for the sampling time corrections
        if self.time_sampling_corrector:
            waveforms*= self.time_sampling_corrector.get_corrections(event,telid)[gain_mask, np.arange(n_pixels)]

        # extract the charge
        charge, peak_time = self.image_extractor(waveforms, telid, gain_mask)

        # correct charge for global scale
        corrected_charge = charge * np.array(self.charge_scale, dtype=np.float32)[gain_mask]

        # correct time with drs4 correction if available
        if self.time_corrector:
            peak_time_drs4_corrected = (peak_time -
                                        self.time_corrector.get_pulse_time_corrections(event)
                                        [gain_mask, np.arange(n_pixels)])

        # add flat-fielding time correction
        peak_time_ff_corrected = (peak_time_drs4_corrected +
                                  self.mon_data.tel[telid].calibration.time_correction.value
                                  [gain_mask, np.arange(n_pixels)])

        # fill dl1 container
        event.dl1.tel[telid].image = corrected_charge
        event.dl1.tel[telid].peak_time = peak_time_ff_corrected.astype(np.float32)


