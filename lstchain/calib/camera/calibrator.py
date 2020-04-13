import numpy as np
import os
from ctapipe.core.traits import Unicode, List, Int
from ctapipe.calib.camera import CameraCalibrator
from ctapipe.image.reducer import *
from ctapipe.image.extractor import *
from ctapipe.io.hdf5tableio import HDF5TableReader
from ctapipe.io.containers import MonitoringContainer
from ctapipe.calib.camera import gainselection
from lstchain.calib.camera.pulse_time_correction import PulseTimeCorrection

__all__ = ['LSTCameraCalibrator']


class LSTCameraCalibrator(CameraCalibrator):
    """
    Calibrator to handle the LST camera calibration chain, in order to fill
    the DL1 data level in the event container.
    """
    extractor_product = Unicode(
        'LocalPeakWindowSum',
        help = 'Name of the charge extractor to be used'
    ).tag(config = True)

    reducer_product = Unicode(
        'NullDataVolumeReducer',
        help = 'Name of the DataVolumeReducer to use'
    ).tag(config = True)

    calibration_path = Unicode(
        '',
        allow_none = True,
        help = 'Path to LST calibration file'
    ).tag(config = True)

    time_calibration_path = Unicode(
        '',
        allow_none = True,
        help = 'Path to drs4 time calibration file'
    ).tag(config = True)

    allowed_tels = List(
        [1],
        help = 'List of telescope to be calibrated'
    ).tag(config = True)

    gain_threshold = Int(
        4094,
        allow_none = True,
        help = 'Threshold for the gain selection in ADC'
    ).tag(config = True)

    def __init__(self, **kwargs):
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
        super().__init__(**kwargs)

        # load the waveform charge extractor
        self.image_extractor = ImageExtractor.from_name(
            self.extractor_product,
            config = self.config
        )
        self.log.info(f"extractor {self.extractor_product}")

        print("EXTRACTOR", self.image_extractor)

        self.data_volume_reducer = DataVolumeReducer.from_name(
            self.reducer_product,
            config = self.config
        )
        self.log.info(f" {self.reducer_product}")

        # declare gain selector if the threshold is defined
        if self.gain_threshold:
            self.gain_selector = gainselection.ThresholdGainSelector(
                threshold = self.gain_threshold)

        # declare time calibrator if correction file exist
        if os.path.exists(self.time_calibration_path):
            self.time_corrector = PulseTimeCorrection(
                calib_file_path = self.time_calibration_path)
        else:
            self.time_corrector = None
            self.log.info(f"File {self.time_calibration_path} not found. No drs4 time corrections")

        # calibration data container
        self.mon_data = MonitoringContainer()

        # initialize the MonitoringContainer() for the moment it reads it from a hdf5 file
        self._initialize_correction()

    def _initialize_correction(self):
        """
        Read the correction from hdf5 calibration file
        """

        self.mon_data.tels_with_data = self.allowed_tels
        self.log.info(f"read {self.calibration_path}")

        try:
            with HDF5TableReader(self.calibration_path) as h5_table:
                assert h5_table._h5file.isopen == True
                for telid in self.allowed_tels:
                    # read the calibration data for the moment only one event
                    table = '/tel_' + str(telid) + '/calibration'
                    next(h5_table.read(table, self.mon_data.tel[telid].calibration))

                    dc_to_pe=self.mon_data.tel[telid].calibration.dc_to_pe
                    # put to zero unusable pixels
                    dc_to_pe[self.mon_data.tel[telid].calibration.unusable_pixels] = 0
                    # eliminate inf values id any (should be done probably before)
                    dc_to_pe[np.isinf(dc_to_pe)] = 0

                    # read the pixel_status container
                    table = '/tel_' + str(telid) + '/pixel_status'
                    next(h5_table.read(table, self.mon_data.tel[telid].pixel_status))
        except:
            self.log.error(f"Problem in reading calibration file {self.calibration_path}")

    def _calibrate_dl0(self, event, telid):
        """
        create dl0 level, for the moment copy the r1
        """        
        waveforms = event.r1.tel[telid].waveform
        if self._check_r1_empty(waveforms):
            return
        
        event.dl0.event_id = event.r1.event_id
        event.mon.tel[telid].calibration = self.mon_data.tel[telid].calibration
        event.mon.tel[telid].pixel_status = self.mon_data.tel[telid].pixel_status


        # subtract the pedestal per sample (should we do it?) and multiply for the calibration coefficients
        #


        event.dl0.tel[telid].waveform = (
                (event.r1.tel[telid].waveform - self.mon_data.tel[telid].calibration.pedestal_per_sample[:, :, np.newaxis])
                * self.mon_data.tel[telid].calibration.dc_to_pe[:, :, np.newaxis])

    def _calibrate_dl1(self, event, telid):
        """
        create calibrated dl1 image and calibrate it
        """
        waveforms = event.dl0.tel[telid].waveform

        if self._check_dl0_empty(waveforms):
            return

        if self.image_extractor.requires_neighbors():
            camera = event.inst.subarray.tel[telid].camera
            self.image_extractor.neighbors = camera.neighbor_matrix_where
        charge, pulse_time = self.image_extractor(waveforms)

        # correct time with drs4 correction if available
        if self.time_corrector:
            pulse_corr_array = self.time_corrector.get_corr_pulse(event, pulse_time)

        # otherwise use the ff time correction (not drs4 corrected)
        else:
            pulse_corr_array = pulse_time + self.mon_data.tel[telid].calibration.time_correction

        # perform the gain selection if the threshold is defined

        if self.gain_threshold:
            waveforms, gain_mask = self.gain_selector(event.r1.tel[telid].waveform)

            event.dl1.tel[telid].image = charge[gain_mask, np.arange(charge.shape[1])]
            event.dl1.tel[telid].pulse_time = pulse_corr_array[gain_mask, np.arange(pulse_corr_array.shape[1])]

            # remember the mask in the lst pixel_status array (this info is missing for the moment in the
            # r1 container). I follow the prescription given in the document
            # "R1 & DL0 Telescope Event Interfaces and Prototype Evaluation" of K. Kosack

            # bit 2 = LG
            gain_mask *= 4

            # bit 3 = HG
            gain_mask[np.where(gain_mask == 0)] = 8

            # bit 1 = pixel broken pixel (coming from the EvB)
            gain_mask += event.lst.tel[telid].evt.pixel_status >> 1 & 1

            # update pixel status
            event.lst.tel[telid].evt.pixel_status = gain_mask

        # if threshold == None
        else:
            event.dl1.tel[telid].image = charge
            event.dl1.tel[telid].pulse_time = pulse_corr_array


