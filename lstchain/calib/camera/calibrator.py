import numpy as np
import os
from ctapipe.core.traits import Unicode, List, Int, Bool
from ctapipe.calib.camera import CameraCalibrator
from ctapipe.calib.camera.calibrator import integration_correction
from ctapipe.image.reducer import DataVolumeReducer
from ctapipe.image.extractor import ImageExtractor
from ctapipe.io.hdf5tableio import HDF5TableReader
from ctapipe.io.containers import MonitoringContainer
from ctapipe.calib.camera import gainselection
from lstchain.calib.camera.pulse_time_correction import PulseTimeCorrection


__all__ = ['LSTCameraCalibrator','get_charge_correction']


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

    calibration_path = Unicode(
        '',
        help='Path to LST calibration file'
    ).tag(config=True)

    time_calibration_path = Unicode(
        '',
        help='Path to drs4 time calibration file'
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

    apply_charge_correction = Bool(
        False,
        help='Apply charge pulse shape charge correction'

    ).tag(config=True)

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
            config=self.config
        )
        self.log.info(f"extractor {self.extractor_product}")

        print("EXTRACTOR", self.image_extractor)

        self.data_volume_reducer = DataVolumeReducer.from_name(
            self.reducer_product,
            config=self.config
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

        # calibration data container
        self.mon_data = MonitoringContainer()

        # initialize the MonitoringContainer() for the moment it reads it from a hdf5 file
        self._initialize_correction()

        # initialize the pulse shape  corrections
        if self.apply_charge_correction:

            # get the pulse shape  corrections
            pulse_correction = get_charge_correction(
                self.image_extractor.window_width,
                self.image_extractor.window_shift,
            )
        else:
            # no pulse shape correction by default
            pulse_correction = np.ones(2)

        self.log.info(f"Pulse shape charge correction {pulse_correction}")

        # global charge corrections : pulse shape * scale
        self.charge_correction = pulse_correction * self.charge_scale

        self.log.info(f"Total charge correction {self.charge_correction}")


    def _initialize_correction(self):
        """
        Read the correction from hdf5 calibration file
        """

        self.mon_data.tels_with_data = self.allowed_tels
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
        create dl0 level, for the moment copy the r1
        """
        waveforms = event.r1.tel[telid].waveform
        if self._check_r1_empty(waveforms):
            return

        event.dl0.event_id = event.r1.event_id

        # if not already done, initialize the event monitoring containers
        if event.mon.tel[telid].calibration.dc_to_pe is None:
            event.mon.tel[telid].calibration = self.mon_data.tel[telid].calibration
            event.mon.tel[telid].flatfield = self.mon_data.tel[telid].flatfield
            event.mon.tel[telid].pedestal = self.mon_data.tel[telid].pedestal
            event.mon.tel[telid].pixel_status = self.mon_data.tel[telid].pixel_status

        #
        # subtract the pedestal per sample and multiply for the calibration coefficients
        #
        event.dl0.tel[telid].waveform = (
                (waveforms - self.mon_data.tel[telid].calibration.pedestal_per_sample[:, :, np.newaxis])
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

        # correct charge for width integration
        corrected_charge = charge * self.charge_correction[:,np.newaxis]

        # correct time with drs4 correction if available
        if self.time_corrector:
            pulse_time = self.time_corrector.get_corr_pulse(event, pulse_time)

        # add flat-fielding time correction
        pulse_time_ff_corrected = pulse_time + self.mon_data.tel[telid].calibration.time_correction

        # perform the gain selection if the threshold is defined
        if self.gain_threshold:
            waveforms, gain_mask = self.gain_selector(event.r1.tel[telid].waveform)

            event.dl1.tel[telid].image = corrected_charge[gain_mask, np.arange(charge.shape[1])]
            event.dl1.tel[telid].pulse_time = pulse_time_ff_corrected[gain_mask, np.arange(pulse_time_ff_corrected.shape[1])]

            # remember which channel has been selected
            event.r1.tel[telid].selected_gain_channel = gain_mask

        # if threshold == None
        else:
            event.dl1.tel[telid].image = corrected_charge
            event.dl1.tel[telid].pulse_time = pulse_time_ff_corrected


def get_charge_correction(window_width, window_shift):
    """
    Obtain charge correction from the reference pulse shape,
    this function is will be not necessary in ctapipe 0.8

    Parameters
    ----------
    window_width: width of the integration window

    window_shift: shift of the integration window

    Returns
    -------
    pulse_correction: pulse correction for HG and LG, np.array(2)

    """
    # read the pulse shape file (to be changed for ctapipe version 0.8)
    try:
        # read pulse shape from oversampled file
        pulse_ref_file = (os.path.join(os.path.dirname(__file__),
                    "../../data/oversampled_pulse_LST_8dynode_pix6_20200204.dat")
                    )
        hg_pulse_shape = []
        lg_pulse_shape = []
        with open(pulse_ref_file, 'r') as file:
                pulse_time_slice, pulse_time_step = file.readline().split()
                for line in file:
                    if "#" not in line:
                        columns = line.split()
                        hg_pulse_shape.append(float(columns[0]))
                        lg_pulse_shape.append(float(columns[1]))

        pulse_shape = np.array((hg_pulse_shape, lg_pulse_shape))

        pulse_correction = integration_correction(pulse_shape.shape[0],
                                                  pulse_shape,
                                                  float(pulse_time_step),
                                                  float(pulse_time_slice),
                                                  window_width,
                                                  window_shift
                                                  )

    except:
        print(f"Problem in reading calibration file {self.calibration_path}")
        raise

    return np.array(pulse_correction)