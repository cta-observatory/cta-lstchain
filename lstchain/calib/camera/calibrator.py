import numpy as np
from ctapipe.core.traits import Unicode, List
from ctapipe.calib.camera import CameraCalibrator
from ctapipe.image.reducer import *
from ctapipe.image.extractor import *
from ctapipe.io.hdf5tableio import HDF5TableReader
from ctapipe.io.containers import MonitoringContainer

__all__ = ['LSTCameraCalibrator']


class LSTCameraCalibrator(CameraCalibrator):
    """
    Calibrator to handle the LST camera calibration chain, in order to fill
    the DL1 data level in the event container.
    """
    extractor_product = Unicode(
        'NeighborPeakWindowSum',
        help='Name of the charge extractor to be used'
    ).tag(config=True)

    reducer_product = Unicode(
        'NullDataVolumeReducer',
        help='Name of the DataVolumeReducer to use'
    ).tag(config=True)

    calibration_path = Unicode(
        '',
        allow_none=True,
        help='Path to LST calibration file'
    ).tag(config=True)

    allowed_tels = List(
        [0],
        help='List of telescope to be calibrated'
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
            The ImageExtractor to use. If None, then NeighborPeakWindowSum
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
        self.data_volume_reducer = DataVolumeReducer.from_name(
            self.reducer_product,
            config=self.config
        )
        self.log.info(f" {self.reducer_product}")

        # calibration data container
        self.mon_data = MonitoringContainer()

        # initialize the MonitoringContainer() for the moment it reads it from a hdf5 file
        self._initialize_correction()

    def _initialize_correction(self):
        """
        Read the correction from hdf5 calibration file
        """

        self.mon_data.tels_with_data=self.allowed_tels
        self.log.info(f"read {self.calibration_path}")

        try:
            with HDF5TableReader(self.calibration_path) as h5_table:
                assert h5_table._h5file.isopen == True
                for telid in self.allowed_tels:
                    # read the calibration data for the moment only one event
                    table = '/tel_' + str(telid) + '/calibration'
                    next(h5_table.read(table, self.mon_data.tel[telid].calibration))
                    # eliminate inf values (should be done probably before)
                    dc_to_pe=self.mon_data.tel[telid].calibration.dc_to_pe

                    dc_to_pe[np.isinf(dc_to_pe)] = 0
                    self.log.info(f"read {self.mon_data.tel[telid].calibration.dc_to_pe}")
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

        # subtract the pedestal per sample (should we do it?) and multiply for the calibration coefficients
        #
        event.dl0.tel[telid].waveform = (
                (event.r1.tel[telid].waveform-self.mon_data.tel[telid].calibration.pedestal_per_sample[:,:,np.newaxis])
                *self.mon_data.tel[telid].calibration.dc_to_pe[:,:,np.newaxis])


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

        event.dl0.event_id = event.r1.event_id
        event.dl1.tel[telid].image = charge
        event.dl1.tel[telid].pulse_time = pulse_time + self.mon_data.tel[telid].calibration.time_correction



