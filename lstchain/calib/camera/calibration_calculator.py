"""
Component for the estimation of the calibration coefficients  events
"""


from abc import abstractmethod
import numpy as np
import os
from ctapipe.core import Component
from ctapipe.core import traits
from ctapipe.core.traits import Unicode, Float, List
from lstchain.calib.camera.flatfield import FlatFieldCalculator
from lstchain.calib.camera.pedestals import PedestalCalculator


__all__ = [
    'CalibrationCalculator',
    'LSTCalibrationCalculator'
]


class CalibrationCalculator(Component):
    """
    Parent class for the camera calibration calculators.
    Fills the MonitoringCameraContainer on the base of calibration events

    Parameters
    ----------

    flatfield_calculator: lstchain.calib.camera.flatfield
         The flatfield to use. If None, then FlatFieldCalculator
            will be used by default.

    pedestal_calculator: lstchain.calib.camera.pedestal
         The pedestal to use. If None, then
           PedestalCalculator will be used by default.

    kwargs

    """

    pedestal_product = traits.enum_trait(
        PedestalCalculator,
        default='PedestalIntegrator'
    )

    flatfield_product = traits.enum_trait(
        FlatFieldCalculator,
        default='FlasherFlatFieldCalculator'
    )

    classes = List([
                    FlatFieldCalculator,
                    PedestalCalculator
                    ]
                   + traits.classes_with_traits(FlatFieldCalculator)
                   + traits.classes_with_traits(PedestalCalculator)

                   )

    def __init__(
        self,
        **kwargs

    ):

        """
        Parent class for the camera calibration calculators.
        Fills the MonitoringCameraContainer on the base of calibration events

        Parameters
        ----------

        flatfield_calculator: lstchain.calib.camera.flatfield
             The flatfield to use. If None, then FlatFieldCalculator
                will be used by default.

        pedestal_calculator: lstchain.calib.camera.pedestal
             The pedestal to use. If None, then
               PedestalCalculator will be used by default.

        kwargs

        """

        super().__init__(**kwargs)

        self.flatfield = FlatFieldCalculator.from_name(
            self.flatfield_product,
            **kwargs
        )
        self.pedestal = PedestalCalculator.from_name(
            self.pedestal_product,
            **kwargs
        )

        msg = "tel_id not the same for all calibration components"
        assert self.pedestal.tel_id == self.flatfield.tel_id, msg

        self.tel_id = self.flatfield.tel_id

        self.log.debug(f"{self.pedestal}")
        self.log.debug(f"{self.flatfield}")


class LSTCalibrationCalculator(CalibrationCalculator):
    """
    Calibration calculator for LST camera
    Fills the MonitoringCameraContainer on the base of calibration events

    Parameters:
    ----------
    minimum_hg_charge_median :
              Temporary cut on HG charge till the calibox TIB do not work
             (default for filter 5.2)

    maximum_lg_charge_std
             Temporary cut on LG std against Lidar events till the calibox TIB do not work
            (default for filter 5.2)

    time_calibration_path:
            Path with the drs4 time calibration corrections
    """

    minimum_hg_charge_median = Float(
        5000,
        help='Temporary cut on HG charge till the calibox TIB do not work (default for filter 5.2)'
    ).tag(config=True)

    maximum_lg_charge_std = Float(
        300,
        help='Temporary cut on LG std against Lidar events till the calibox TIB do not work (default for filter 5.2) '
    ).tag(config=True)

    def __init__(self, **kwargs):
        """
         Calibration calculator for LST camera
         Fills the MonitoringCameraContainer on the base of calibration events

         Parameters:
         ----------
         minimum_hg_charge_median :
             Temporary cut on HG charge till the calibox TIB do not work
             (default for filter 5.2)

         maximum_lg_charge_std
             Temporary cut on LG std against Lidar events till the calibox TIB do not work
            (default for filter 5.2)

        """
        super().__init__(**kwargs)

    def calculate_calibration_coefficients(self, event):
        """
        Calculate calibration coefficients from flatfield and pedestal statistics
        associated to the present event

        Parameters
        ----------
        event: EventAndMonDataContainer

        """

        ped_data = event.mon.tel[self.tel_id].pedestal
        ff_data = event.mon.tel[self.tel_id].flatfield
        status_data = event.mon.tel[self.tel_id].pixel_status
        calib_data = event.mon.tel[self.tel_id].calibration

        # mask from pedestal and flat-field data
        monitoring_unusable_pixels = np.logical_or(status_data.pedestal_failing_pixels,
                                                   status_data.flatfield_failing_pixels)
        # calibration unusable pixels are an OR of all masks
        calib_data.unusable_pixels = np.logical_or(monitoring_unusable_pixels,
                                                   status_data.hardware_failing_pixels)
        # Extract calibration coefficients with F-factor method
        # Assume fix F2 factor, F2=1+Var(gain)/Mean(Gain)**2 must be known from elsewhere
        F2 = 1.2

        # calculate photon-electrons
        n_pe = F2 * (ff_data.charge_median - ped_data.charge_median) ** 2 / (
                ff_data.charge_std ** 2 - ped_data.charge_std ** 2)

        # fill WaveformCalibrationContainer (this is an example)
        calib_data.time = ff_data.sample_time
        calib_data.time_range = ff_data.sample_time_range
        calib_data.n_pe = n_pe

        # find signal median of good pixels
        masked_npe = np.ma.array(n_pe, mask=calib_data.unusable_pixels)
        npe_signal_median = np.ma.median(masked_npe, axis=1)

        # Flat field factor
        ff = np.ma.getdata(npe_signal_median[:, np.newaxis] / n_pe)

        calib_data.dc_to_pe = n_pe / (ff_data.charge_median - ped_data.charge_median) * ff

        # put the time around zero
        camera_time_median = np.median(ff_data.time_median, axis=1)
        calib_data.time_correction = -ff_data.relative_time_median - camera_time_median[:, np.newaxis]

        ped_extractor_name = self.config.get("PedestalCalculator").get("charge_product")
        ped_width = self.config.get(ped_extractor_name).get("window_width")
        calib_data.pedestal_per_sample = ped_data.charge_median / ped_width

        # put to zero unusable pixels
        calib_data.dc_to_pe[calib_data.unusable_pixels] = 0
        calib_data.pedestal_per_sample[calib_data.unusable_pixels] = 0

        # eliminate inf values id any (still necessary?)
        calib_data.dc_to_pe[np.isinf(calib_data.dc_to_pe)] = 0



    def process_interleaved(self, event):
        """
        Process interleaved calibration events (pedestals and FF)
        Parameters
        ----------
        """
        new_ped = False
        new_ff = False

        # if pedestal event
        if event.r1.tel[self.tel_id].trigger_type == 32:

            new_ped = self.pedestal.calculate_pedestals(event)


        # if flat-field event: no calibration  TIB for the moment,
        # use a cut on the charge for ff events and on std for rejecting Magic Lidar events
        elif event.r1.tel[self.tel_id].trigger_type == 4 or (
                np.median(np.sum(event.r1.tel[self.tel_id].waveform[0], axis=1))
                > self.minimum_hg_charge_median
                and np.std(np.sum(event.r1.tel[self.tel_id].waveform[1], axis=1))
                < self.maximum_lg_charge_std):

            new_ff = self.flatfield.calculate_relative_gain(event)


            # if new ff, calculate new calibration coefficients
            if new_ff:
                self.calculate_calibration_coefficients(event)


        return new_ped, new_ff

    def force_interleaved_results(self, event):
        """
        Force output

        """
        #store results
        self.pedestal.store_results(event)
        self.flatfield.store_results(event)

        # calculates calibration values
        self.calculate_calibration_coefficients(event)

