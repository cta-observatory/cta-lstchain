"""
Extract flat field coefficients from flasher data files.
"""
import numpy as np
from traitlets import Dict, List, Unicode, Float, Bool


from ctapipe.core import Provenance, traits
from ctapipe.io import HDF5TableWriter
from ctapipe.core import Tool
from ctapipe.io import EventSource

from lstchain.calib.camera.flatfield import FlatFieldCalculator
from lstchain.calib.camera.pedestals import PedestalCalculator
from lstchain.calib.camera import CameraR0Calibrator

__all__ = [
    'CalibrationHDF5Writer'
]


class CalibrationHDF5Writer(Tool):
    """
     Tool that generates a HDF5 file with camera calibration coefficients.
     This is just an example on how the monitoring containers can be filled using
     the calibration Components in calib/camera.
     This example is based on an input file with interleaved pedestal and flat-field events

     For getting help run:
     python calc_camera_calibration.py --help

     """

    name = "CalibrationHDF5Writer"
    description = "Generate a HDF5 file with camera calibration coefficients"

    minimum_hg_charge_median = Float(
        5000,
        help='Temporary cut on HG charge till the calibox TIB do not work (default for filter 5.2)'
    ).tag(config=True)

    maximum_lg_charge_std = Float(
        300,
        help='Temporary cut on LG std against Lidar events till the calibox TIB do not work (default for filter 5.2) '
    ).tag(config=True)

    one_event = Bool(
        False,
        help='Stop after first calibration event'
    ).tag(config=True)

    output_file = Unicode(
        'calibration.hdf5',
        help='Name of the output file'
    ).tag(config=True)

    log_file = Unicode(
        'None',
        help='Name of the log file'
    ).tag(config=True)

    pedestal_product = traits.enum_trait(
        PedestalCalculator,
        default='PedestalIntegrator'
    )

    flatfield_product = traits.enum_trait(
        FlatFieldCalculator,
        default='FlasherFlatFieldCalculator'
    )

    r0calibrator_product =traits.enum_trait(
        CameraR0Calibrator,
        default='NullR0Calibrator'
    )

    aliases = Dict(dict(
        input_file='EventSource.input_url',
        output_file='CalibrationHDF5Writer.output_file',
        log_file='CalibrationHDF5Writer.log_file',
        max_events='EventSource.max_events',
        pedestal_file= 'LSTR0Corrections.pedestal_path',
        flatfield_product='CalibrationHDF5Writer.flatfield_product',
        pedestal_product='CalibrationHDF5Writer.pedestal_product',
        r0calibrator_product='CalibrationHDF5Writer.r0calibrator_product',
    ))

    classes = List([EventSource,
                    FlatFieldCalculator,
                    PedestalCalculator
                    ]
                   + traits.classes_with_traits(CameraR0Calibrator)
                   + traits.classes_with_traits(FlatFieldCalculator)
                   + traits.classes_with_traits(PedestalCalculator)

                   )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        """
         Tool that generates a HDF5 file with camera calibration coefficients.
         Input file must contain interleaved pedestal and flat-field events  

         For getting help run:
         python calc_camera_calibration.py --help
         
        """
        self.eventsource = None
        self.flatfield = None
        self.pedestal = None
        self.container = None
        self.writer = None
        self.r0calibrator = None
        self.tel_id = None
        self.tot_events = 0

    def setup(self):
        kwargs = dict(parent=self)

        self.eventsource = EventSource.from_config(**kwargs)

        # remember how many event in the files
        self.tot_events = len(self.eventsource.multi_file)
        self.log.debug(f"Input file has file {self.tot_events} events")

        self.flatfield = FlatFieldCalculator.from_name(
            self.flatfield_product,
            **kwargs
        )
        self.pedestal = PedestalCalculator.from_name(
            self.pedestal_product,
            **kwargs
        )

        if self.r0calibrator_product:
            self.r0calibrator = CameraR0Calibrator.from_name(
                self.r0calibrator_product,
                **kwargs
            )

        msg = "tel_id not the same for all calibration components"
        assert self.r0calibrator.tel_id == self.pedestal.tel_id == self.flatfield.tel_id, msg

        self.tel_id = self.flatfield.tel_id

        group_name = 'tel_' + str(self.tel_id)

        self.log.debug(f"Open output file {self.output_file}")

        self.writer = HDF5TableWriter(
            filename=self.output_file, group_name=group_name, overwrite=True
        )

    def start(self):
        '''Calibration coefficient calculator'''

        new_ped = False
        new_ff = False
        end_of_file = False
        try:

            self.log.debug(f"Start loop")
            for count, event in enumerate(self.eventsource):

                if count % 1000 == 0:
                    self.log.debug(f"Event {count}")

                # if last event write results
                if count == self.tot_events-1 or count == self.eventsource.max_events-1:
                    self.log.debug(f"Last event, count = {count}")
                    end_of_file = True

                # save the config, to be retrieved as data.meta['config']
                if count == 0:
                    ped_data = event.mon.tel[self.tel_id].pedestal
                    ped_data.meta['config'] = self.config

                    ff_data = event.mon.tel[self.tel_id].flatfield
                    ff_data.meta['config'] = self.config

                    status_data = event.mon.tel[self.tel_id].pixel_status
                    status_data.meta['config'] = self.config

                    calib_data = event.mon.tel[self.tel_id].calibration
                    calib_data.meta['config'] = self.config

                # correct for low level calibration
                self.r0calibrator.calibrate(event)

                # reject event without trigger type
                if event.r1.tel[self.tel_id].trigger_type == -1:
                    continue

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

                # write pedestal results when enough statistics or end of file
                if new_ped or end_of_file:

                    # update the monitoring container with the last statistics
                    if end_of_file:
                        self.pedestal.store_results(event)

                    # write the event
                    self.log.debug(f"Write pedestal data at event n. {count+1}, id {event.r0.event_id} "
                                   f"stat = {ped_data.n_events} events")

                    # write on file
                    self.writer.write('pedestal', ped_data)

                    new_ped = False

                # write flatfield results when enough statistics (also for pedestals) or end of file
                if (new_ff and ped_data.n_events > 0) or end_of_file:

                    # update the monitoring container with the last statistics
                    if end_of_file:
                        self.flatfield.store_results(event)

                    self.log.debug(f"Write flatfield data at event n. {count+1}, id {event.r0.event_id} "
                                   f"stat = {ff_data.n_events} events")

                    # write on file
                    self.writer.write('flatfield', ff_data)

                    new_ff = False

                    # Then, calculate calibration coefficients

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
                    calib_data.dc_to_pe = n_pe/(ff_data.charge_median-ped_data.charge_median)
                    # put the time around zero
                    camera_time_median = np.median(ff_data.time_median, axis=1)
                    calib_data.time_correction = -ff_data.relative_time_median - camera_time_median[:, np.newaxis]

                    ped_extractor_name = self.config.get("PedestalCalculator").get("charge_product")
                    ped_width=self.config.get(ped_extractor_name).get("window_width")
                    calib_data.pedestal_per_sample = ped_data.charge_median/ped_width

                    # write calib and pixel status

                    self.log.debug(f"Write pixel_status data")
                    self.writer.write('pixel_status',status_data)

                    self.log.debug(f"Write calibration data")
                    self.writer.write('calibration', calib_data)
                    if self.one_event:
                        break

                    #self.writer.write('mon', event.mon.tel[self.tel_id])
        except ValueError as e:
            self.log.error(e)

    def finish(self):
        Provenance().add_output_file(
            self.output_file,
            role='mon.tel.calibration'
        )
        self.writer.close()


def main():
    exe = CalibrationHDF5Writer()

    exe.run()


if __name__ == '__main__':
    main()
