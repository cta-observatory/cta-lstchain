"""
Extract flat field coefficients from flasher data files.
"""
import numpy as np
from traitlets import Dict, List, Unicode, Int, Bool


from ctapipe.core import Provenance, traits
from ctapipe.io import HDF5TableWriter
from ctapipe.core import Tool
from ctapipe.io import EventSource
from ctapipe.containers import PixelStatusContainer
from lstchain.calib.camera.calibration_calculator import CalibrationCalculator
from lstchain.io import add_config_metadata, add_global_metadata, global_metadata, write_metadata
from ctapipe.containers import EventType

__all__ = [
    'CalibrationHDF5Writer'
]


class CalibrationHDF5Writer(Tool):
    """
     Tool that generates a HDF5 file with camera calibration coefficients.
     This is just an example on how the monitoring containers can be filled using
     the calibration Components in calib/camera.
     This example is based on an input file with pedestal and flat-field events

     For getting help run:
     python calc_camera_calibration.py --help

     """

    name = "CalibrationHDF5Writer"
    description = "Generate a HDF5 file with camera calibration coefficients"

    one_event = Bool(
        False,
        help='Stop after first calibration event'
    ).tag(config=True)

    output_file = Unicode(
        'calibration.hdf5',
        help='Name of the output file'
    ).tag(config=True)

    calibration_product = traits.create_class_enum_trait(
       CalibrationCalculator,
        default_value='LSTCalibrationCalculator'
    )

    events_to_skip = Int(
        1000,
        help='Number of first events to skip due to bad DRS4 pedestal correction'
    ).tag(config=True)

    aliases = Dict(dict(
        input_file='EventSource.input_url',
        max_events='EventSource.max_events',
        output_file='CalibrationHDF5Writer.output_file',
        calibration_product='CalibrationHDF5Writer.calibration_product',
        events_to_skip='CalibrationHDF5Writer.events_to_skip'
    ))

    classes = List([EventSource,
                    CalibrationCalculator
                    ]
                   + traits.classes_with_traits(CalibrationCalculator)
                   + traits.classes_with_traits(EventSource)
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
        self.processor = None
        self.writer = None
        self.tot_events = 0
        self.simulation = False

    def setup(self):

        self.log.debug(f"Open  file")
        self.eventsource = EventSource.from_config(parent=self)

        tel_id = self.eventsource.lst_service.telescope_id
        if self.eventsource.r0_r1_calibrator.drs4_pedestal_path.tel[tel_id] is None:
            raise IOError("Missing (mandatory) drs4 pedestal file in trailets")

        # if data remember how many event in the files
        if "LSTEventSource" in str(type(self.eventsource)):
            self.tot_events = len(self.eventsource.multi_file)
            self.log.debug(f"Input file has file {self.tot_events} events")
        else:
            self.tot_events = self.eventsource.max_events
            self.simulation = True

        self.processor = CalibrationCalculator.from_name(
            self.calibration_product,
            parent=self,
            subarray = self.eventsource.subarray
        )

        group_name = 'tel_' + str(tel_id)

        self.log.debug(f"Open output file {self.output_file}")

        self.writer = HDF5TableWriter(
            filename=self.output_file, group_name=group_name, overwrite=True
        )

    def start(self):
        '''Calibration coefficient calculator'''

        metadata = global_metadata(self.eventsource)
        write_metadata(metadata, self.output_file)

        tel_id = self.eventsource.lst_service.telescope_id
        new_ped = False
        new_ff = False
        end_of_file = False

        try:
            self.log.debug(f"Start loop")
            self.log.debug(f"If not simulation, skip first {self.events_to_skip} events")
            for count, event in enumerate(self.eventsource):

                if count % 1000 == 0 and count> self.events_to_skip:
                    self.log.debug(f"Event {count}")

                # if last event write results
                max_events_reached = (
                        self.eventsource.max_events is not None and count == self.eventsource.max_events - 1)
                if count == self.tot_events-1 or max_events_reached:
                    self.log.debug(f"Last event, count = {count}")
                    end_of_file = True

                # save the config, to be retrieved as data.meta['config']
                if count == 0:

                    if self.simulation:
                        initialize_pixel_status(event.mon.tel[tel_id],event.r1.tel[tel_id].waveform.shape)

                    ped_data = event.mon.tel[tel_id].pedestal
                    add_config_metadata(ped_data, self.config)
                    add_global_metadata(ped_data, metadata)

                    ff_data = event.mon.tel[tel_id].flatfield
                    add_config_metadata(ff_data, self.config)
                    add_global_metadata(ff_data, metadata)

                    status_data = event.mon.tel[tel_id].pixel_status
                    add_config_metadata(status_data, self.config)
                    add_global_metadata(status_data, metadata)

                    calib_data = event.mon.tel[tel_id].calibration
                    add_config_metadata(calib_data, self.config)
                    add_global_metadata(calib_data, metadata)


                # skip first events which are badly drs4 corrected
                if not self.simulation and count < self.events_to_skip:
                    continue

                # if pedestal event
                if event.trigger.event_type==EventType.SKY_PEDESTAL or (
                    self.simulation and
                    np.median(np.sum(event.r1.tel[tel_id].waveform[0], axis=1))
                    < self.processor.minimum_hg_charge_median):


                    new_ped = self.processor.pedestal.calculate_pedestals(event)


                # if flat-field event: no calibration  TIB for the moment,
                # use a cut on the charge for ff events and on std for rejecting Magic Lidar events
                elif event.trigger.event_type==EventType.FLATFIELD or (
                        self.simulation and np.median(np.sum(event.r1.tel[tel_id].waveform[0], axis=1))
                        > self.processor.minimum_hg_charge_median
                        and np.std(np.sum(event.r1.tel[tel_id].waveform[1], axis=1))
                        < self.processor.maximum_lg_charge_std):

                   new_ff = self.processor.flatfield.calculate_relative_gain(event)

                # write pedestal results when enough statistics or end of file
                if new_ped or end_of_file:

                    # update the monitoring container with the last statistics
                    if end_of_file:
                        self.processor.pedestal.store_results(event)

                    # write the event
                    self.log.debug(f"Write pedestal data at event n. {count+1}, id {event.index.event_id} "
                                   f"stat = {ped_data.n_events} events")

                    # write on file
                    self.writer.write('pedestal', ped_data)

                    new_ped = False

                # write flatfield results when enough statistics (also for pedestals) or end of file
                if (new_ff and ped_data.n_events > 0) or end_of_file:

                    # update the monitoring container with the last statistics
                    if end_of_file:
                        self.processor.flatfield.store_results(event)

                    self.log.debug(f"Write flatfield data at event n. {count+1}, id {event.index.event_id} "
                                   f"stat = {ff_data.n_events} events")

                    # write on file
                    self.writer.write('flatfield', ff_data)

                    new_ff = False

                    # Then, calculate calibration coefficients
                    self.processor.calculate_calibration_coefficients(event)

                    # write calib and pixel status
                    self.log.debug(f"Write pixel_status data")
                    self.writer.write('pixel_status',status_data)

                    self.log.debug(f"Write calibration data")
                    self.writer.write('calibration', calib_data)
                    if self.one_event:
                        break

                    #self.writer.write('mon', event.mon.tel[tel_id])
        except ValueError as e:
            self.log.error(e)

    def finish(self):
        Provenance().add_output_file(
            self.output_file,
            role='mon.tel.calibration'
        )
        self.writer.close()

def initialize_pixel_status(mon_camera_container,shape):
    """
    Initialize the pixel status container in the case of
    simulation events (this should be done in the event source, but
    added here for the moment)
    """
    # initialize the container
    status_container = PixelStatusContainer()
    status_container.hardware_failing_pixels = np.zeros((shape[0],shape[1]), dtype=bool)
    status_container.pedestal_failing_pixels = np.zeros((shape[0],shape[1]), dtype=bool)
    status_container.flatfield_failing_pixels = np.zeros((shape[0],shape[1]), dtype=bool)

    mon_camera_container.pixel_status = status_container


def main():
    exe = CalibrationHDF5Writer()

    exe.run()


if __name__ == '__main__':
    main()
