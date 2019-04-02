"""
Extract pedestals from pedestal events
"""
from traitlets import Dict, List, Unicode
import ctapipe.utils.tools as tool_utils
from ctapipe.core import Provenance
from ctapipe.io import HDF5TableWriter
from ctapipe.core import Tool
from ctapipe.io import EventSource


from lstchain.calib.camera.pedestals import PedestalCalculator
from ctapipe_io_lst.containers import PedestalContainer
from lstchain.calib.camera.r0 import CameraR0Calibrator


class PedestalHDF5Writer(Tool):
    name = "PedestalHDF5Writer"
    description = "Generate a HDF5 file with pedestal values"

    output_file = Unicode(
        'pedestal.hdf5',
        help='Name of the output file'
    ).tag(config=True)

    calculator_product = tool_utils.enum_trait(
        PedestalCalculator,
        default='PedestalIntegrator'
    )
    r0calibrator_product = tool_utils.enum_trait(
        CameraR0Calibrator,
        default='NullR0Calibrator'
    )

    aliases = Dict(dict(
        input_file='EventSource.input_url',
        max_events='EventSource.max_events',
        tel_id='PedestalCalculator.tel_id',
        sample_duration='PedestalCalculator.sample_duration',
        sample_size='PedestalCalculator.sample_size',
        n_channels='PedestalCalculator.n_channels',
        charge_product = 'PedestalCalculator.charge_product'
    ))

    classes = List([EventSource,
                    PedestalCalculator,
                    PedestalContainer,
                    CameraR0Calibrator,
                    HDF5TableWriter
                    ] + tool_utils.classes_with_traits(PedestalCalculator)
                      + tool_utils.classes_with_traits(CameraR0Calibrator))

    def __init__(self, **kwargs):

        super().__init__(**kwargs)
        self.eventsource = None
        self.pedestal = None
        self.container = None
        self.writer = None
        self.group_name = None
        self.r0calibrator = None

    def setup(self):
        kwargs = dict(parent=self)
        self.eventsource = EventSource.from_config(**kwargs)
        self.pedestal = PedestalCalculator.from_name(
            self.calculator_product,
            **kwargs
        )
        self.r0calibrator = CameraR0Calibrator.from_name(
            self.r0calibrator_product,
            **kwargs
        )
        self.group_name = 'tel_' + str(self.pedestal.tel_id)

        self.writer = HDF5TableWriter(
            filename=self.output_file, group_name=self.group_name, overwrite=True
        )


    def start(self):
        '''Pedestal calculator'''

        write_config = True

        # loop on events
        for count, event in enumerate(self.eventsource):

            # perform R0->R1
            self.r0calibrator.calibrate(event)

            # fill pedestal monitoring container
            if self.pedestal.calculate_pedestals(event):

                ped_data = event.mon.tel[self.pedestal.tel_id].pedestal
                self.log.debug(f" r0 {event.r0.tel[0].waveform.shape}")
                self.log.debug(f" r1 {event.r1.tel[0].waveform.shape}")
                if write_config:
                    ped_data.meta['config']=self.config
                    write_config = False

                self.log.debug(f"write event in table: {self.group_name}/pedestal")

                # write data to file
                self.writer.write('pedestal', ped_data)

    def finish(self):
        Provenance().add_output_file(
            self.output_file,
            role='mon.tel.pedestal'
        )
        self.writer.close()


def main():
    exe = PedestalHDF5Writer()
    exe.run()


if __name__ == '__main__':
    main()
