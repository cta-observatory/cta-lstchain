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

    aliases = Dict(dict(
        pedestal_calculator='PedestalHDF5Writer.calculator_product',
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
                    HDF5TableWriter
                    ] + tool_utils.classes_with_traits(PedestalCalculator))

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.eventsource = None
        self.pedestal = None
        self.container = None
        self.writer = None

    def setup(self):
        kwargs = dict(parent=self)
        self.eventsource = EventSource.from_config(**kwargs)
        self.pedestal = PedestalCalculator.from_name(
            self.calculator_product,
            **kwargs
        )

        self.writer = HDF5TableWriter(
            filename=self.output_file, group_name='pedestals', overwrite=True
        )

    def start(self):
        '''Pedestal calculator'''

        for count, event in enumerate(self.eventsource):

            if __name__ == '__main__':
                ped_data = self.pedestal.calculate_pedestals(event)


            if ped_data:
                # save the config, to be retrieved as data.meta['config']
                
                write_config = False
                if not write_config:
                    ped_data.meta['config']=self.config
                    write_config = True

                table_name = 'tel_' + str(self.pedestal.tel_id)

                self.log.info("write event in table: /pedestal/%s",
                              table_name)
                self.writer.write(table_name, ped_data)

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
