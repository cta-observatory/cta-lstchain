"""
Extract flat field coefficients from flasher data files.
"""
from traitlets import Dict, List, Unicode
import ctapipe.utils.tools as tool_utils
from ctapipe.core import Provenance
from ctapipe.io import HDF5TableWriter
from ctapipe.core import Tool
from ctapipe.io import EventSource

from ctapipe.image import WaveformCleaner, ChargeExtractor

from lstchain.calib.camera import FlatFieldCalculator
from ctapipe_io_lst.containers import FlatFieldContainer

__all__=['FlatFieldHDF5Writer']


class FlatFieldHDF5Writer(Tool):
    name = "FlatFieldHDF5Writer"
    description = "Generate a HDF5 file with flat field coefficients"

    output_file = Unicode(
        'flatfield.hdf5',
        help='Name of the output flat field file file'
    ).tag(config=True)

    cleaner_product = tool_utils.enum_trait(
        WaveformCleaner,
        default='NullWaveformCleaner'
    )
    calculator_product = tool_utils.enum_trait(
        FlatFieldCalculator,
        default='FlasherFlatFieldCalculator'
    )

    aliases = Dict(dict(
        input_file='EventSource.input_url',
        max_events='EventSource.max_events',
        flatfield_calculator='FlatFieldHDF5Writer.calculator_product',
        tel_id='FlatFieldCalculator.tel_id',
        sample_duration='FlatFieldCalculator.sample_duration',
        sample_size='FlatFieldCalculator.sample_size',
        n_channels='FlatFieldCalculator.n_channels',
        charge_product='FlatFieldCalculator.charge_product',
        cleaner='FlatFieldHDF5Writer.cleaner_product',
        baseline_start='BaselineWaveformCleaner.baseline_start',
        baseline_end='BaselineWaveformCleaner.baseline_end'
    ))

    classes = List([EventSource,
                    WaveformCleaner,
                    FlatFieldCalculator,
                    FlatFieldContainer,
                    HDF5TableWriter
                    ] + tool_utils.classes_with_traits(WaveformCleaner)
                   + tool_utils.classes_with_traits(ChargeExtractor)
                   + tool_utils.classes_with_traits(FlatFieldCalculator)
                   )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.eventsource = None
        self.flatfield = None
        self.container = None
        self.writer = None
        self.cleaner = None

    def setup(self):

        kwargs = dict(parent=self)

        self.eventsource = EventSource.from_config(**kwargs)
        self.flatfield = FlatFieldCalculator.from_name(
            self.calculator_product,
            **kwargs
        )
        self.cleaner = WaveformCleaner.from_name(
            self.cleaner_product,
            **kwargs
        )
        self.writer = HDF5TableWriter(

            filename=self.output_file, group_name='flatfield', overwrite=True
        )


    def start(self):
        '''Flat field coefficient calculator'''

        table_name = 'tel_' + str(self.flatfield.tel_id)
        self.log.info("write events in table: /flatfield/%s",
                      table_name)

        write_config = False
        for count, event in enumerate(self.eventsource):
            # one should add hier the pedestal subtraction and/or cleaner
            ff_data = self.flatfield.calculate_relative_gain(event)


            if ff_data:
                # save the config, to be retrieved as data.meta['config']
                if not write_config:
                    ff_data.meta['config'] = self.config
                    write_config = True

                self.writer.write(table_name, ff_data)

    def finish(self):
        Provenance().add_output_file(
            self.output_file,
            role='mon.tel.flatfield'
        )
        self.writer.close()


def main():
    exe = FlatFieldHDF5Writer()
    exe.run()


if __name__ == '__main__':
    main()
