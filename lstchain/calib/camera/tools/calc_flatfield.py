"""
Extract flat field coefficients from flasher data files.
"""
from traitlets import Dict, List, Unicode

from ctapipe.core import Provenance
from ctapipe.io import HDF5TableWriter
from ctapipe.core import Tool
from ctapipe.io import EventSourceFactory

from ctapipe.image import ChargeExtractorFactory, WaveformCleanerFactory

# to be changed for ctapipe
from lstchain.calib.camera import FlatFieldFactory
from lstchain.io.containers import FlatFieldContainer


class PedestalsHDF5Writer(Tool):
    name = "FlatFieldHDF5Writer"
    description = "Generate a HDF5 file with flat field coefficients"

    output_file = Unicode(
        'flatfield.hdf5',
        help='Name of the output flat field file file'
    ).tag(config=True)

    aliases = Dict(dict(
        input_file='EventSourceFactory.input_url',
        max_events='EventSourceFactory.max_events',
        allowed_tels='EventSourceFactory.allowed_tels',
        charge_extractor='ChargeExtractorFactory.product',
        window_width='ChargeExtractorFactory.window_width',
        t0='ChargeExtractorFactory.t0',
        window_shift='ChargeExtractorFactory.window_shift',
        sig_amp_cut_HG='ChargeExtractorFactory.sig_amp_cut_HG',
        sig_amp_cut_LG='ChargeExtractorFactory.sig_amp_cut_LG',
        lwt='ChargeExtractorFactory.lwt',
        cleaner='WaveformCleanerFactory.product',
        baseline_start='WaveformCleanerFactory.baseline_start',
        baseline_end='WaveformCleanerFactory.baseline_end',
        generator='FlatFieldFactory.product',
        tel_id='FlatFieldFactory.tel_id',
        sample_duration='FlatFieldFactory.sample_duration',
        sample_size='FlatFieldFactory.sample_size',
        n_channels='FlatFieldFactory.n_channels',
    ))

    classes = List([EventSourceFactory,
                    ChargeExtractorFactory,
                    WaveformCleanerFactory,
                    FlatFieldFactory,
                    FlatFieldContainer,
                    HDF5TableWriter
                    ])

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.eventsource = None
        self.flatfield = None
        self.container = None
        self.writer = None

    def setup(self):
        kwargs = dict(config=self.config, tool=self)
        self.eventsource = EventSourceFactory.produce(**kwargs)
        self.flatfield = FlatFieldFactory.produce(**kwargs)

        #self.container = FlatFieldContainer()
        self.writer = HDF5TableWriter(
            filename=self.output_file, group_name='flatfield', overwrite=True
        )

    def start(self):
        '''Flat field coefficient calculator'''

        for count, event in enumerate(self.eventsource):

            ff_data = self.flatfield.calculate_relative_gain(event)

            if ff_data:

                table_name = 'tel_' + str(self.flatfield.tel_id)

                self.log.info("write event in table: /flatfield/%s",
                              table_name)
                self.writer.write(table_name, ff_data)

    def finish(self):
        Provenance().add_output_file(
            self.output_file,
            role='mon.tel.flatfield'
        )
        self.writer.close()


def main():
    exe = PedestalsHDF5Writer()
    exe.run()


if __name__ == '__main__':
    main()
