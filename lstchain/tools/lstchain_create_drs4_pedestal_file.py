import numpy as np
from tqdm import tqdm
import numba
import tables
from traitlets.config import Config

from ctapipe.io.hdf5tableio import DEFAULT_FILTERS
from ctapipe.core import Tool, Provenance, ToolConfigurationError
from ctapipe.core.traits import Path, Integer, flag, Bool

from ctapipe_io_lst import LSTEventSource
from ctapipe_io_lst.calibration import get_spike_A_positions
from ctapipe_io_lst.constants import N_GAINS, N_PIXELS, N_CAPACITORS_PIXEL, N_SAMPLES

from ..statistics import OnlineStats


@numba.njit(cache=True, inline='always')
def flat_index(gain, pixel, cap):
    return N_PIXELS * N_CAPACITORS_PIXEL * gain + N_CAPACITORS_PIXEL * pixel + cap


@numba.njit(cache=True)
def fill_stats(
    waveform,
    first_cap,
    last_first_cap,
    last_readout_time,
    baseline_stats,
    spike0_stats,
    spike1_stats,
    spike2_stats,
    skip_samples_front,
    skip_samples_end,
):
    for gain in range(N_GAINS):
        for pixel in range(N_PIXELS):
            fc = first_cap[gain, pixel]
            last_fc = last_first_cap[gain, pixel]
            spike_positions = get_spike_A_positions(fc, last_fc)

            for sample in range(skip_samples_front, N_SAMPLES - skip_samples_end):
                cap = (fc + sample) % N_CAPACITORS_PIXEL

                # ignore samples where we don't have a last readout time yet
                if last_readout_time[gain, pixel, cap] == 0:
                    continue

                idx = flat_index(gain, pixel, cap)

                # if sample in spike_positions or (sample - 1) in spike_positions or (sample - 2) in spike_positions:
                if sample in spike_positions:
                    spike0_stats.add_value(idx, waveform[gain, pixel, sample])
                elif sample - 1 in spike_positions:
                    spike1_stats.add_value(idx, waveform[gain, pixel, sample])
                elif sample - 2 in spike_positions:
                    spike2_stats.add_value(idx, waveform[gain, pixel, sample])
                else:
                    baseline_stats.add_value(idx, waveform[gain, pixel, sample])


class DRS4PedestalAndSpikeHeight(Tool):
    name = 'lstchain_create_drs4_pedestal_file'

    output_path = Path(directory_ok=False).tag(config=True)
    skip_samples_front = Integer(default_value=10).tag(config=True)
    skip_samples_end = Integer(default_value=1).tag(config=True)

    progress_bar = Bool(
        help="show progress bar during processing", default_value=False
    ).tag(config=True)

    full_statistics = Bool(
        help=(
            "If True, write spike{1,2,3} mean, count, std for each capacitor."
            " Otherwise, only mean spike height for each gain, pixel is written"
        ),
        default_value=False
    ).tag(config=True)

    overwrite = Bool(
        help=(
            "If true, overwrite output without asking,"
            " else fail if output file already exists"
        ),
        default_value=False
    ).tag(config=True)

    aliases = {
        ('i', 'input'): 'LSTEventSource.input_url',
        ('o', 'output'): 'DRS4PedestalAndSpikeHeight.output_path',
        ('m', 'max-events'): 'LSTEventSource.max_events',
    }


    flags = {
        **flag(
            "overwrite",
            "DRS4PedestalAndSpikeHeight.overwrite",
            "Overwrite output file if it exists",
            "Fail if output file already exists",
        ),
        **flag(
            "progress",
            "DRS4PedestalAndSpikeHeight.progress_bar",
            "show a progress bar during event processing",
            "don't show a progress bar during event processing",
        ),
        **flag(
            "full-statistics",
            "DRS4PedestalAndSpikeHeight.full_statistics",
            "Wether to write the full statistics about spikes or not",
        ),
    }

    classes = [LSTEventSource]

    def setup(self):
        self.output_path = self.output_path.expanduser().resolve()
        if self.output_path.exists():
            if self.overwrite:
                self.log.warning("Overwriting %s", self.output_path)
                self.output_path.unlink()
            else:
                raise ToolConfigurationError(
                    f"Output file {self.output_path} exists"
                    ", use the `overwrite` option or choose another `output_path` "
                )
        self.log.debug("output path: %s", self.output_path)
        Provenance().add_output_file(str(self.output_path), role="DL1/Event")

        self.source = LSTEventSource(parent=self)

        # set some config options, these are necessary for this tool,
        # so we set them here and not via the config system
        self.source.r0_r1_calibrator.r1_sample_start = 0
        self.source.r0_r1_calibrator.r1_sample_end = N_SAMPLES

        self.source.r0_r1_calibrator.offset = 0
        self.source.r0_r1_calibrator.apply_spike_correction = False
        self.source.r0_r1_calibrator.apply_timelapse_correction = True
        self.source.r0_r1_calibrator.apply_drs4_pedestal_correction = False

        n_stats = N_GAINS * N_PIXELS * N_CAPACITORS_PIXEL
        self.baseline_stats = OnlineStats(n_stats)
        self.spike0_stats = OnlineStats(n_stats)
        self.spike1_stats = OnlineStats(n_stats)
        self.spike2_stats = OnlineStats(n_stats)

    def start(self):
        tel_id = self.source.tel_id

        for event in tqdm(self.source, disable=not self.progress_bar):
            fill_stats(
                event.r1.tel[tel_id].waveform,
                self.source.r0_r1_calibrator.first_cap[tel_id],
                self.source.r0_r1_calibrator.first_cap_old[tel_id],
                self.source.r0_r1_calibrator.last_readout_time[tel_id],
                self.baseline_stats,
                self.spike0_stats,
                self.spike1_stats,
                self.spike2_stats,
                skip_samples_front=self.skip_samples_front,
                skip_samples_end=self.skip_samples_end,
            )

    def mean_spike_height(self):
        '''Calculate mean spike height for each gain, pixel'''
        shape = (N_GAINS, N_PIXELS, N_CAPACITORS_PIXEL)
        mean_baseline = self.baseline_stats.mean.reshape(shape)
        spike_heights = np.full((N_GAINS, N_PIXELS, 3), np.nan, dtype=np.float32)

        for i in range(3):
            stats = getattr(self, f'spike{i}_stats')
            counts = stats.counts.reshape(shape)
            spike_height = stats.mean.reshape(shape) - mean_baseline
            spike_height[counts == 0] = 0
            mean_height = np.average(spike_height, weights=counts, axis=2)
            spike_heights[:, :, i] = mean_height

        return spike_heights

    def finish(self):
        self.log.info('Writing output to %s', self.output_path)

        with tables.open_file(self.output_path, 'w') as f:
            Provenance().add_output_file(str(self.output_path))

            shape = (N_GAINS, N_PIXELS, N_CAPACITORS_PIXEL)

            g = f.create_group('/', 'baseline')
            for attr in ('counts', 'mean', 'std'):
                array = getattr(self.baseline_stats, attr).reshape(shape)

                # store mean / std as int32 scaled by 100
                # aka fixed precision with two decimal digits
                if attr in ('mean', 'std'):
                    array *= 100

                array = array.astype(np.int32)
                f.create_carray(g, attr, obj=array, filters=DEFAULT_FILTERS)


            f.create_carray(
                "/", "spike_height",
                obj=self.mean_spike_height()
            )

            mean_baseline = self.baseline_stats.mean.reshape(shape)
            if self.full_statistics:
                for i in range(3):
                    stats = getattr(self, f'spike{i}_stats')
                    g = f.create_group('/', f'spike{i}')
                    for attr in ('counts', 'mean', 'std'):
                        array = getattr(stats, attr).reshape(shape)

                        # correct mean for baseline
                        if attr == 'mean':
                            array -= mean_baseline

                        # store mean / std as int32 scaled by 100
                        # aka fixed precision with two decimal digits
                        if attr in ('mean', 'std'):
                            array *= 100

                        array = array.astype(np.int32)
                        f.create_carray(g, attr, obj=array, filters=DEFAULT_FILTERS)


def main():
    tool = DRS4PedestalAndSpikeHeight()
    tool.run()


if __name__ == '__main__':
    main()
