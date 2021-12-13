import numpy as np
from tqdm import tqdm
import numba
import tables

from ctapipe.io.hdf5tableio import HDF5TableWriter
from ctapipe.io.tableio import FixedPointColumnTransform
from ctapipe.core import Tool, Provenance, ToolConfigurationError, Container, Field
from ctapipe.core.traits import Path, Integer, flag, Bool

from ctapipe_io_lst import LSTEventSource
from ctapipe_io_lst.calibration import get_spike_A_positions
from ctapipe_io_lst.constants import N_GAINS, N_PIXELS, N_CAPACITORS_PIXEL, N_SAMPLES

from ..statistics import OnlineStats



class DRS4CalibrationContainer(Container):
    baseline_mean = Field(
        None,
        "Mean baseline of each capacitor, shape (N_GAINS, N_PIXELS, N_CAPACITORS_PIXEL)",
        dtype=np.float32,
        ndim=3,
    )
    baseline_std = Field(
        None,
        "Std Dev. of the baseline calculation, shape (N_GAINS, N_PIXELS, N_CAPACITORS_PIXEL)",
        dtype=np.float32,
        ndim=3,
    )
    baseline_counts = Field(
        None,
        "Number of events used for the baseline calculation, shape (N_GAINS, N_PIXELS, N_CAPACITORS_PIXEL)",
        dtype=np.int32,
        ndim=3,
    )

    spike_height = Field(
        None,
        "Mean spike height for each pixel, shape (N_GAINS, N_PIXELS, 3)",
        ndim=3,
        dtype=np.float32,
    )


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

    output_path = Path(
        directory_ok=False,
        help='Path for the output hdf5 file of pedestal baseline and spike heights',
    ).tag(config=True)
    skip_samples_front = Integer(
        default_value=10,
        help='Do not include first N samples in pedestal calculation'
    ).tag(config=True)
    skip_samples_end = Integer(
        default_value=1,
        help='Do not include last N samples in pedestal calculation'
    ).tag(config=True)

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

        self.source = LSTEventSource(
            parent=self,
            pointing_information=False,
            trigger_information=False,
        )

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

            # np.ma does not raise an error if the weights sum to 0
            mean_height = np.ma.average(spike_height, weights=counts, axis=2)
            # convert masked array to dense, replacing invalid values with nan
            spike_heights[:, :, i] = mean_height.filled(np.nan)

        return spike_heights

    def finish(self):
        tel_id = self.source.tel_id
        self.log.info('Writing output to %s', self.output_path)
        key = f'r1/monitoring/drs4_baseline/tel_{tel_id:03d}'


        with HDF5TableWriter(self.output_path) as writer:
            Provenance().add_output_file(str(self.output_path))
            trafo = FixedPointColumnTransform(
                scale=10,
                offset=0,
                source_dtype=np.float32,
                target_dtype=np.int32,
            )
            for col in ['baseline_mean', 'baseline_std', 'spike_height']:
                writer.add_column_transform(key, col, trafo)

            shape = (N_GAINS, N_PIXELS, N_CAPACITORS_PIXEL)
            spike_height = self.mean_spike_height()

            drs4_calibration = DRS4CalibrationContainer(
                baseline_mean=self.baseline_stats.mean.reshape(shape).astype(np.float32),
                baseline_std=self.baseline_stats.std.reshape(shape).astype(np.float32),
                baseline_counts=self.baseline_stats.counts.reshape(shape).astype(np.int32),
                spike_height=spike_height,
            )

            writer.write(key, drs4_calibration)


def main():
    tool = DRS4PedestalAndSpikeHeight()
    tool.run()


if __name__ == '__main__':
    main()
