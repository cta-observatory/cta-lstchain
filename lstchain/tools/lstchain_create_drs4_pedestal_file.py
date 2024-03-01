import numba
import numpy as np
from ctapipe.core import (
    Container,
    Field,
    Provenance,
    Tool,
    ToolConfigurationError,
)
from ctapipe.core.traits import (
    Bool,
    Integer,
    Path,
    flag,
)
from ctapipe.io.hdf5tableio import HDF5TableWriter
from ctapipe_io_lst import LSTEventSource, EVBPreprocessingFlag, TriggerBits
from ctapipe_io_lst.calibration import get_spike_A_positions
from ctapipe_io_lst.constants import (
    N_CAPACITORS_PIXEL,
    N_GAINS,
    N_PIXELS,
    N_SAMPLES,
)
from tqdm import tqdm

from ..statistics import OnlineStats


class DRS4CalibrationContainer(Container):
    baseline_mean = Field(
        None,
        "Mean baseline of each capacitor, shape (N_GAINS, N_PIXELS, N_CAPACITORS_PIXEL)",
        dtype=np.int16,
        ndim=3,
    )
    baseline_std = Field(
        None,
        "Std Dev. of the baseline calculation, shape (N_GAINS, N_PIXELS, N_CAPACITORS_PIXEL)",
        dtype=np.int16,
        ndim=3,
    )
    baseline_counts = Field(
        None,
        "Number of events used for the baseline calculation, shape (N_GAINS, N_PIXELS, N_CAPACITORS_PIXEL)",
        dtype=np.uint16,
        ndim=3,
    )

    spike_height = Field(
        None,
        "Mean spike height for each pixel, shape (N_GAINS, N_PIXELS, 3)",
        ndim=3,
        dtype=np.int16,
    )


def convert_to_int16(array):
    '''Convert an array to int16, rounding and clipping before to avoid under/overflow'''
    dtype = np.int16
    info = np.iinfo(dtype)
    array = np.clip(np.round(array), info.min, info.max)
    return array.astype(dtype)


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
        help="Show progress bar during processing", default_value=True
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
            "Show a progress bar during event processing",
            "Do not show a progress bar during event processing",
        ),
        **flag(
            "full-statistics",
            "DRS4PedestalAndSpikeHeight.full_statistics",
            "Whether to write the full statistics about spikes or not",
        ),
        **flag(
            "timelapse-correction",
            "LSTR0Corrections.apply_timelapse_correction",
            "Whether to apply drs4 timelapse correction or not.",
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

        unknown_spike_heights = np.isnan(spike_heights).any(axis=2)
        n_unknown_spike_heights = np.count_nonzero(unknown_spike_heights)

        if n_unknown_spike_heights > 0:
            self.log.warning(f'Could not determine spike height for {n_unknown_spike_heights} channels')
            self.log.warning(f'Gain, pixel: {np.nonzero(unknown_spike_heights)}')

            # replace any unknown pixels with the mean over the camera
            camera_mean_spike_height = np.nanmean(spike_heights, axis=(0, 1))
            self.log.warning(f'Using camera mean of {camera_mean_spike_height} for these pixels')
            spike_heights[unknown_spike_heights] = camera_mean_spike_height

        return spike_heights

    def check_baseline_values(self, baseline_mean):
        """
        Check the values of the drs4 baseline for issues.

        In case of no EVBv5 data or no pre-calibration by EVBv6,
        we expect no negative and no small values.

        In case of EVBv6 applied baseline corrections, values should
        be close to 0, but negative values are ok.
        """

        check_large_values = False
        check_negative = True
        check_small = True

        if self.source.cta_r1:
            preprocessing = self.source.evb_preprocessing[TriggerBits.MONO]
            if EVBPreprocessingFlag.BASELINE_SUBTRACTION in preprocessing:
                check_large_values = True
                check_negative = False
                check_small = False


        if check_negative:
            negative = baseline_mean < 0
            n_negative = np.count_nonzero(negative)
            if n_negative > 0:
                gain, pixel, capacitor = np.nonzero(negative)
                self.log.critical(f'{n_negative} baseline values are smaller than 0')
                self.log.info("Gain | Pixel | Capacitor | Baseline ")
                for g, p, c in zip(gain, pixel, capacitor):
                    self.log.info(
                        f"{g:4d} | {p:4d} | {c:9d} | {baseline_mean[g][p][c]:6.1f}"
                    )

        if check_small:
            small = baseline_mean < 25
            n_small = np.count_nonzero(small)
            if n_small > 0:
                gain, pixel, capacitor = np.nonzero(small)
                self.log.warning(f'{n_small} baseline values are smaller than 25')
                self.log.info("Gain | Pixel | Capacitor | Baseline ")
                for g, p, c in zip(gain, pixel, capacitor):
                    self.log.info(
                        f"{g:4d} | {p:4d} | {c:9d} | {baseline_mean[g][p][c]:6.1f}"
                    )

        if check_large_values:
            large = np.abs(baseline_mean) > 25
            n_large = np.count_nonzero(large)
            if n_large > 0:
                gain, pixel, capacitor = np.nonzero(large)
                self.log.warning(f'{n_large} baseline values have an abs value >= 25')
                self.log.info("Gain | Pixel | Capacitor | Baseline ")
                for g, p, c in zip(gain, pixel, capacitor):
                    self.log.info(
                        f"{g:4d} | {p:4d} | {c:9d} | {baseline_mean[g][p][c]:6.1f}"
                    )

    def finish(self):
        tel_id = self.source.tel_id
        self.log.info('Writing output to %s', self.output_path)
        key = f'r1/monitoring/drs4_baseline/tel_{tel_id:03d}'

        shape = (N_GAINS, N_PIXELS, N_CAPACITORS_PIXEL)
        baseline_mean = self.baseline_stats.mean.reshape(shape)
        baseline_std = self.baseline_stats.std.reshape(shape)
        baseline_counts = self.baseline_stats.counts.reshape(shape).astype(np.uint16)

        self.check_baseline_values(baseline_mean)

        # Convert baseline mean and spike heights to uint16, handle missing
        # values and values smaller 0, larger maxint
        baseline_mean = convert_to_int16(baseline_mean)
        baseline_std = convert_to_int16(baseline_std)
        spike_height = convert_to_int16(self.mean_spike_height())

        with HDF5TableWriter(self.output_path) as writer:
            Provenance().add_output_file(str(self.output_path))

            drs4_calibration = DRS4CalibrationContainer(
                baseline_mean=baseline_mean,
                baseline_std=baseline_std,
                baseline_counts=baseline_counts,
                spike_height=spike_height,
            )

            writer.write(key, drs4_calibration)


def main():
    tool = DRS4PedestalAndSpikeHeight()
    tool.run()


if __name__ == '__main__':
    main()
