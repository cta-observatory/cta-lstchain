import os
from functools import partial
from pathlib import Path

import h5py
import numpy as np
from ctapipe.coordinates import EngineeringCameraFrame
from ctapipe.core import Tool, traits
from ctapipe.visualization import CameraDisplay
from ctapipe_io_lst import constants
from ctapipe_io_lst import load_camera_geometry
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import MaxNLocator
from scipy.optimize import curve_fit
from traitlets import List, Int, Dict, Float
from ..io import read_calibration_file

__all__ = [
    'FitIntensityScan'
]

MIN_N_RUNS = 5


class FitIntensityScan(Tool):
    """
     Tool that generates a HDF5 file with the results of the fit
     of the signal of an intensity scan (filter scan in the case of LST), this is useful to estimate the
     quadratic noise term to include in the standard F-factor formula

     To be run with
     lstchain_fit_intensity_scan --config config.json

     """

    name = "FitFilterScan"
    description = "Tool to fit an intensity scan"

    signal_range = List(
        [[1500, 14000], [200, 14000]],
        help='Signal range to include in the fit for [HG,LG] (camera median in [ADC])'
    ).tag(config=True)

    gain_channels = List(
        [0, 1],
        help='Gain channel to process (HG=0, LG=1)'
    ).tag(config=True)

    sub_run = Int(
        0,
        help='Sub run number to process'
    ).tag(config=True)

    run_list = List(
        help='List of runs',
    ).tag(config=True)

    input_dir = traits.Path(
        directory_ok=True,
        help='directory with the input files',
    ).tag(config=True)

    input_prefix = traits.Unicode(
        default_value="calibration",
        help='Prefix to select calibration files to fit',
    ).tag(config=True)

    output_path = traits.Path(
        directory_ok=False, default_value="filter_scan_fit.h5",
        help='Path the output hdf5 file',
    ).tag(config=True)

    plot_path = traits.Path(
        directory_ok=False, default_value="filter_scan_fit.pdf",
        help='Path to pdf file with check plots',
    ).tag(config=True)

    fit_initialization = List(
        [[100.0, 0.001], [6.0, 0.001]],
        help='Fit parameters initalization [gain (ADC/pe), B term] for HG and LG'
    ).tag(config=True)

    fractional_variance_error = Float(
        0.02,
        help='Constant fractional error assumed for the y fit coordinate (variance)'
    ).tag(config=True)

    squared_excess_noise_factor = Float(
        1.222,
        help='Excess noise factor squared: 1+ Var(gain)/Mean(Gain)**2'
    ).tag(config=True)

    aliases = Dict(dict(
        signal_range='FitIntensityScan.signal_range',
        input_dir='FitIntensityScan.input_dir',
        output_path='FitIntensityScan.output_path',
        plot_path='FitIntensityScan.plot_path',
        sub_run='FitIntensityScan.sub_run',
        gain_channels='FitIntensityScan.gain_channels',
        run_list='FitIntensityScan.run_list',
        input_prefix='FitIntensityScan.input_prefix',
    ))

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        """
         For getting help run:
         python calc_camera_calibration.py --help
         
        """
        for chan in self.gain_channels:
            if not self.signal_range[chan]:
                raise ValueError(f"Trailet signal_range {self.signal_range} inconsistent with"
                                 f"trailet {self.gain_channels}. \n")

        self.unusable_pixels = [None, None]
        self.signal = [None, None]
        self.variance = [None, None]
        self.selected_runs = [[], []]
        self.fit_parameters = np.zeros((constants.N_GAINS, constants.N_PIXELS, 2))
        self.fit_cov_matrix = np.zeros((constants.N_GAINS, constants.N_PIXELS, 4))
        self.fit_error = np.zeros((constants.N_GAINS, constants.N_PIXELS))

    def setup(self):
        channel = ["HG", "LG"]

        # loop on runs and memorize data
        try:
            for i, run in enumerate(self.run_list):

                file_list = sorted(
                    Path(f"{self.input_dir}").rglob(f'{self.input_prefix}*.Run{run:05d}.{self.sub_run:04d}.h5'))

                if len(file_list) == 0:
                    raise IOError(f"Input file for run {run} do not found. \n")

                if len(file_list) > 1:
                    raise IOError(f"Input file for run {run} is more than one: {file_list} \n")

                inp_file = file_list[0]
                if os.path.getsize(inp_file) < 100:
                    raise IOError(f"file size run {run} is too short \n")

                self.log.debug(f"Read file {inp_file}")               
                mon = read_calibration_file(inp_file)
                
                for chan in self.gain_channels:
                    # verify that the median signal is inside the asked range
                    median_charge = np.nanmedian(mon.flatfield.charge_median[chan])

                    if median_charge > self.signal_range[chan][1] or median_charge < self.signal_range[chan][0]:
                        self.log.debug(
                            f"{channel[chan]}: skip run {run}, signal out of range {median_charge:6.1f} ADC")
                        continue

                    signal = mon.flatfield.charge_median[chan] - mon.pedestal.charge_median[chan]
                    variance = mon.flatfield.charge_std[chan] ** 2 - mon.pedestal.charge_std[chan] ** 2

                    if self.signal[chan] is None:
                        self.signal[chan] = signal
                        self.variance[chan] = variance
                        self.unusable_pixels[chan] = mon.calibration.unusable_pixels[chan]

                    else:
                        self.signal[chan] = np.column_stack((self.signal[chan], signal))
                        self.variance[chan] = np.column_stack((self.variance[chan], variance))
                        self.unusable_pixels[chan] = np.column_stack(
                            (self.unusable_pixels[chan], mon.calibration.unusable_pixels[chan]))
                    self.selected_runs[chan].append(run)
                    self.log.info(f"{channel[chan]}: select run {run}, median charge {median_charge:6.1f} ADC\n")

            # check to have enough selected runs
            for chan in self.gain_channels:
                if self.signal[chan] is None:
                    raise IOError(f"--> Zero runs selected for channel {channel[chan]} \n")

                if self.signal[chan].size < MIN_N_RUNS * constants.N_PIXELS:
                    raise IOError(
                        f"--> Not enough runs selected for channel {channel[chan]}: {int(self.signal[chan].size / constants.N_PIXELS)} runs \n")

        except ValueError as e:
            self.log.error(e)

    def start(self):
        '''loop to fit each pixel '''

        # only positive parameters
        bounds = [0, 200]

        funfit = partial(quadratic_fit, f2=self.squared_excess_noise_factor)

        for pix in np.arange(constants.N_PIXELS):

            if pix % 100 == 0:
                self.log.debug(f"Pixel {pix}")

            # loop over channel
            for chan in self.gain_channels:

                # fit parameters initialization
                p0 = np.array(self.fit_initialization[chan])

                mask = self.unusable_pixels[chan][pix]
                sig = np.ma.array(self.signal[chan][pix], mask=mask).compressed()
                var = np.ma.array(self.variance[chan][pix], mask=mask).compressed()

                # skip the pixel if not enough data
                if sig.shape[0] < MIN_N_RUNS:
                    self.log.debug(
                        f"Not enough data in pixel {pix} and channel {chan} for the fit ({sig.shape[0]} runs)\n")
                    self.fit_error[chan, pix] = 1
                    continue

                # we assume a constant fractional error
                sigma = self.fractional_variance_error * var

                try:
                    par, par_cov = curve_fit(funfit, sig, var, bounds=bounds, sigma=sigma, p0=p0)
                    self.fit_parameters[chan, pix] = par
                    self.fit_cov_matrix[chan, pix] = par_cov.reshape(4)

                except Exception as e:

                    self.log.error(e)
                    self.log.error(f"Error for pixel {pix} and channel {chan}:\n")
                    self.log.error(f"signal {sig}\n")
                    self.log.error(f"variance {var}\n")

                    self.fit_error[chan, pix] = 1

    def finish(self):
        """
        write fit results in h5 file and the check-plots in pdf file
        """

        gain = np.ma.array(self.fit_parameters.T[0], mask=self.fit_error.T)
        quadratic_term = np.ma.array(self.fit_parameters.T[1], mask=self.fit_error.T)

        # give to the badly fitted pixel a median value for the B term
        median_quadratic_term = np.ma.median(quadratic_term, axis=0)

        fill_array = np.ones((constants.N_PIXELS, constants.N_GAINS)) * median_quadratic_term

        quadratic_term_corrected = np.ma.filled(quadratic_term, fill_array)

        with h5py.File(self.output_path, 'w') as hf:
            hf.create_dataset('gain', data=gain.T)
            hf.create_dataset('B_term', data=quadratic_term_corrected.T)
            hf.create_dataset('covariance_matrix', data=self.fit_cov_matrix)
            hf.create_dataset('bad_fit_mask', data=self.fit_error)

            # remember the camera median and the variance per run
            channel = ["HG", "LG"]
            for chan in [0, 1]:
                if self.signal[chan] is not None:
                    hf.create_dataset(f'signal_{channel[chan]}', data=self.signal[chan])
                    hf.create_dataset(f'variance_{channel[chan]}', data=self.variance[chan])
                    hf.create_dataset(f'runs_{channel[chan]}', data=self.selected_runs[chan])

            hf.create_dataset('runs', data=self.run_list)
            hf.create_dataset('sub_run', data=self.sub_run)

            # plot open pdf
            with PdfPages(self.plot_path) as pdf:
                plt.rc("font", size=15)

                for chan in self.gain_channels:
                    # plot the used runs and their median camera charge
                    fig = plt.figure((chan + 1), figsize=(8, 20))
                    fig.suptitle(f"{channel[chan]} channel", fontsize=25)
                    ax = plt.subplot(2, 1, 1)
                    ax.grid(True)
                    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
                    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
                    ax.yaxis.set_major_locator(plt.MultipleLocator(1))

                    plt.plot(np.nanmedian(self.signal[chan], axis=0), self.selected_runs[chan], "o")
                    plt.xlabel(r'$\mathrm{\overline{Q}-\overline{ped}}$ [ADC]')
                    plt.ylabel(r'Runs used in the fit')

                    plt.subplot(2, 1, 2)
                    camera = load_camera_geometry()
                    camera = camera.transform_to(EngineeringCameraFrame())
                    disp = CameraDisplay(camera)
                    image = self.fit_parameters.T[1].T * 100
                    mymin = np.median(image[chan]) - 3 * np.std(image[chan])
                    mymax = np.median(image[chan]) + 3 * np.std(image[chan])
                    disp.set_limits_minmax(mymin, mymax)
                    mask = np.where(self.fit_error[chan] == 1)[0]
                    disp.highlight_pixels(mask, linewidth=2.5, color="green")
                    disp.image = image[chan]
                    disp.cmap = plt.cm.coolwarm
                    plt.title(f"{channel[chan]} Fitted B values [%]")
                    disp.add_colorbar()
                    plt.tight_layout()
                    pdf.savefig()

                    # plot the fit results and residuals for four arbitrary  pixels
                    fig = plt.figure((chan + 1) * 10, figsize=(11, 22))
                    fig.suptitle(f"{channel[chan]} channel", fontsize=25)

                    pad = 0
                    for pix in [0, 600, 1200, 1800]:
                        pad += 1
                        plt.subplot(4, 2, pad)
                        plt.grid(which='minor')

                        mask = self.unusable_pixels[chan][pix]
                        sig = np.ma.array(self.signal[chan][pix], mask=mask).compressed()
                        var = np.ma.array(self.variance[chan][pix], mask=mask).compressed()
                        popt = self.fit_parameters[chan, pix]

                        # plot points
                        plt.plot(sig, var, 'o', color="C0")

                        # plot fit
                        min_x = min(1000, np.min(sig) * 0.9)
                        max_x = max(10000, np.max(sig) * 1.1)
                        x = np.arange(np.min(sig), np.max(sig))

                        plt.plot(x, quadratic_fit(x, *popt), '--', color="C1",
                                 label=f'Pixel {pix}:\ng={popt[0]:5.2f} [ADC/pe] , B={popt[1]:5.3f}')
                        plt.xlim(min_x, max_x)
                        plt.xlabel('Q-ped [ADC]')
                        plt.ylabel(r'$\mathrm{\sigma_Q^2-\sigma_{ped}^2}$ [$ADC^2$]')
                        plt.xscale('log')
                        plt.yscale('log')
                        plt.legend()

                        # plot residuals
                        pad += 1
                        plt.subplot(4, 2, pad)
                        plt.grid(which='both', axis='both')

                        popt = self.fit_parameters[chan, pix]
                        plt.plot(sig, (quadratic_fit(sig, *popt) - var) / var * 100, 'o', color="C0")
                        plt.xlim(min_x, max_x)
                        plt.xscale('log')
                        plt.ylabel('fit residuals %')
                        plt.xlabel('Q-ped [ADC]')
                        plt.hlines(0, 0, np.max(sig), linestyle='dashed', color="black")

                    plt.tight_layout()
                    pdf.savefig()


def quadratic_fit(t, b=1, c=1, f2=1.222):
    return b * f2 * t + c ** 2 * t ** 2


def main():
    exe = FitIntensityScan()

    exe.run()


if __name__ == '__main__':
    main()
