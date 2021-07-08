
import os
import numpy as np
import h5py
from scipy.optimize import curve_fit
from traitlets import List,Int
from ctapipe.core import Tool, traits

from ctapipe.containers import (
    FlatFieldContainer,
    WaveformCalibrationContainer,
    PedestalContainer,
)
from ctapipe.io.hdf5tableio import HDF5TableReader
from ctapipe_io_lst import constants
#from lstchain.visualization import plot_calib as calib

__all__ = [
    'FitFilterScan'
]


class FitFilterScan(Tool):
    """
     Tool that generates a HDF5 file with the results of the fit
     of the signal of filter scan, this is useful to estimate the
     quadratic noise term to include in the standard F-factor formula
     """

    name = "FitFilterScan"
    description = "Tool to fit a filter scan"

    signal_range = List(
        [400,12000],
        help='Signal charge range  (camera median in [ADC])'
    ).tag(config=True)

    gain_channel = Int(
        0,
        help='Gain channel to process (HG=0, LG=1)'
    ).tag(config=True)

    sub_run = Int(
        0,
        help='Sub run number to process'
    ).tag(config=True)

    run_list = List(
        help='List of runs',
    ).tag(config=True)

    filter_list = List(
        help='List of filters',
    ).tag(config=True)

    input_dir = traits.Path(
        directory_ok=True,
        help='directory with the input files',
    ).tag(config=True)

    output_path = traits.Path(
        directory_ok=False, default="filter_scan_fit.h5",
        help='Path to file with list of runs',
    ).tag(config=True)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        """
         For getting help run:
         python calc_camera_calibration.py --help
         
        """

        self.unusable_pixels = None
        self.signal = None
        self.variance = None
        self.fit_parameters = np.zeros((constants.N_PIXELS,2))
        self.fit_cov_matrix = np.zeros((constants.N_PIXELS,4))
        self.fit_error = np.zeros(constants.N_PIXELS)

    def setup(self):

        ff_data = FlatFieldContainer()
        ped_data = PedestalContainer()
        calib_data = WaveformCalibrationContainer()

        chan=self.gain_channel

        # loop on runs and memorize data
        try:
            for i in np.arange(len(self.run_list)):
                run = self.run_list[i]

                inp_file = f"{self.input_dir}/calibration.Run{run:05d}.{self.sub_run:04d}.h5"

                if not os.path.exists(inp_file):
                    raise IOError(f"Input file for run {run} does not exists. \n")

                if os.path.getsize(inp_file) < 100:
                    raise IOError(f"file size run {run} is too short \n")

                if read_calibration_file(inp_file,ff_data,calib_data, ped_data):

                    # verify that the median signal is inside the asked range
                    median_charge = np.median(ff_data.charge_median[chan])

                    if median_charge > self.signal_range[1] or median_charge < self.signal_range[0]:
                        self.log.debug(f"!!! signal to low or too high for run {run}. Skipped")
                        continue

                    signal = ff_data.charge_median[chan] - ped_data.charge_median[chan]
                    variance = ff_data.charge_std[chan] ** 2 - ped_data.charge_std[chan] ** 2

                    if self.signal is None:
                        self.signal = signal
                        self.variance = variance
                        self.unusable_pixels = calib_data.unusable_pixels[chan]

                    else:
                        self.signal=np.column_stack((self.signal, signal))
                        self.variance = np.column_stack((self.variance, variance))
                        self.unusable_pixels = np.column_stack((self.unusable_pixels,calib_data.unusable_pixels[chan]))

                    self.log.debug(f"Select run {run}, median charge {median_charge}\n")
                else:
                    raise IOError(f"--> Problem in reading {run}\n")


        except ValueError as e:
            self.log.error(e)


    def start(self):
        '''loop to fit each pixel '''

        # fit parameters initialization
        if self.gain_channel == 1:
            p0 = np.array([6.0, 0.001])
        else:
            p0 = np.array([100.0, 0.001])
        bounds = [0, 200]

        # loop over pixels
        for pix in np.arange(constants.N_PIXELS):
            if pix % 100 == 0 :
                self.log.debug(f"Pixel {pix}")

            mask = self.unusable_pixels[pix]
            sig = np.ma.array(self.signal[pix],mask=mask).compressed()
            var = np.ma.array(self.variance[pix],mask=mask).compressed()

            # skip the pixel if not enough data
            if sig.shape[0] < 5:
                self.log.debug(f"Not enough data in pixel {pix} for the fit ({sig.shape(0)} runs)\n")
                self.fit_error[pix] = 1
                continue

            # we assume a 2% error
            sigma = 0.02 * var
            try:
                par, par_cov = curve_fit(quadratic_fit, sig, var, bounds=bounds, sigma=sigma, p0=p0)
                self.fit_parameters[pix] = par
                self.fit_cov_matrix[pix] = par_cov.reshape(4)

            except ValueError as e:
                self.log.error(e)
                self.log.error(f"Error for pixel {pix}:\n")
                self.log.error(f"signal {sig}\n")
                self.log.error(f"variance {var}\n")

                self.fit_error[pix] = 1

    def finish(self):
        """
        write fit results to h5 file
        """
        gain = np.ma.array(self.fit_parameters.T[0], mask=self.fit_error)
        quadratic_term = np.ma.array(self.fit_parameters.T[1], mask=self.fit_error)

        # give to the badly fitted pixel a median value for the B term
        median_quadratic_term = np.ma.median(quadratic_term)
        fill_array = np.ones(constants.N_PIXELS) * median_quadratic_term

        quadratic_term_corrected = np.ma.filled(quadratic_term, fill_array)
        with h5py.File(self.output_path, 'w') as hf:
            hf.create_dataset('gain', data=gain)
            hf.create_dataset('B_term', data=quadratic_term_corrected)
            hf.create_dataset('covariance_matrix', data=self.fit_cov_matrix)

            hf.create_dataset('bad_fit_mask', data=self.fit_error)
            hf.create_dataset('median_signal', data=np.median(self.signal,axis=1))
            hf.create_dataset('median_variance', data=np.median(self.variance,axis=1))

            hf.create_dataset('used_runs', data=self.run_list)
            hf.create_dataset('used_filter', data=self.filter_list)



def quadratic_fit(t, b=1, c=1):
    F2 = 1.222
    return b * F2 * t + c ** 2 * t ** 2

def read_calibration_file(file_name, ff_data, calib_data, ped_data, tel_id=1):
    with HDF5TableReader(file_name) as h5_table:

        try:
            assert h5_table._h5file.isopen == True

            table = f"/tel_{tel_id}/flatfield"
            next(h5_table.read(table, ff_data))

            table = f"/tel_{tel_id}/calibration"
            next(h5_table.read(table, calib_data))

            table = f"/tel_{tel_id}/pedestal"
            next(h5_table.read(table, ped_data))

        except Exception:
            print(f"----> no correct tables {table} in {file_name}")
            h5_table.close()
            return False

    h5_table.close()
    return True

def main():
    exe = FitFilterScan()

    exe.run()


if __name__ == '__main__':
    main()
