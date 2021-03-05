from matplotlib import pyplot as plt
from ctapipe.visualization import CameraDisplay
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from ctapipe_io_lst import load_camera_geometry
from ctapipe.coordinates import EngineeringCameraFrame

# read back the monitoring containers written with the tool calc_camera_calibration.py
from ctapipe.containers import (
    FlatFieldContainer,
    WaveformCalibrationContainer,
    PedestalContainer,
    PixelStatusContainer,
)

from ctapipe.io.hdf5tableio import HDF5TableReader

__all__ = ["read_file", "plot_all"]

ff_data = FlatFieldContainer()
ped_data = PedestalContainer()
calib_data = WaveformCalibrationContainer()
status_data = PixelStatusContainer()

channel = ["HG", "LG"]

plot_dir = "none"


def read_file(file_name, tel_id=1):
    """
    read camera calibration quantities

    Parameters
    ----------
    file_name:   calibration hdf5 file

    tel_id:      telescope id
    """
    with HDF5TableReader(file_name) as h5_table:
        assert h5_table._h5file.isopen == True

        table = f"/tel_{tel_id}/flatfield"
        next(h5_table.read(table, ff_data))
        table = f"/tel_{tel_id}/calibration"
        next(h5_table.read(table, calib_data))
        table = f"/tel_{tel_id}/pedestal"
        next(h5_table.read(table, ped_data))
        table = f"/tel_{tel_id}/pixel_status"
        next(h5_table.read(table, status_data))


def plot_all(ped_data, ff_data, calib_data, run=0, plot_file=None):
    """
    plot camera calibration quantities

    Parameters
    ----------
    ped_data:   pedestal container PedestalContainer()

    ff_data:    flat-field container FlatFieldContainer()

    calib_data: calibration container WaveformCalibrationContainer()

    run: run number

    plot_file: name of the output PDF file. No file is produced if name is not provided

    """
    # read geometry
    camera = load_camera_geometry()
    camera = camera.transform_to(EngineeringCameraFrame())

    # plot open pdf
    if plot_file is not None:
        with PdfPages(plot_file) as pdf:

            plt.rc("font", size=15)

            # first figure
            fig = plt.figure(1, figsize=(12, 24))
            plt.tight_layout()
            fig.suptitle(f"Run {run}", fontsize=25)
            pad = 420
            image = ff_data.charge_median
            mask = ff_data.charge_median_outliers
            for chan in np.arange(2):
                pad += 1
                plt.subplot(pad)
                plt.tight_layout()
                disp = CameraDisplay(camera)
                mymin = np.median(image[chan]) - 2 * np.std(image[chan])
                mymax = np.median(image[chan]) + 2 * np.std(image[chan])
                disp.set_limits_minmax(mymin, mymax)
                disp.highlight_pixels(mask[chan], linewidth=2)
                disp.image = image[chan]
                disp.cmap = plt.cm.coolwarm
                # disp.axes.text(lposx, 0, f'{channel[chan]} signal charge (ADC)', rotation=90)
                plt.title(f"{channel[chan]} signal charge [ADC]")
                disp.add_colorbar()

            image = ff_data.charge_std
            mask = ff_data.charge_std_outliers
            for chan in np.arange(2):
                pad += 1
                plt.subplot(pad)
                plt.tight_layout()
                disp = CameraDisplay(camera)
                mymin = np.median(image[chan]) - 2 * np.std(image[chan])
                mymax = np.median(image[chan]) + 2 * np.std(image[chan])
                disp.set_limits_minmax(mymin, mymax)
                disp.highlight_pixels(mask[chan], linewidth=2)
                disp.image = image[chan]
                disp.cmap = plt.cm.coolwarm
                # disp.axes.text(lposx, 0, f'{channel[chan]} signal std [ADC]', rotation=90)
                plt.title(f"{channel[chan]} signal std [ADC]")
                disp.add_colorbar()

            image = ped_data.charge_median
            mask = ped_data.charge_median_outliers
            for chan in np.arange(2):
                pad += 1
                plt.subplot(pad)
                plt.tight_layout()
                disp = CameraDisplay(camera)
                mymin = np.median(image[chan]) - 2 * np.std(image[chan])
                mymax = np.median(image[chan]) + 2 * np.std(image[chan])
                disp.set_limits_minmax(mymin, mymax)
                disp.highlight_pixels(mask[chan], linewidth=2)
                disp.image = image[chan]
                disp.cmap = plt.cm.coolwarm
                # disp.axes.text(lposx, 0, f'{channel[chan]} pedestal [ADC]', rotation=90)
                plt.title(f"{channel[chan]} pedestal [ADC]")
                disp.add_colorbar()

            image = ped_data.charge_std
            mask = ped_data.charge_std_outliers
            for chan in np.arange(2):
                pad += 1
                plt.subplot(pad)
                plt.tight_layout()
                disp = CameraDisplay(camera)
                mymin = np.median(image[chan]) - 2 * np.std(image[chan])
                mymax = np.median(image[chan]) + 2 * np.std(image[chan])
                disp.set_limits_minmax(mymin, mymax)
                disp.highlight_pixels(mask[chan], linewidth=2)
                disp.image = image[chan]
                disp.cmap = plt.cm.coolwarm
                # disp.axes.text(lposx, 0, f'{channel[chan]} pedestal std [ADC]', rotation=90)
                plt.title(f"{channel[chan]} pedestal std [ADC]")
                disp.add_colorbar()

            plt.subplots_adjust(top=0.92)

            pdf.savefig()
            plt.close()

            # second figure
            fig = plt.figure(2, figsize=(12, 24))
            plt.tight_layout()
            fig.suptitle(f"Run {run}", fontsize=25)
            pad = 420

            # time
            image = ff_data.time_median
            mask = ff_data.time_median_outliers
            for chan in np.arange(2):
                pad += 1
                plt.subplot(pad)
                plt.tight_layout()
                disp = CameraDisplay(camera)
                disp.highlight_pixels(mask[chan], linewidth=2)
                disp.image = image[chan]
                disp.cmap = plt.cm.coolwarm
                # disp.axes.text(lposx, 0, f'{channel[chan]} time', rotation=90)
                plt.title(f"{channel[chan]} time")
                disp.add_colorbar()

            image = ff_data.relative_gain_median
            mask = calib_data.unusable_pixels
            for chan in np.arange(2):
                pad += 1
                plt.subplot(pad)
                plt.tight_layout()
                disp = CameraDisplay(camera)
                disp.highlight_pixels(mask[chan], linewidth=2)
                mymin = np.median(image[chan]) - 2 * np.std(image[chan])
                mymax = np.median(image[chan]) + 2 * np.std(image[chan])
                disp.set_limits_minmax(mymin, mymax)
                disp.image = image[chan]
                disp.cmap = plt.cm.coolwarm
                disp.set_limits_minmax(0.7, 1.3)
                plt.title(f"{channel[chan]} relative signal")
                # disp.axes.text(lposx, 0, f'{channel[chan]} relative gain', rotation=90)
                disp.add_colorbar()

            # pe
            image = calib_data.n_pe
            mask = calib_data.unusable_pixels
            image = np.where(np.isnan(image), 0, image)
            for chan in np.arange(2):
                pad += 1
                plt.subplot(pad)
                plt.tight_layout()
                disp = CameraDisplay(camera)
                disp.highlight_pixels(mask[chan], linewidth=2)
                disp.image = image[chan]
                mymin = np.median(image[chan]) - 2 * np.std(image[chan])
                mymax = np.median(image[chan]) + 2 * np.std(image[chan])
                disp.set_limits_minmax(mymin, mymax)
                disp.cmap = plt.cm.coolwarm
                plt.title(f"{channel[chan]} photon-electrons")
                # disp.axes.text(lposx, 0, f'{channel[chan]} photon-electrons', rotation=90)
                disp.add_colorbar()

            # pe histogram
            pad += 1
            plt.subplot(pad)
            plt.tight_layout()
            for chan in np.arange(2):
                n_pe = calib_data.n_pe[chan]
                # select good pixels
                select = np.logical_not(mask[chan])
                median = int(np.median(n_pe[select]))
                rms = np.std(n_pe[select])
                mymin = median - 4 * rms
                mymax = median + 4 * rms
                label = f"{channel[chan]} Median {median:3.2f}, std {rms:5.2f}"
                plt.hist(
                    n_pe[select],
                    label=label,
                    histtype="step",
                    range=(mymin, mymax),
                    bins=50,
                    stacked=True,
                    alpha=0.5,
                    fill=True,
                )
                plt.legend()
            plt.xlabel(f"pe", fontsize=20)
            plt.ylabel("pixels", fontsize=20)

            # pe scatter plot
            pad += 1
            plt.subplot(pad)
            plt.tight_layout()
            HG = calib_data.n_pe[0]
            LG = calib_data.n_pe[1]
            HG = np.where(np.isnan(HG), 0, HG)
            LG = np.where(np.isnan(LG), 0, LG)
            mymin = np.median(LG) - 2 * np.std(LG)
            mymax = np.median(LG) + 2 * np.std(LG)
            plt.hist2d(LG, HG, bins=[100, 100])
            plt.xlabel("LG", fontsize=20)
            plt.ylabel("HG", fontsize=20)

            x = np.arange(mymin, mymax)
            plt.plot(x, x)
            plt.ylim(mymin, mymax)
            plt.xlim(mymin, mymax)
            plt.subplots_adjust(top=0.92)

            pdf.savefig()
            plt.close()

            # figures 3 and 4: histograms
            for chan in np.arange(2):
                n_pe = calib_data.n_pe[chan]

                gain_median = ff_data.relative_gain_median[chan]
                # charge_median = ff_data.charge_median[chan]
                charge_mean = ff_data.charge_mean[chan]
                charge_std = ff_data.charge_std[chan]
                # median_ped = ped_data.charge_median[chan]
                mean_ped = ped_data.charge_mean[chan]
                ped_std = ped_data.charge_std[chan]

                # select good pixels
                select = np.logical_not(mask[chan])
                fig = plt.figure(chan + 10, figsize=(12, 18))
                fig.tight_layout(rect=[0, 0.0, 1, 0.95])

                fig.suptitle(f"Run {run} channel: {channel[chan]}", fontsize=25)

                # charge
                plt.subplot(321)
                plt.tight_layout()
                median = int(np.median(charge_mean[select]))
                rms = np.std(charge_mean[select])
                label = f"Median {median:3.2f}, std {rms:5.0f}"
                plt.xlabel("charge (ADC)", fontsize=20)
                plt.ylabel("pixels", fontsize=20)
                plt.hist(charge_mean[select], bins=50, label=label)
                plt.legend()

                plt.subplot(322)
                plt.tight_layout()
                plt.ylabel("pixels", fontsize=20)
                plt.xlabel("charge std", fontsize=20)
                median = np.median(charge_std[select])
                rms = np.std(charge_std[select])
                label = f"Median {median:3.2f}, std {rms:3.2f}"
                plt.hist(charge_std[select], bins=50, label=label)
                plt.legend()

                # pedestal charge
                plt.subplot(323)
                plt.tight_layout()
                plt.ylabel("pixels", fontsize=20)
                plt.xlabel("pedestal", fontsize=20)
                median = np.median(mean_ped[select])
                rms = np.std(mean_ped[select])
                label = f"Median {median:3.2f}, std {rms:3.2f}"
                plt.hist(mean_ped[select], bins=50, label=label)
                plt.legend()

                # pedestal std
                plt.subplot(324)
                plt.ylabel("pixels", fontsize=20)
                plt.xlabel("pedestal std", fontsize=20)
                median = np.median(ped_std[select])
                rms = np.std(ped_std[select])
                label = f"Median {median:3.2f}, std {rms:3.2f}"
                plt.hist(ped_std[select], bins=50, label=label)
                plt.legend()

                # relative gain
                plt.subplot(325)
                plt.tight_layout()
                plt.ylabel("pixels", fontsize=20)
                plt.xlabel("relative signal", fontsize=20)
                median = np.median(gain_median[select])
                rms = np.std(gain_median[select])
                label = f"Relative gain {median:3.2f}, std {rms:5.2f}"
                plt.hist(gain_median[select], bins=50, label=label)
                plt.legend()

                # photon electrons
                plt.subplot(326)
                plt.tight_layout()
                plt.ylabel("pixels", fontsize=20)
                plt.xlabel("pe", fontsize=20)
                median = np.median(n_pe[select])
                rms = np.std(n_pe[select])
                label = f"Median {median:3.2f}, std {rms:3.2f}"
                plt.hist(n_pe[select], bins=50, label=label)
                plt.legend()
                plt.subplots_adjust(top=0.92)

                pdf.savefig(plt.gcf())
                plt.close()
