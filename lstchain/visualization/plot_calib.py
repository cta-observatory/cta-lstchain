import numpy as np
from astropy import units as u
import logging
import sys
from ctapipe.coordinates import EngineeringCameraFrame
from ctapipe.visualization import CameraDisplay
from ctapipe_io_lst import load_camera_geometry
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

log = logging.getLogger(__name__)

__all__ = [
    "plot_calibration_results",
]

channel = ["HG", "LG"]

plot_dir = "none"


def plot_calibration_results(ped_data, ff_data, calib_data, run=0, plot_file=None, calib_type="Cat-A"):
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
    
    if calib_type == "Cat-A":
        charge_unit = "[ADC]"
        gain_unit = "[ADC/pe]"
    elif calib_type == "Cat-B":
        charge_unit = "[pe]" 
        gain_unit = "[cat-B/cat-A]" 
    else:  
        log.critical(f'Wrong calib_type: {calib_type}. It must be Cat-A or Cat-B')
        sys.exit(1)

    # read geometry
    camera = load_camera_geometry()
    camera = camera.transform_to(EngineeringCameraFrame())

    # plot open pdf
    if plot_file is not None:
        with PdfPages(plot_file) as pdf:

            plt.rc("font", size=15)

            # first figure
            fig = plt.figure(figsize=(12, 24))
            plt.tight_layout()
            fig.suptitle(f"Run {run}, {calib_type} calibration", fontsize=25)
            pad = 420
            image = ff_data.charge_median
            mask = ff_data.charge_median_outliers

            for chan in np.arange(2):
                pad += 1
                plt.subplot(pad)
                plt.tight_layout()
                select = np.logical_not(mask[chan])
                disp = CameraDisplay(camera)
                mymin = np.median(image[chan][select]) - 2 * np.std(image[chan][select])
                mymax = np.median(image[chan][select]) + 2 * np.std(image[chan][select])            
                disp.highlight_pixels(mask[chan], linewidth=2)
                disp.image = image[chan]
                disp.cmap = plt.cm.coolwarm
                disp.set_limits_minmax(mymin, mymax)
                
                plt.title(f"{channel[chan]} signal charge {charge_unit}")
                disp.add_colorbar()

            image = ff_data.charge_std
            mask = ff_data.charge_std_outliers
            for chan in np.arange(2):
                pad += 1
                plt.subplot(pad)
                plt.tight_layout()
                select = np.logical_not(mask[chan])
                disp = CameraDisplay(camera)
                mymin = np.median(image[chan][select]) - 2 * np.std(image[chan][select])
                mymax = np.median(image[chan][select]) + 2 * np.std(image[chan][select])
                disp.highlight_pixels(mask[chan], linewidth=2)
                disp.image = image[chan]
                disp.set_limits_minmax(mymin, mymax)
                disp.cmap = plt.cm.coolwarm
                # disp.axes.text(lposx, 0, f'{channel[chan]} signal std [ADC]', rotation=90)
                plt.title(f"{channel[chan]} signal std {charge_unit}")
                disp.add_colorbar()

            image = ped_data.charge_median
            mask = ped_data.charge_median_outliers
            for chan in np.arange(2):
                pad += 1
                plt.subplot(pad)
                plt.tight_layout()
                select = np.logical_not(mask[chan])
                disp = CameraDisplay(camera)
                mymin = np.median(image[chan][select]) - 2 * np.std(image[chan][select])
                mymax = np.median(image[chan][select]) + 2 * np.std(image[chan][select])
                disp.highlight_pixels(mask[chan], linewidth=2)
                disp.image = image[chan]
                disp.set_limits_minmax(mymin, mymax)
                disp.cmap = plt.cm.coolwarm
                # disp.axes.text(lposx, 0, f'{channel[chan]} pedestal [ADC]', rotation=90)
                plt.title(f"{channel[chan]} pedestal {charge_unit}")
                disp.add_colorbar()

            image = ped_data.charge_std
            mask = ped_data.charge_std_outliers
            for chan in np.arange(2):
                pad += 1
                plt.subplot(pad)
                plt.tight_layout()
                select = np.logical_not(mask[chan])
                disp = CameraDisplay(camera)
                mymin = np.median(image[chan][select]) - 2 * np.std(image[chan][select])
                mymax = np.median(image[chan][select]) + 2 * np.std(image[chan][select])
                
                disp.highlight_pixels(mask[chan], linewidth=2)
                disp.image = image[chan]
                disp.set_limits_minmax(mymin, mymax)
                disp.cmap = plt.cm.coolwarm
                # disp.axes.text(lposx, 0, f'{channel[chan]} pedestal std [ADC]', rotation=90)
                plt.title(f"{channel[chan]} pedestal std {charge_unit}")
                disp.add_colorbar()

            plt.subplots_adjust(top=0.92)

            pdf.savefig()
            plt.close()

            # second figure
            fig = plt.figure(figsize=(12, 24))
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
                disp.image = image[chan].to_value(u.ns) 
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
                disp.image = image[chan]
                disp.set_limits_minmax(mymin, mymax)
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
                plt.title(f"{channel[chan]} pe, {np.sum(mask[chan])} unusable pixels")
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
            plt.xlabel("pe", fontsize=20)
            plt.ylabel("pixels", fontsize=20)

            # pe scatter plot
            pad += 1
            plt.subplot(pad)
            plt.tight_layout()
            HG = calib_data.n_pe[0]
            LG = calib_data.n_pe[1]
            HG = np.ma.array(np.where(np.isnan(HG), 0, HG),mask=mask[chan])
            LG = np.ma.array(np.where(np.isnan(LG), 0, LG),mask=mask[chan])
            
            mymin = np.ma.median(LG) - 2 * np.ma.std(LG)
            mymax = np.ma.median(LG) + 2 * np.ma.std(LG)
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
                dc_to_pe =  calib_data.dc_to_pe[chan]
                gain_median = ff_data.relative_gain_median[chan]
                charge_median = ff_data.charge_median[chan]
                #charge_mean = ff_data.charge_mean[chan]
                charge_std = ff_data.charge_std[chan]
                n_ff = ff_data.n_events
                median_ped = ped_data.charge_median[chan]
                #mean_ped = ped_data.charge_mean[chan]
                ped_std = ped_data.charge_std[chan]
                n_ped = ped_data.n_events

                dc_to_pe = calib_data.dc_to_pe[chan]
                time_correction = calib_data.time_correction[chan]

                # select good pixels
                select = np.logical_not(mask[chan])
                fig = plt.figure(figsize=(12, 24))
                fig.tight_layout(rect=[0, 0.0, 1, 0.95])

                fig.suptitle(f"Run {run} channel: {channel[chan]}", fontsize=25)

                # charge
                plt.subplot(421)
                plt.title(f"FF sample of {n_ff} events")
                plt.tight_layout()
                median = int(np.median(charge_median[select]))
                rms = np.std(charge_median[select])
                label = f"Median {median:3.2f}, std {rms:5.0f}"
                plt.xlabel("charge (ADC)", fontsize=20)
                plt.ylabel("pixels", fontsize=20)
                plt.hist(charge_median[select], bins=50, label=label)
                plt.legend()

                plt.subplot(422)
                plt.tight_layout()
                plt.ylabel("pixels", fontsize=20)
                plt.xlabel("charge std", fontsize=20)
                median = np.median(charge_std[select])
                rms = np.std(charge_std[select])
                label = f"Median {median:3.2f}, std {rms:3.2f}"
                plt.hist(charge_std[select], bins=50, label=label)
                plt.legend()

                # pedestal charge
                plt.subplot(423)
                plt.tight_layout()
                plt.title(f"pedestal sample of {n_ped} events")
                plt.ylabel("pixels", fontsize=20)
                plt.xlabel(f"pedestal {charge_unit}", fontsize=20)
                median = np.median(median_ped[select])
                rms = np.std(median_ped[select])
                label = f"Median {median:3.2f}, std {rms:3.2f}"
                plt.hist(median_ped[select], bins=50, label=label)
                plt.legend()

                # pedestal std
                plt.subplot(424)
                plt.ylabel("pixels", fontsize=20)
                plt.xlabel(f"pedestal std {charge_unit}", fontsize=20)
                median = np.median(ped_std[select])
                rms = np.std(ped_std[select])
                label = f"Median {median:3.2f}, std {rms:3.2f}"
                plt.hist(ped_std[select], bins=50, label=label)
                plt.legend()

                # relative gain
                plt.subplot(425)
                plt.tight_layout()
                plt.ylabel("pixels", fontsize=20)
                plt.xlabel("relative signal", fontsize=20)
                median = np.median(gain_median[select])
                rms = np.std(gain_median[select])
                label = f"Relative gain {median:3.2f}, std {rms:5.2f}"
                plt.hist(gain_median[select], bins=50, label=label)
                plt.legend()

                # photon electrons
                plt.subplot(426)
                plt.tight_layout()
                plt.ylabel("pixels", fontsize=20)
                plt.xlabel("time corrections [ns]", fontsize=20)
                median = np.median(time_correction[select])
                rms = np.std(time_correction[select])
                label = f"Median {median:3.2f}, std {rms:3.2f}"
                plt.hist(time_correction[select].value, bins=50, label=label)
                plt.legend()
                plt.subplots_adjust(top=0.92)
                """
                # photon electrons
                plt.subplot(427)
                plt.tight_layout()
                plt.ylabel("pixels", fontsize=20)
                plt.xlabel("pe", fontsize=20)
                median = np.median(n_pe[select])
                rms = np.std(n_pe[select])
                label = f"Median {median:3.2f}, std {rms:3.2f}"
                plt.hist(n_pe[select], bins=50, label=label)
                plt.legend()
                plt.subplots_adjust(top=0.92)
                """
               
                # gain on camera
                plt.subplot(427)
                denominator = dc_to_pe
                numerator = 1.

                gain = np.divide(numerator, denominator, out=np.zeros_like(denominator), where=denominator != 0)
                median = np.median(gain[select])
                std = np.std(gain[select])
                                
                plt.tight_layout()
                disp = CameraDisplay(camera)
                disp.highlight_pixels(mask[chan], linewidth=2)
                mymin = median - 2 * std
                mymax = median + 2 * std
                disp.set_limits_minmax(mymin, mymax)
                disp.image = gain
                disp.cmap = plt.cm.coolwarm
                
                plt.title(f"flat-fielded gain {gain_unit}")             
                disp.add_colorbar()
                plt.subplots_adjust(top=0.92)
                # gain
                plt.subplot(428)
                plt.tight_layout()
                plt.ylabel("pixels", fontsize=20)
                plt.xlabel(f"flat-fielded gain {gain_unit}", fontsize=20)
                median = np.median(gain)
                rms = np.std(gain)
                label = f"Median {median:3.2f}, std {rms:3.2f}"
                plt.hist(gain[select], bins=50, label=label)
                plt.legend()
                plt.subplots_adjust(top=0.92)

                pdf.savefig(plt.gcf())
                plt.close()
