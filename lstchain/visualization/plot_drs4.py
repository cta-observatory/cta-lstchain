from matplotlib import pyplot as plt

import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from ctapipe.io import EventSource
from ctapipe.coordinates import EngineeringCameraFrame

from traitlets.config import Config

from lstchain.calib.camera.pedestals import PedestalIntegrator
from ctapipe.visualization import CameraDisplay
import logging


log = logging.getLogger(__name__)

log.setLevel(logging.INFO)
handler = logging.StreamHandler()
logging.getLogger().addHandler(handler)


__all__ = ["plot_pedestals"]

channel = ["HG", "LG"]


def plot_pedestals(data_file, pedestal_file, run=0, plot_file=None, tel_id=1, offset_value=400, sample_size=1000):
    """
    plot pedestal quantities quantities

    Parameters
    ----------
    data_file:   pedestal run

    pedestal_file:   file with drs4 corrections

    run: run number of data to be corrected

    plot_file:  name of output pdf file

    tel_id: id of the telescope

    offset_value: baseline off_set
    """

    config = {
        "LSTEventSource": {
            "allowed_tels": [1],
            "LSTR0Corrections": {
                "drs4_pedestal_path": pedestal_file,
            },
        }
    }
    # event_reader
    reader = EventSource(data_file, config=Config(config), max_events=None)
    t = np.linspace(2, 37, 36)

    # configuration for the charge integrator
    charge_config = Config(
        {
            "FixedWindowSum": {
                "window_shift": 6,
                "window_width": 12,
                "peak_index": 18,
            }
        }
    )
    # declare the pedestal component
    pedestal = PedestalIntegrator(
        tel_id=tel_id,
        time_sampling_correction_path=None,
        sample_size=sample_size,
        sample_duration=1000000,
        charge_median_cut_outliers=[-10, 10],
        charge_std_cut_outliers=[-10, 10],
        charge_product="FixedWindowSum",
        config=charge_config,
        subarray=reader.subarray,
    )

    for i, event in enumerate(reader):
        if tel_id != event.trigger.tels_with_trigger[0]:
            raise Exception(
                f"Given wrong telescope id {tel_id}, files has id {event.trigger.tels_with_trigger[0]}"
            )

        are_pedestals_calculated = pedestal.calculate_pedestals(event)
        if are_pedestals_calculated:
            ped_data = event.mon.tel[tel_id].pedestal
            break

    camera_geometry = reader.subarray.tels[tel_id].camera.geometry
    camera_geometry = camera_geometry.transform_to(EngineeringCameraFrame())

    if are_pedestals_calculated and plot_file is not None:
        with PdfPages(plot_file) as pdf:

            plt.rc("font", size=15)

            # first figure
            fig = plt.figure(1, figsize=(12, 24))
            plt.tight_layout()
            n_samples = charge_config["FixedWindowSum"]["window_width"]
            fig.suptitle(f"Run {run}, integration on {n_samples} samples", fontsize=25)
            pad = 420

            image = ped_data.charge_median
            mask = ped_data.charge_median_outliers
            for chan in np.arange(2):
                pad += 1
                plt.subplot(pad)
                plt.tight_layout()
                disp = CameraDisplay(camera_geometry)
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
                disp = CameraDisplay(camera_geometry)
                mymin = np.median(image[chan]) - 2 * np.std(image[chan])
                mymax = np.median(image[chan]) + 2 * np.std(image[chan])
                disp.set_limits_minmax(mymin, mymax)
                disp.highlight_pixels(mask[chan], linewidth=2)
                disp.image = image[chan]
                disp.cmap = plt.cm.coolwarm
                # disp.axes.text(lposx, 0, f'{channel[chan]} pedestal std [ADC]', rotation=90)
                plt.title(f"{channel[chan]} pedestal std [ADC]")
                disp.add_colorbar()

            #  histograms
            for chan in np.arange(2):
                mean_ped = ped_data.charge_mean[chan]
                ped_std = ped_data.charge_std[chan]

                # select good pixels
                select = np.logical_not(mask[chan])

                # fig.suptitle(f"Run {run} channel: {channel[chan]}", fontsize=25)
                pad += 1
                # pedestal charge
                plt.subplot(pad)
                plt.tight_layout()
                plt.ylabel("pixels")
                plt.xlabel(f"{channel[chan]} pedestal")
                median = np.median(mean_ped[select])
                rms = np.std(mean_ped[select])
                label = f"{channel[chan]} Median {median:3.2f}, std {rms:3.2f}"
                plt.hist(mean_ped[select], bins=50, label=label)
                plt.legend()
                pad += 1
                # pedestal std
                plt.subplot(pad)
                plt.ylabel("pixels")
                plt.xlabel(f"{channel[chan]} pedestal std")
                median = np.median(ped_std[select])
                rms = np.std(ped_std[select])
                label = f" Median {median:3.2f}, std {rms:3.2f}"
                plt.hist(ped_std[select], bins=50, label=label)
                plt.legend()

            plt.subplots_adjust(top=0.94, bottom=0.04, right=0.96)

            pdf.savefig()
            plt.close()

            # event_reader
            # reader = EventSource(data_file, config=Config(config), max_events=1000)

            pix = 0
            pad = 420
            offset_value = reader.r0_r1_calibrator.offset.tel[tel_id]

            # plot corrected waveforms of first 8 events
            for i, ev in enumerate(reader):
                for chan in np.arange(2):

                    if pad == 420:
                        # new figure

                        fig = plt.figure(ev.index.event_id * 1000, figsize=(12, 24))
                        fig.suptitle(f"Run {run}, pixel {pix}", fontsize=25)
                        plt.tight_layout()
                    pad += 1
                    plt.subplot(pad)

                    # remove samples at beginning / end of waveform
                    start = reader.r0_r1_calibrator.r1_sample_start.tel[tel_id]
                    end = reader.r0_r1_calibrator.r1_sample_end.tel[tel_id]

                    plt.subplots_adjust(top=0.92)
                    label = f"event {ev.index.event_id}, {channel[chan]}: R0"
                    plt.step(
                        t,
                        ev.r0.tel[tel_id].waveform[chan, pix, start:end],
                        color="blue",
                        label=label,
                    )

                    label = "baseline correction \n + dt corr + interp. spikes"

                    plt.step(
                        t,
                        ev.r1.tel[tel_id].waveform[chan, pix] + offset_value,
                        alpha=0.5,
                        color="green",
                        label=label,
                    )
                    plt.plot([0, 40], [offset_value, offset_value], "k--", label="offset")
                    plt.xlabel("time sample [ns]")
                    plt.ylabel("counts [ADC]")
                    plt.legend()
                    plt.ylim(200, 600)

                if pad == 428:
                    pad = 420
                    plt.subplots_adjust(top=0.92)
                    pdf.savefig()
                    plt.close()

                if i == 8:
                    break

    elif not are_pedestals_calculated:
        log.error("Not able to calculate pedestals or output pdf file not especified.")

    elif plot_file is None:
        log.warning("Not PDF outputfile specified.")
