
from matplotlib import pyplot as plt
from traitlets.config.loader import Config
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from ctapipe.io import event_source
from lstchain.calib.camera.r0 import LSTR0Corrections
from ctapipe.instrument import  CameraGeometry


from ctapipe.calib.camera.pedestals import PedestalIntegrator
from ctapipe.visualization import CameraDisplay

__all__ = ['plot_pedestals',
           ]

channel = ['HG', 'LG']


def plot_pedestals(data_file, pedestal_file, run=0 , plot_file="none", tel_id=1, offset_value=400):
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

    # plot open pdf
    if plot_file != "none":
        pp = PdfPages(plot_file)

    plt.rc('font', size=15)

    # r0 calibrator
    r0_calib = LSTR0Corrections(pedestal_path=pedestal_file, offset=offset_value,
                                tel_id=tel_id )

    # event_reader
    reader = event_source(data_file, max_events=1000)
    t = np.linspace(2, 37, 36)

    # configuration for the charge integrator
    charge_config = Config({
        "FixedWindowSum": {
            "window_start": 12,
            "window_width": 12,
        }

    })
    # declare the pedestal component
    pedestal = PedestalIntegrator(tel_id=tel_id,
                                  sample_size=1000,
                                  sample_duration=1000000,
                                  charge_median_cut_outliers=[-10, 10],
                                  charge_std_cut_outliers=[-10, 10],
                                  charge_product="FixedWindowSum",
                                  config=charge_config)

    for i, event in enumerate(reader):
        if tel_id != event.r0.tels_with_data[0]:
            raise Exception(f"Given wrong telescope id {tel_id}, files has id {event.r0.tels_with_data[0]}")

        # move from R0 to R1
        r0_calib.calibrate(event)

        ok = pedestal.calculate_pedestals(event)
        if ok:
            ped_data = event.mon.tel[tel_id].pedestal
            break

    camera = CameraGeometry.from_name("LSTCam", 2)

    # plot open pdf
    if plot_file != "none":
        pp = PdfPages(plot_file)

    plt.rc('font', size=15)

    ### first figure
    fig = plt.figure(1, figsize=(12, 24))
    plt.tight_layout()
    n_samples = charge_config["FixedWindowSum"]['window_width']
    fig.suptitle(f"Run {run}, integration on {n_samples} samples", fontsize=25)
    pad = 420

    image = ped_data.charge_median
    mask = ped_data.charge_median_outliers
    for chan in (np.arange(2)):
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
        plt.title(f'{channel[chan]} pedestal [ADC]')
        disp.add_colorbar()

    image = ped_data.charge_std
    mask = ped_data.charge_std_outliers
    for chan in (np.arange(2)):
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
        plt.title(f'{channel[chan]} pedestal std [ADC]')
        disp.add_colorbar()

    ###  histograms
    for chan in np.arange(2):
        mean_ped = ped_data.charge_mean[chan]
        ped_std = ped_data.charge_std[chan]

        # select good pixels
        select = np.logical_not(mask[chan])

        #fig.suptitle(f"Run {run} channel: {channel[chan]}", fontsize=25)
        pad += 1
        # pedestal charge
        plt.subplot(pad)
        plt.tight_layout()
        plt.ylabel('pixels')
        plt.xlabel(f'{channel[chan]} pedestal')
        median = np.median(mean_ped[select])
        rms = np.std(mean_ped[select])
        label = f"{channel[chan]} Median {median:3.2f}, std {rms:3.2f}"
        plt.hist(mean_ped[select], bins=50, label=label)
        plt.legend()
        pad += 1
        # pedestal std
        plt.subplot(pad)
        plt.ylabel('pixels')
        plt.xlabel(f'{channel[chan]} pedestal std')
        median = np.median(ped_std[select])
        rms = np.std(ped_std[select])
        label = f" Median {median:3.2f}, std {rms:3.2f}"
        plt.hist(ped_std[select], bins=50, label=label)
        plt.legend()

    plt.subplots_adjust(top=0.94)
    if plot_file != "none":

        pp.savefig()

    pix = 0
    pad = 420
    # plot corrected waveforms of first 8 events
    for i, ev in enumerate(reader):
        for chan in np.arange(2):

            if pad == 420:
                # new figure

                fig = plt.figure(ev.r0.event_id, figsize=(12, 24))
                fig.suptitle(f"Run {run}, pixel {pix}", fontsize=25)
                plt.tight_layout()
            pad += 1
            plt.subplot(pad)

            plt.subplots_adjust(top=0.92)
            label = f"event {ev.r0.event_id}, {channel[chan]}: R0"
            plt.step(t, ev.r0.tel[tel_id].waveform[chan, pix, 2:38], color="blue", label=label)

            r0_calib.subtract_pedestal(ev,tel_id)
            label = "+ pedestal substraction"
            plt.step(t, ev.r1.tel[tel_id].waveform[chan, pix, 2:38], color="red", alpha=0.5,  label=label)

            r0_calib.time_lapse_corr(ev,tel_id)
            r0_calib.interpolate_spikes(ev,tel_id)
            label = "+ dt corr + interp. spikes"
            plt.step(t, ev.r1.tel[tel_id].waveform[chan, pix, 2:38],  alpha=0.5, color="green",label=label)
            plt.plot([0, 40], [offset_value, offset_value], 'k--',  label="offset")
            plt.xlabel("time sample [ns]")
            plt.ylabel("counts [ADC]")
            plt.legend()
            plt.ylim([-50, 500])

        if plot_file != "none" and pad == 428:
            pad = 420
            plt.subplots_adjust(top=0.92)
            pp.savefig()

        if i == 8:
            break

    if plot_file != "none":
        pp.close()

