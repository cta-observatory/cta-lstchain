
from matplotlib import pyplot as plt

import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from ctapipe.io import event_source
from lstchain.calib.camera.r0 import LSTR0Corrections


__all__ = ['plot_drs4',
           ]

channel = ['HG', 'LG']


def plot_waveforms(data_file, pedestal_file, run= 0 , plot_file="none"):
    """
     plot camera calibration quantities

     Parameters
     ----------
     data_file:   pedestal run

     pedestal_file:   file with drs4 corrections

     run: run number of data to be corrected

     plot_file:  name of output pdf file
     """

    # plot open pdf
    if plot_file != "none":
        pp = PdfPages(plot_file)

    plt.rc('font', size=15)
    offset_value=400
    tel_id = 1

    # r0 calibrator
    r0_calib = LSTR0Corrections(pedestal_path=pedestal_file, offset=offset_value,
                                r1_sample_start=2, r1_sample_end=38, tel_id=tel_id )

    # event_reader
    reader = event_source(data_file, max_events=8)

    pix = 0
    pad = 420

    t = np.linspace(2, 37, 36)
    tel_id = 1
    for ev in reader:

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
            pp.savefig()

    if plot_file != "none":
        pp.close()