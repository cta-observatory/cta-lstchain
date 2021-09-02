#!/fefs/aswg/workspace/jakub.jurysek/software/anaconda3/envs/lst-dev/bin/python3.7

import os
from matplotlib import pyplot as plt
import numpy as np
import astropy.units as u

from ctapipe.io import EventSource
from ctapipe.visualization import CameraDisplay
from ctapipe.instrument import CameraGeometry
from ctapipe.calib import CameraCalibrator
from ctapipe.containers import EventType

from ctapipe_io_lst import LSTEventSource
from traitlets.config import Config

from optparse import OptionParser
import pandas as pd


def find_flash_event(event, telescope=None, n_pixels=None):

    pixel_values = np.max(event.r0.tel[telescope].waveform[0], axis = 1)
    mask_flash = pixel_values > 800
    n_pixels_on = sum(mask_flash)
    if n_pixels_on > 0.9 * n_pixels:
        flash = True
    else: flash = False
    return flash


if __name__ == "__main__":

    opts_parser = OptionParser()
    opts_parser.add_option("-d", "--display", dest="display", action="store_true", help="Display written data", default=False)
    opts_parser.add_option("-t", "--telid", dest="telescope", help="Telescope ID", default=1, type=int)

    opts_parser.add_option("-o", "--outpath", dest="outpath", help="Output path", default="./", type=str)

    opts_parser.add_option("--data", dest="data", help="input data file", default="", type=str)
    opts_parser.add_option("--pedestal", dest="pedestal", help="pedestal file", default="", type=str)
    opts_parser.add_option("--timing_corr", dest="timing_corr", help="file for timing correction", default="", type=str)
    opts_parser.add_option("--calibration", dest="calibration", help="file for calibration", default="", type=str)
    opts_parser.add_option("--pointing", dest="pointing", help="pointing file", default="", type=str)
    opts_parser.add_option("--max_events", dest="max_events", help="maximum number of pedestal events read", default=10000000, type=int)

    #retrieve args
    (options, args) = opts_parser.parse_args()
    display         = options.display
    tel_id          = options.telescope
    out_path        = options.outpath
    datafile        = options.data
    pedestal        = options.pedestal
    timing_corr     = options.timing_corr
    calibration     = options.calibration
    pointing_file   = options.pointing
    max_events      = options.max_events

    print("Data file", datafile)
    print("Pedestal file:", pedestal)
    print("Timing file:", timing_corr)
    print("Calibration file:", calibration)
    print("Pointing file:", pointing_file)

    config = Config({
        "LSTEventSource": {
            "LSTR0Corrections": {
                "drs4_pedestal_path": pedestal,
                "drs4_time_calibration_path": timing_corr,
                "calibration_path": calibration,
                "select_gain": False
            },
            "PointingSource": {
                "drive_report_path": pointing_file
            }
        }
    })

    source = LSTEventSource(input_url=datafile, config=config)
    subarray = LSTEventSource.create_subarray(geometry_version=3)
    geom = subarray.tel[tel_id].camera.geometry
    n_pixels = subarray.tels[tel_id].camera.geometry.n_pixels

    n_ped_events = 0
    n_flat_events = 0
    mat_all = []
    mask_flash = np.ones(n_pixels)

    for i, event in enumerate(source):

        if n_ped_events < max_events:
            event_id = event.index.event_id
            obs_id = event.index.obs_id

            #print(obs_id, event_id, event.trigger.tel[tel_id].time.unix_tai)

            if event.trigger.event_type == EventType.SKY_PEDESTAL:

                print(obs_id, event_id, "Pedestal event detected!")

                # check if the tagged pedestal event is not a flash event
                is_flash = find_flash_event(event, telescope=tel_id, n_pixels=n_pixels)

                if not is_flash:
                    n_ped_events +=1
                    data = event.r1.tel[tel_id].waveform
                    time = event.trigger.tel[tel_id].time.unix_tai
                    gain = event.r1.tel[tel_id].selected_gain_channel
                    altitude = event.pointing.tel[tel_id].altitude.value
                    azimuth = event.pointing.tel[tel_id].azimuth.value

                    if config["LSTEventSource"]["LSTR0Corrections"]["select_gain"]:
                        px_variance = np.var(data, axis=1)
                        px_mean = np.mean(data, axis=1)
                    else:
                        px_variance = np.var(data, axis=2)
                        px_mean = np.mean(data, axis=2)

                    #print(event)
                    # storing values extracted from all pedestal events
                    mat_all.append([obs_id, event_id, time, azimuth, altitude, mask_flash, px_mean, px_variance, data])
                    #print([obs_id, event_id, time, azimuth, altitude, mask_flash, px_mean, px_variance, data])

                    if display:
                        fig, ax = plt.subplots(figsize=(10,10))
                        disp0 = CameraDisplay(geom, ax=ax)
                        disp0.cmap = "viridis"
                        if config["LSTEventSource"]["LSTR0Corrections"]["select_gain"]:
                            disp0.image = px_variance
                        else:
                            disp0.image = px_variance[0, :]    # show always gain0
                        disp0.add_colorbar(ax=ax)
                        plt.savefig("mean_var_" + str(event_id) + ".png")
                        plt.close()

                        if not config["LSTEventSource"]["LSTR0Corrections"]["select_gain"]:
                            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
                            ax1.plot(event.r0.tel[tel_id].waveform[0, 10, :], label='r0_0')
                            ax1.plot(event.r0.tel[tel_id].waveform[1, 10, :], label='r0_1')
                            ax1.legend()
                            ax2.plot(data[0, 10, :], label='r1_0')
                            ax2.plot(data[1, 10, :], label='r1_1')
                            ax2.legend()
                            plt.savefig("r1_" + str(event_id) + ".png")
                            plt.close()
                else:
                    print(obs_id, event_id, "Is actually just baddly tagged flash event! (It won\"t be saved)")

            elif event.trigger.event_type == EventType.FLATFIELD:

                print(obs_id, event_id, "Flatfield/flash event detected!")
                n_flat_events +=1

                pixel_values = np.max(event.r0.tel[tel_id].waveform[0], axis = 1)
                mask_flash = pixel_values > 800
                n_pixels_on = sum(mask_flash)
                print("N ON pixels", n_pixels_on)

                if display:
                    fig, ax = plt.subplots(figsize=(10,10))
                    disp0 = CameraDisplay(geom, ax=ax)
                    disp0.cmap = "viridis"
                    disp0.image = pixel_values
                    disp0.add_colorbar(ax=ax)
                    plt.savefig("flash_" + str(event_id) + ".png")
                    plt.close()

        else:
            print("Maximum alowed number of pedestal events reached.")
            break

    print("N of pedestal events found:", n_ped_events)
    print("N of flatfield events found:", n_flat_events)

    # saving outputs
    pd_data = pd.DataFrame(mat_all, columns=["obs_id", "event_id", "time", "azimuth", "altitude", "mask_flash", "px_mean", "px_var", "data"])
    print(pd_data['time'].min(), pd_data['time'].max(), pd_data['time'].mean())
    data_out = os.path.join(out_path, datafile.split("/")[-1].split(".fits")[0] + "_ped.pkl")
    pd_data.to_pickle(data_out)
    print("All pedestal events saved to:", data_out)
