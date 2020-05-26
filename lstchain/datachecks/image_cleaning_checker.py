import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import tables
from ctapipe.image.extractor import ImageExtractor
from ctapipe.instrument import CameraGeometry
from ctapipe.visualization import CameraDisplay
from ctapipe.image.cleaning import tailcuts_clean
from ctapipe_io_lst import LSTEventSource
from traitlets.config import Config
from lstchain.calib.camera.r0 import LSTR0Corrections
from lstchain.calib.camera.calibrator import LSTCameraCalibrator
from lstchain.io.config import read_configuration_file
from lstchain.io.lstcontainers import LSTEventType


def check_standard_cleaning(config_file_path, list_of_data_file,
                            max_events=60000, tel_id=1):
    """
    Function to check cleaning using tailcuts_clean
    """
    config_dict = read_configuration_file(config_file_path)
    config = Config(config_dict)
    print(config_dict)
    calib_file = config_dict["calib_file"]
    time_calib_file = config_dict["time_calib_file"]
    cleaning_parameters = config_dict["tailcut"]

    r1_dl1_calibrator = LSTCameraCalibrator(
                                    calibration_path=calib_file,
                                    time_calibration_path=time_calib_file,
                                    extractor_product=config_dict['charge_product'],
                                    config=config,
                                    gain_threshold = Config(config).gain_selector_config['threshold'],
                                    allowed_tels=[tel_id]
                                            )

    signal_place_after_clean = np.zeros(1855)
    sum_ped_ev = 0
    survived_ped_ev = 0


    for input_file in list_of_data_file:
        print(input_file)
        r0_r1_calibrator = LSTR0Corrections(pedestal_path=None,
                                            r1_sample_start=3,
                                            r1_sample_end=39)
        reader = LSTEventSource(input_url=input_file,
                                max_events=max_events)

        for i, ev in enumerate(reader):
            r0_r1_calibrator.calibrate(ev)
            if LSTEventType.is_pedestal(ev.r1.tel[tel_id].trigger_type):
                sum_ped_ev += 1
                r1_dl1_calibrator(ev)
                img = ev.dl1.tel[tel_id].image
                geom = ev.inst.subarray.tel[tel_id].camera
                clean = tailcuts_clean(
                                    geom,
                                    img,
                                    **cleaning_parameters
                                    )
                cleaned = img.copy()
                cleaned[~clean] = 0.0
                signal_place_after_clean[np.where(clean == True)] += 1
                if np.sum(cleaned>0) > 0:
                    survived_ped_ev += 1

    print("{}/{}".format(survived_ped_ev, sum_ped_ev))
    image_survied_pixels = signal_place_after_clean/sum_ped_ev
    run_id = input_file.split("/")[-1][8:21]
    plot_survived_pedestal_image(image_survied_pixels, run_id, survived_ped_ev, sum_ped_ev, [])

def plot_survived_pedestal_image(image, run_id, survived_ped_ev, sum_ped_ev, noise_pixels_id_list):
    fig, ax = plt.subplots(figsize=(10, 8))
    geom = CameraGeometry.from_name('LSTCam-003')
    disp0 = CameraDisplay(geom, ax=ax)
    disp0.image = image
    disp0.highlight_pixels(noise_pixels_id_list, linewidth=3)
    disp0.add_colorbar(ax=ax, label="N times signal remain after cleaning [%]")
    disp0.cmap = 'gnuplot2'
    ax.set_title("{} \n {}/{}".format(run_id, survived_ped_ev, sum_ped_ev), fontsize=25)
    ax.set_xlabel(" ")
    ax.set_ylabel(" ")
    plt.tight_layout()
    plt.show()
