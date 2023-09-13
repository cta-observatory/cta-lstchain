#!/usr/bin/env python3

"""
Pipeline to test train three Random Forests destinated to Energy, disp
reconstruction and Gamma/Hadron separation and test the performance
of Random Forests.

Inputs are DL1 files
Outputs are the RF trained models

Usage:

$>python lstchain_mc_rfperformance.py

"""

import argparse
import logging
import sys

import matplotlib.pyplot as plt
import pandas as pd
from ctapipe.instrument import SubarrayDescription
from ctapipe_io_lst import OPTICS

from lstchain.io import (
    read_configuration_file,
    replace_config,
    standard_config,
)
from lstchain.io.io import dl1_params_lstcam_key
from lstchain.reco import dl1_to_dl2
from lstchain.reco.utils import filter_events
from lstchain.visualization import plot_dl2

log = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description="Train and Apply Random Forests.")

# Required argument
parser.add_argument('--input-file-gamma-train', '--g-train', type=str,
                    dest='gammafile',
                    help='path to the dl1 file of gamma events for training')

parser.add_argument('--input-file-proton-train', '--p-train', type=str,
                    dest='protonfile',
                    help='path to the dl1 file of proton events for training')

parser.add_argument('--input-file-gamma-test', '--g-test', type=str,
                    dest='gammatest',
                    help='path to the dl1 file of gamma events for test')

parser.add_argument('--input-file-proton-test', '--p-test', type=str,
                    dest='protontest',
                    help='path to the dl1 file of proton events for test')

# Optional arguments
parser.add_argument(
    '--no-save-models',
    dest='save_models',
    action='store_false',
    help='Disable storing trained models',
)

parser.add_argument('--batch', '-b', action='store', type=bool,
                    dest='batch',
                    help='Boolean. True for running it without plotting output',
                    default=True)

parser.add_argument('--output_dir', '-o', action='store', type=str,
                    dest='path_models',
                    help='Path to store the resulting RF',
                    default='./saved_models/')

parser.add_argument('--config', '-c', action='store', type=str,
                    dest='config_file',
                    help='Path to a configuration file. If none is given, a standard configuration is applied',
                    default=None
                    )


def main():
    args = parser.parse_args()

    custom_config = {}
    if args.config_file is not None:
        custom_config = read_configuration_file(args.config_file)

    config = replace_config(standard_config, custom_config)

    try:
        subarray_info = SubarrayDescription.from_hdf(args.gammatest)
        tel_id = config["allowed_tels"][0] if "allowed_tels" in config else 1
        effective_focal_length = subarray_info.tel[tel_id].optics.equivalent_focal_length
    except OSError:
        print("subarray table is not readable because of the version incompatibility.")
        print("The effective focal length for the standard LST optics will be used.")
        effective_focal_length = OPTICS.effective_focal_length
        
    reg_energy, reg_disp_norm, cls_disp_sign, cls_gh = dl1_to_dl2.build_models(
        args.gammafile,
        args.protonfile,
        save_models=args.save_models,
        path_models=args.path_models,
        free_model_memory=False,
        custom_config=config,
    )

    gammas = filter_events(pd.read_hdf(args.gammatest, key=dl1_params_lstcam_key),
                           config["events_filters"],
                           )
    proton = filter_events(pd.read_hdf(args.protontest, key=dl1_params_lstcam_key),
                           config["events_filters"],
                           )

    data = pd.concat([gammas, proton], ignore_index=True)

    dl2 = dl1_to_dl2.apply_models(data, cls_gh, reg_energy, reg_disp_norm=reg_disp_norm,
                                  cls_disp_sign=cls_disp_sign, effective_focal_length=effective_focal_length,
                                  custom_config=config)

    ####PLOT SOME RESULTS#####

    selected_gammas = dl2.query('reco_type==0 & mc_type==0')

    if (len(selected_gammas) == 0):
        log.warning('No gammas selected, I will not plot any output')
        sys.exit()

    plot_dl2.plot_features(dl2)
    if not args.batch:
        plt.show()

    plot_dl2.energy_results(selected_gammas)
    if not args.batch:
        plt.show()

    plot_dl2.direction_results(selected_gammas)
    if not args.batch:
        plt.show()

    plot_dl2.plot_disp_vector(selected_gammas)
    if not args.batch:
        plt.show()

    plot_dl2.plot_pos(dl2)
    if not args.batch:
        plt.show()

    plot_dl2.plot_roc_gamma(dl2)
    if not args.batch:
        plt.show()

    if args.save_models:
        plot_dl2.plot_models_features_importances(args.path_models, args.config_file)
        if not args.batch:
            plt.show()

    plt.hist(dl2[dl2['mc_type'] == 101]['gammaness'], bins=100)
    plt.hist(dl2[dl2['mc_type'] == 0]['gammaness'], bins=100)
    if not args.batch:
        plt.show()


if __name__ == '__main__':
    main()
