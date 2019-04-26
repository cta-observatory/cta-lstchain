"""Pipeline for test the performance of Random Forests.
4
Usage:

$>python lst-rfperformance.py

"""

import numpy as np
import pandas as pd
import argparse
import matplotlib.pyplot as plt
from sklearn.externals import joblib
from lstchain.reco import dl1_to_dl2
from lstchain.visualization import plot_dl2
from lstchain.reco import utils
from sklearn.model_selection import train_test_split
import os
import ctaplot
import astropy.units as u

parser = argparse.ArgumentParser(description="Train Random Forests.")

# Required argument
parser.add_argument('--gammafile', '-fg', type=str,
                    dest='gammafile',
                    help='path to the dl1 file of gamma events for training')

parser.add_argument('--protonfile', '-fp', type=str,
                    dest='protonfile',
                    help='path to the dl1 file of proton events for training')

parser.add_argument('--gammatest', '-gt', type=str,
                    dest='gammatest',
                    help='path to the dl1 file of gamma events for test')

parser.add_argument('--protontest', '-pt', type=str,
                    dest='protontest',
                    help='path to the dl1 file of proton events for test')

parser.add_argument('--storerf', '-s', action='store', type=bool,
                    dest='storerf',
                    help='Boolean. True for storing trained RF in 3 files'
                    'Deafult=False, any user input will be considered True',
                    default=True)

# Optional arguments
parser.add_argument('--opath', '-o', action='store', type=str,
                     dest='path_models',
                     help='Path to store the resulting RF',
                     default='./saved_models/')

parser.add_argument('--config_file', '-conf', action='store', type=str,
                    dest='config_file',
                    help='Path to a configuration file. If none is given, a standard configuration is applied',
                    default=None
                    )


args = parser.parse_args()


if __name__ == '__main__':
    #Train the models
    features = ['intensity', 'width', 'length', 'x', 'y', 'psi', 'phi', 'wl',
                'skewness', 'kurtosis','r', 'time_gradient', 'intercept',
                'leakage', 'n_islands' ]

    intensity_min = np.log10(200)
    leakage_cut = 0.2
    r_min = 0.15

    reg_energy, reg_disp_vector, cls_gh = dl1_to_dl2.build_models(
        args.gammafile,
        args.protonfile,
        features,
        intensity_min = intensity_min,
        leakage_cut = leakage_cut,
        save_models=args.storerf,
        path_models=args.path_models,
        config_file=args.config_file
    )

    gammas = dl1_to_dl2.filter_events(pd.read_hdf(args.gammatest, key='events/LSTCam'),
                                      leakage_cut = leakage_cut,
                                      intensity_min=intensity_min,
                                      r_min=r_min)
    proton = dl1_to_dl2.filter_events(pd.read_hdf(args.protontest, key='events/LSTCam'),
                                      leakage_cut = leakage_cut,
                                      intensity_min=intensity_min,
                                      r_min=r_min)

    data = pd.concat([gammas,proton], ignore_index=True)

    dl2 = dl1_to_dl2.apply_models(data, features,
                                  cls_gh, reg_energy, reg_disp_vector)

    ####PLOT SOME RESULTS#####

    gammas = dl2[dl2.gammaness>=0.5]
    protons = dl2[dl2.gammaness<0.5]
    gammas.hadro_rec = 0
    protons.hadro_rec = 1

    focal_length = 28 * u.m
    src_pos_reco = utils.reco_source_position_sky(gammas.x.values * u.m,
                                                  gammas.y.values * u.m,
                                                  gammas.disp_dx_rec.values * u.m,
                                                  gammas.disp_dy_rec.values * u.m,
                                                  focal_length,
                                                  gammas.mc_alt_tel.values * u.rad,
                                                  gammas.mc_az_tel.values * u.rad)


    plot_dl2.plot_features(dl2)
    plt.show()

    plot_dl2.plot_e(gammas, 10, 1.5, 3.5)
    plt.show()

    plot_dl2.calc_resolution(gammas)
    plt.show()

    plot_dl2.plot_e_resolution(gammas, 10, 1.5, 3.5)
    plt.show()

    plot_dl2.plot_disp_vector(gammas)
    plt.show()



    ctaplot.plot_theta2(gammas.mc_alt,
                        np.arctan(np.tan(gammas.mc_az)),
                        src_pos_reco.alt.rad,
                        np.arctan(np.tan(src_pos_reco.az.rad)),
                        bins=50, range=(0, 1),
    )

    plt.show()
    ctaplot.plot_angular_res_per_energy(src_pos_reco.alt.rad,
                                        np.arctan(np.tan(src_pos_reco.az.rad)),
                                        gammas.mc_alt,
                                        np.arctan(np.tan(gammas.mc_az)),
                                        10**(gammas.mc_energy-3)
    )
    plt.show()


    features_ = ['intensity', 'width', 'length', 'x', 'y', 'psi', 'phi', 'wl',
                 'skewness', 'kurtosis','r', 'time_gradient', 'intercept',
                 'leakage', 'n_islands',
                 'e_rec', 'disp_dx_rec', 'disp_dy_rec']


    plt.show()
    plot_dl2.plot_pos(dl2)
    plt.show()
    plot_dl2.plot_ROC(cls_gh, dl2, features_, -1)
    plt.show()
    plot_dl2.plot_importances(cls_gh, features_)
    plt.show()
    plot_dl2.plot_importances(reg_energy, features)
    plt.show()
    plot_dl2.plot_importances(reg_disp_vector, features)
    plt.show()

    plt.hist(dl2[dl2['hadroness']==1]['gammaness'], bins=100)
    plt.hist(dl2[dl2['hadroness']==0]['gammaness'], bins=100)
    plt.show()
