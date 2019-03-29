"""Pipeline for test the performance of Random Forests.

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

parser = argparse.ArgumentParser(description="Train Random Forests.")

# Required argument
parser.add_argument('--gammafile', '-fg', type=str,
                    dest='gammafile',
                    help='path to the dl1 file of gamma events for training')

parser.add_argument('--protonfile', '-fp', type=str,
                    dest='protonfile',
                    help='path to the dl1 file of proton events for training')

parser.add_argument('--storerf', '-s', action='store', type=bool,
                    dest='storerf',
                    help='Boolean. True for storing trained RF in 3 files'
                    'Deafult=False, any user input will be considered True',
                    default=False)

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
                'skewness', 'kurtosis','r', 'intercept', 'time_gradient']

    intensity_min = np.log10(100)
    r_max = 0.8
    
    reg_energy, reg_disp_vector, cls_gh = dl1_to_dl2.build_models(
        args.gammafile,
        args.protonfile,
        features,
        intensity_min = intensity_min, 
        r_max = r_max, 
        save_models=args.storerf,
        path_models=args.path_models,
        config_file=args.config_file
    )

    gammas = dl1_to_dl2.filter_events(pd.read_hdf(args.gammafile), 
                                      r_max=r_max, intensity_min=intensity_min)
    proton = dl1_to_dl2.filter_events(pd.read_hdf(args.protonfile), 
                                      r_max=r_max, intensity_min=intensity_min)
    data = pd.concat([gammas,proton], ignore_index=True)

    dl2 = dl1_to_dl2.apply_models(data, features, 
                                  cls_gh, reg_energy, reg_disp_vector)
    

    plot_dl2.plot_features(dl2)
    plt.show()
    plot_dl2.plot_e(dl2)
    plt.show()
    #plot_dl2.plot_disp(dl2)
    #plt.show()
    fig, axes = plt.subplots(1, 2, figsize=(15,6))

    axes[0].hist2d(dl2.disp_dx, dl2.disp_dx_rec, bins=60);
    axes[0].set_xlabel('mc_disp')
    axes[0].set_ylabel('reco_disp')
    axes[0].set_title('disp_dx')
    
    axes[1].hist2d(dl2.disp_dy, dl2.disp_dy_rec, bins=60);
    axes[1].set_xlabel('mc_disp')
    axes[1].set_ylabel('reco_disp')
    axes[1].set_title('disp_dy');
    
    features_ = ['intensity', 'width', 'length', 'x', 'y', 'psi', 'phi', 'wl', 
                 'skewness', 'kurtosis','r', 'intercept', 'time_gradient', 'e_rec',
                 'disp_dx_rec', 'disp_dy_rec']
    plt.show()
    plot_dl2.plot_pos(dl2)
    plt.show()
    plot_dl2.plot_ROC(cls_gh, dl2, features_, -1)
    plt.show()
    
