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

    features = ['intensity',
                'time_gradient',
                'width',
                'length',
                'wl',
                'phi',
                'psi']

    #Split data in train and test events:
    df_gammas = pd.read_hdf(args.gammafile)
    df_proton = pd.read_hdf(args.protonfile)

    train, test = train_test_split(df_gammas, test_size=0.2)
    test = test.append(df_proton,
                       ignore_index=True)

    reg_energy, reg_disp = dl1_to_dl2.train_reco(train, features)
    
    test['e_rec'] = reg_energy.predict(test[features])
    test['disp_rec'] = reg_disp.predict(test[features])

    test['src_x_rec'], test['src_y_rec'] = utils.disp_to_pos(test['disp_rec'],
                                                             test['x'],
                                                             test['y'],
                                                             test['psi'])
    
    features_sep = list(features)
    features_sep.append('e_rec')
    features_sep.append('disp_rec')

    train, test = train_test_split(test, test_size=0.25)
    #Train the Classifier
    
    cls_gh = dl1_to_dl2.train_sep(train,
                                  features_sep)

    test['hadro_rec'] = cls_gh.predict(test[features_sep])
    
    if args.storerf==True:
        os.makedirs(args.path_models, exist_ok=True)
        file_energy = args.path_models + "/reg_energy.sav"
        file_disp = args.path_models + "/reg_disp.sav"
        file_cls_gh = args.path_models + "/cls_gh.sav"
        joblib.dump(reg_energy, file_energy)
        joblib.dump(reg_disp, file_disp)
        joblib.dump(cls_gh, file_cls_gh)

    
    #Plot some results
    e_cuts = [-1, np.log10(500), np.log10(1000)]
    
    for e_cut in e_cuts:
        test = test[test['e_rec']>e_cut]
        plot_dl2.plot_features(test)
        plt.show()
        plot_dl2.plot_e(test)
        plt.show()
        plot_dl2.plot_disp(test)
        plt.show()
        plot_dl2.plot_pos(test)
        plt.show()
        plot_dl2.plot_importances(cls_gh, features_sep)
        plt.show()
        plot_dl2.plot_ROC(cls_gh, test, features_sep, e_cut)
        plt.show()
