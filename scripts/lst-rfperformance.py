"""Pipeline for test the performance of Random Forests.

Usage:

$>python lst-rfperformance.py

"""
import sys                                                   
sys.path.insert(0, '../')
import numpy as np
import pandas as pd
import argparse
import matplotlib.pyplot as plt 
from sklearn.externals import joblib
import sys
sys.path.insert(0, '../')
from lstchain.reco import reco_dl1_to_dl2
from lstchain.visualization import plot_dl2
from lstchain.reco import utils 

parser = argparse.ArgumentParser(description = "Train Random Forests.")

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
                     default='../../cta-lstchain-extra/reco/models/')
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
    df_gammas = pd.read_hdf(args.gammafile,
                            key="gamma_events")
    df_proton = pd.read_hdf(args.protonfile,
                            key="proton_events")
    
    train,test = reco_dl1_to_dl2.split_traintest(df_gammas,0.5)
    test = test.append(df_proton,
                       ignore_index=True)

    RFreg_Energy, RFreg_Disp = reco_dl1_to_dl2.trainRFreco(train,features)
    
    test['e_rec'] = RFreg_Energy.predict(test[features])
    test['disp_rec'] = RFreg_Disp.predict(test[features])

    test['src_x_rec'],test['src_y_rec'] = utils.disp_to_pos(test['disp_rec'],
                                                      test['x'],
                                                      test['y'],
                                                      test['psi'])
    
    features_sep = list(features)
    features_sep.append('e_rec')
    features_sep.append('disp_rec')
    
    train,test = reco_dl1_to_dl2.split_traintest(test,0.75)
    #Train the Classifier
    
    RFcls_GH = reco_dl1_to_dl2.trainRFsep(train,
                          features_sep)

    test['hadro_rec'] = RFcls_GH.predict(test[features_sep])
    
    if args.storerf==True:
        fileE = args.path_models+"/RFreg_Energy.sav"
        fileD = args.path_models+"/RFreg_Disp.sav"
        fileH = args.path_models+"/RFcls_GH.sav"
        joblib.dump(RFreg_Energy, fileE)
        joblib.dump(RFreg_Disp, fileD)
        joblib.dump(RFcls_GH, fileH)

    
    #Plot some results
    e_cuts = [-1,np.log10(500),np.log10(1000)]
    
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
        plot_dl2.plot_importances(RFcls_GH,features_sep)
        plt.show()
        plot_dl2.plot_ROC(RFcls_GH,test,features_sep,e_cut)
        plt.show()
