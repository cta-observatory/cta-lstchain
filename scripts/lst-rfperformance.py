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
'''
parser.add_argument('--gammafile', '-fg', type=str,
                    dest='gammafile',
                    help='path to the dl1 file of gamma events for training')

parser.add_argument('--protonfile', '-fp', type=str,
                    dest='protonfile',
                    help='path to the dl1 file of proton events for training')
'''
parser.add_argument('--inpath', '-i', action='store', type=str,
                     dest='path_sets',
                     help='Path to sets for train/test',
                     default='../../cta-lstchain-extra/reco/models/sample_data/dl1')

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
                'skewness',
                'kurtosis',
                'impact']
    train_gfile = args.path_sets+'/gammas_train.h5'
    train_sepfile = args.path_sets+'/train_separation.h5'
    test_sepfile = args.path_sets+'/test_separation.h5'

    train_g = pd.read_hdf(train_gfile)
    train_sep = pd.read_hdf(train_sepfile)
    test_sep = pd.read_hdf(test_sepfile)
    
    RFreg_Energy, RFreg_Disp = reco_dl1_to_dl2.trainRFreco(train_g,features)
    train_sep['e_rec'] = RFreg_Energy.predict(train_sep[features])
    train_sep['disp_rec'] = RFreg_Energy.predict(train_sep[features])
    test_sep['e_rec'] = RFreg_Energy.predict(test_sep[features])
    test_sep['disp_rec'] = RFreg_Disp.predict(test_sep[features])

    train_sep['src_x_rec'],train_sep['src_y_rec'] = utils.disp_to_pos(train_sep['disp_rec'],
                                                      train_sep['x'],
                                                      train_sep['y'],
                                                      train_sep['psi'])
    test_sep['src_x_rec'],test_sep['src_y_rec'] = utils.disp_to_pos(test_sep['disp_rec'],
                                                      test_sep['x'],
                                                      test_sep['y'],
                                                      test_sep['psi'])
    
    features_sep = list(features)
    features_sep.append('e_rec')
    features_sep.append('disp_rec')
        
    #Train the Classifier
    
    RFcls_GH = reco_dl1_to_dl2.trainRFsep(train_sep,
                          features_sep)

    test_sep['prob'] = RFcls_GH.predict_proba(test_sep[features_sep])[0:,1]
    #test_sep['hadro_rec'] = RFcls_GH.predict(test_sep[features_sep])
    test_sep['hadro_rec'] = test_sep['prob']>0.4
    
    if args.storerf==True:
        fileE = args.path_models+"/RFreg_Energy.sav"
        fileD = args.path_models+"/RFreg_Disp.sav"
        fileH = args.path_models+"/RFcls_GH.sav"
        joblib.dump(RFreg_Energy, fileE)
        joblib.dump(RFreg_Disp, fileD)
        joblib.dump(RFcls_GH, fileH)

    
    #Plot some results
    e_cuts = [-1,np.log10(250),np.log10(500),np.log10(1000)]

    for e_cut in e_cuts:
        test_sep = test_sep[test_sep['e_rec']>e_cut]
        plt.hist(test_sep[test_sep['hadroness']==0]['prob'],bins=50)
        plt.hist(test_sep[test_sep['hadroness']==1]['prob'],bins=50)
        plt.show()
        plot_dl2.plot_features(test_sep)
        plt.show()
        plot_dl2.plot_e(test_sep)
        plt.show()
        plot_dl2.plot_disp(test_sep)
        plt.show()
        plot_dl2.plot_pos(test_sep)
        plt.show()
        plot_dl2.plot_importances(RFcls_GH,features_sep)
        plt.show()
        plot_dl2.plot_ROC(RFcls_GH,test_sep,features_sep,e_cut)
        plt.show()
