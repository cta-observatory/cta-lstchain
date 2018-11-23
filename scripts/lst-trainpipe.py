"""Pipeline for training three Random Forests destinated to Energy, disp_
reconstruction and Gamma/Hadron separation.
The resulting RF models can be stored in files for later use on data.

Usage:

$> python lst-trainpipe arg1 arg2 ...

"""
import sys                                                   
sys.path.insert(0, '../')
import argparse
from lstchain.reco import reco_dl1_to_dl2

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
                     default='./results/')

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


    RFreg_Energy, RFreg_Disp, RFcls_GH = reco_dl1_to_dl2.buildModels(
        args.gammafile,
        args.protonfile,
        features,
        args.storerf,
        args.path_models,
    )
    
