"""Pipeline for reconstruction of Energy, disp and gamma/hadron
separation of events stored in a simtelarray file.
Result is a dataframe with dl2 data.
Already trained Random Forests are required.

Usage:

$> python lst-recopipe arg1 arg2 ...

"""

from lstchain.reco import dl0_to_dl1
from lstchain.reco import dl1_to_dl2
from lstchain.visualization import plot_dl2
from sklearn.externals import joblib
from ctapipe.utils import get_dataset_path
import matplotlib.pyplot as plt 
import argparse
import os
import pandas as pd 
from distutils.util import strtobool

parser = argparse.ArgumentParser(description="Reconstruct events")

# Required arguments
parser.add_argument('--datafile', '-f', type=str,
                    dest='datafile',
                    help='path to the file with simtelarray events',
                    default=get_dataset_path('gamma_test_large.simtel.gz'))

parser.add_argument('--pathmodels', '-p', action='store', type=str,
                     dest='path_models',
                     help='Path where to find the trained RF',
                     default='./trained_models')

parser.add_argument('--storeresults', '-s', action='store', type=lambda x: bool(strtobool(x)),
                    dest='storeresults',
                    help='Boolean. True for storing the reco dl2 events'
                    'Default=True, use False otherwise',
                    default=True)
# Optional argument
parser.add_argument('--outdir', '-o', action='store', type=str,
                     dest='outdir',
                     help='Path where to store the reco dl2 events',
                     default='./dl2_results')
parser.add_argument('--maxevents', '-x', action='store', type=int,
                     dest='max_events',
                     help='Maximum number of events to analyze',
                     default=None)

args = parser.parse_args()

if __name__ == '__main__':

    #Get out the data from the Simtelarray file:

    dl0_to_dl1.max_events = args.max_events

    dl0_to_dl1.r0_to_dl1(args.datafile)
    output_filename = 'dl1_' + os.path.basename(args.datafile).split('.')[0] + '.h5'
    data = pd.read_hdf(output_filename,key='events/LSTCam')

    #Load the trained RF for reconstruction:
    fileE = args.path_models + "/reg_energy.sav"
    fileD = args.path_models + "/reg_disp.sav"
    fileH = args.path_models + "/cls_gh.sav"
    
    RFreg_Energy = joblib.load(fileE)
    RFreg_Disp = joblib.load(fileD)
    RFcls_GH = joblib.load(fileH)
    
    #Apply the models to the data
    features = ['intensity',
                'time_gradient',
                'width',
                'length',
                'wl',
                'phi',
                'psi']

    dl2 = dl1_to_dl2.apply_models(data, features, RFcls_GH, RFreg_Energy, RFreg_Disp)

    if args.storeresults==True:
        #Store results
        os.makedirs(args.outdir, exist_ok=True)
        outfile = args.outdir+"/dl2_events.hdf5"
        dl2.to_hdf(outfile, key="dl2_events", mode="w")

    #Plot some results
        
    plot_dl2.plot_features(dl2)
    plt.show()
    plot_dl2.plot_e(dl2)
    plt.show()
    plot_dl2.plot_disp(dl2)
    plt.show()
    plot_dl2.plot_pos(dl2)
    plt.show()

    
