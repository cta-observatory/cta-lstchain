"""Pipeline for reconstruction of Energy, Disp and gamma/hadron
separation of events stored in a simtelarray file.
Result is a dataframe with dl2 data.
Already trained Random Forests are required.

Usage:

$> python lst-recopipe arg1 arg2 ...

"""
import calib_dl0_to_dl1
import reco_dl1_to_dl2
import plot_dl2
from sklearn.externals import joblib
from ctapipe.utils import get_dataset_path
import matplotlib.pyplot as plt 
import argparse

parser = argparse.ArgumentParser(description = "Train Random Forests.")

# Required arguments
parser.add_argument('--datafile', '-f', type=str,
                    dest='datafile',
                    help='path to the file with simtelarray events',
                    default=get_dataset_path('gamma_test_large.simtel.gz'))

parser.add_argument('--pathmodels', '-p', action='store', type=str,
                     dest='path_models',
                     help='Path where to find the trained RF',
                     default='./results/')
parser.add_argument('--storeresults', '-s', action='store', type=bool,
                    dest='storeresults',
                    help='Boolean. True for storing the reco dl2 events'
                    'Default=False, any user input will be considered True',
                    default=False)
# Optional argument
parser.add_argument('--outdir', '-o', action='store', type=str,
                     dest='outdir',
                     help='Path where to store the reco dl2 events',
                     default='./results/')

args = parser.parse_args()

if __name__ == '__main__':

    #Get out the data from the Simtelarray file:

    data = calib_dl0_to_dl1.get_events(args.datafile,False)

    #Load the trained RF for reconstruction:
    fileE = args.path_models+"RFreg_Energy.sav"
    fileD = args.path_models+"RFreg_Disp.sav"
    fileH = args.path_models+"RFcls_GH.sav"
    
    RFreg_Energy = joblib.load(fileE)
    RFreg_Disp = joblib.load(fileD)
    RFcls_GH = joblib.load(fileH)
    
    #Apply the models to the data
    features = ['intensity',
                'time_gradient',
                'width',
                'length',
                'w/l',
                'phi',
                'psi']
    dl2 = data
    reco_dl1_to_dl2.ApplyModels(data,dl2,features,RFcls_GH,RFreg_Energy,RFreg_Disp)

    if args.storeresults==True:
        #Store results
        outfile = args.outdir+"dl2_events.hdf5"
        dl2.to_hdf(outfile,key="dl2_events",mode="w")

    #Plot some results
        
    plot_dl2.plot_features(dl2)
    plt.show()
    plot_dl2.plot_E(dl2)
    plt.show()
    plot_dl2.plot_Disp(dl2)
    plt.show()
    plot_dl2.plot_pos(dl2)
    plt.show()
