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
from lstchain.io import get_dataset_keys
from sklearn.externals import joblib
from ctapipe.utils import get_dataset_path
import matplotlib.pyplot as plt 
import argparse
import os
import pandas as pd 
from distutils.util import strtobool
import numpy as np
import tables

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
    dl0_to_dl1.allowed_tels={1}
    dl0_to_dl1.r0_to_dl1(args.datafile)
    dl1_file = 'dl1_' + os.path.basename(args.datafile).split('.')[0] + '.h5'
    
    #dl1_file=args.datafile
    #data = pd.read_hdf(args.datafile, key='events/LSTCam')
    intensity_min = np.log10(50)

    data = dl1_to_dl2.filter_events(pd.read_hdf(dl1_file, key='events/LSTCam'), intensity_min=intensity_min)

    #Load the trained RF for reconstruction:
    fileE = args.path_models + "/reg_energy.sav"
    fileD = args.path_models + "/reg_disp_vector.sav"
    fileH = args.path_models + "/cls_gh.sav"
    
    reg_energy = joblib.load(fileE)
    reg_disp_vector = joblib.load(fileD)
    cls_gh = joblib.load(fileH)
    
    #Apply the models to the data

    features = ['intensity', 'width', 'length', 'x', 'y', 'psi', 'phi', 'wl', 
                'skewness', 'kurtosis','r', 'intercept', 'time_gradient']

    dl2 = dl1_to_dl2.apply_models(data, features, cls_gh, reg_energy, reg_disp_vector)

    if args.storeresults==True:
        #Store results
        os.makedirs(args.outdir, exist_ok=True)
        outfile = args.outdir+'/dl2_' + os.path.basename(args.datafile).split('.')[0] + '.h5'
        dl2.to_hdf(outfile, key="events/LSTCam", mode="w")

        keys = get_dataset_keys(dl1_file)
        groups = set([k.split('/')[0] for k in keys])
        groups.remove('events') # we don't want to copy DL1 events

        f1 = tables.open_file(dl1_file)
        with tables.open_file(outfile, mode='a') as dl2_file:
            nodes = {}
            for g in groups:
                nodes[g] = f1.copy_node('/', name=g, newparent=dl2_file.root, newname=g, recursive=True)

    #Plot some results
    '''
    plot_dl2.plot_features(dl2)
    plt.show()
    plot_dl2.plot_e(dl2)
    plt.show()
    fig, axes = plt.subplots(1, 2, figsize=(15,6))

    axes[0].hist2d(dl2.disp_dx, dl2.disp_dx_rec, bins=60);
    axes[0].set_xlabel('mc_disp')
    axes[0].set_ylabel('reco_disp')
    axes[0].set_title('disp_dx')
    
    axes[1].hist2d(dl2.disp_dy, dl2.disp_dy_rec, bins=60);
    axes[1].set_xlabel('mc_disp')
    axes[1].set_ylabel('reco_disp')
    axes[1].set_title('disp_dy');
    plot_dl2.plot_pos(dl2)
    plt.show()
    plot_dl2.calc_resolution(dl2)
    plt.show()
    plot_dl2.plot_e_resolution(dl2,15)
    plt.show()
    '''
