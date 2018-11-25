
import argparse
from ctapipe.utils import get_dataset_path

from lstchain.reco import dl0_to_dl1
from lstchain.reco import reco_dl1_to_dl2
from lstchain.visualization import plot_dl2
import matplotlib.pyplot as plt
import os

parser = argparse.ArgumentParser(description = "Train Random Forests.")

# Required argument
parser.add_argument('--gammafile', '-fg', type=str,
                    dest='gammafile',
                    help='path to the dl1 file of gamma events for training',
                    )

parser.add_argument('--protonfile', '-fp', type=str,
                    dest='protonfile',
                    help='path to the dl1 file of proton events for training',
                    )

parser.add_argument('--storerf', '-srf', action='store', type=bool,
                    dest='storerf',
                    help='Boolean. True for storing trained RF in 3 files'
                    'Deafult=False, any user input will be considered True',
                    default=False)

parser.add_argument('--datafile', '-f', type=str,
                    dest='datafile',
                    help='path to the file with simtelarray events',
                    default=get_dataset_path('gamma_test_large.simtel.gz'))

parser.add_argument('--storeresults', '-s', action='store', type=bool,
                    dest='storeresults',
                    help='Boolean. True for storing the reco dl2 events'
                    'Default=False, any user input will be considered True',
                    default=False)

# Optional arguments
parser.add_argument('--opath', '-om', action='store', type=str,
                     dest='path_models',
                     help='Path to store the resulting RF',
                     default='./results/')

parser.add_argument('--outdir', '-or', action='store', type=str,
                     dest='outdir',
                     help='Path where to store the reco dl2 events',
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

    RFreg_Energy, RFreg_disp, RFcls_GH = reco_dl1_to_dl2.buildModels(args.gammafile,
                                                                     args.protonfile,
                                                                     features,
                                                                     SaveModels=args.storerf,
                                                                     path_models=args.path_models,
                                                                     )

    #Get out the data from the Simtelarray file:
    
    data = dl0_to_dl1.get_events(args.datafile, False)

    
    #Apply the models to the data
    dl2 = reco_dl1_to_dl2.ApplyModels(data, features, RFcls_GH, RFreg_Energy, RFreg_disp)
    
    if args.storeresults==True:
        #Store results
        if not os.path.exists(args.outdir):
            os.mkdir(args.outdir)
        outfile = args.outdir + "/dl2_events.hdf5"
        dl2.to_hdf(outfile, key="dl2_events", mode="w")

    #Plot some results
        
    plot_dl2.plot_features(dl2)
    plt.show()
    plot_dl2.plot_E(dl2)
    plt.show()
    plot_dl2.plot_disp(dl2)
    plt.show()
    plot_dl2.plot_pos(dl2)
    plt.show()

