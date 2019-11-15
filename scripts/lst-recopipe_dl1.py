"""Pipeline for reconstruction of Energy, disp and gamma/hadron
separation of events stored in a simtelarray file.
Result is a dataframe with dl2 data.
Already trained Random Forests are required.

Usage:

$> python lst-recopipe arg1 arg2 ...

"""

from lstchain.reco import dl0_to_dl1
from lstchain.reco import dl1_to_dl2
from lstchain.io import get_dataset_keys
from sklearn.externals import joblib
from ctapipe.utils import get_dataset_path
import argparse
import os
import shutil
import pandas as pd 
from distutils.util import strtobool
from lstchain.reco.utils import filter_events
from lstchain.io import read_configuration_file, standard_config, replace_config
from lstchain.io import write_dl2_dataframe
from lstchain.io.io import dl1_params_lstcam_key
import tables
import numpy as np

parser = argparse.ArgumentParser(description="Reconstruct events")

# Required arguments
parser.add_argument('--datafile', '-f', type=str,
                    dest='datafile',
                    help='path to a DL1 HDF5 file',
                    )

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

parser.add_argument('--config_file', '-conf', action='store', type=str,
                    dest='config_file',
                    help='Path to a configuration file. If none is given, a standard configuration is applied',
                    default=None
                    )

args = parser.parse_args()

if __name__ == '__main__':

    custom_config = {}
    if args.config_file is not None:
        try:
            custom_config = read_configuration_file(args.config_file)
        except("Custom configuration could not be loaded !!!"):
            pass

    config = replace_config(standard_config, custom_config)

    data = pd.read_hdf(args.datafile, key=dl1_params_lstcam_key)
    data = filter_events(data, filters=config["events_filters"])


    #Load the trained RF for reconstruction:
    fileE = args.path_models + "/reg_energy.sav"
    fileD = args.path_models + "/reg_disp_vector.sav"
    fileH = args.path_models + "/cls_gh.sav"
    
    reg_energy = joblib.load(fileE)
    reg_disp_vector = joblib.load(fileD)
    cls_gh = joblib.load(fileH)
    
    #Apply the models to the data

    dl2 = dl1_to_dl2.apply_models(data, cls_gh, reg_energy, reg_disp_vector, custom_config=config)

    if args.storeresults==True:
        os.makedirs(args.outdir, exist_ok=True)
        outfile = args.outdir + '/dl2_' + os.path.basename(args.datafile).split('.')[0] + '.h5'

        shutil.copyfile(args.datafile, outfile)
        write_dl2_dataframe(dl2.astype(float), outfile)
