# script to add the source dependent parameters to a DL1 file

import os
import argparse
import pandas as pd
from lstchain.reco.dl1_to_dl2 import get_source_dependent_parameters
from lstchain.io import read_configuration_file
from lstchain.io.io import dl1_params_src_dep_lstcam_key, write_dataframe, dl1_params_lstcam_key


parser = argparse.ArgumentParser(description="Add the source dependent parameters to a DL1 file")

# Required arguments
parser.add_argument('--datafile', '-f', type=str,
                    dest='datafile',
                    help='path to a DL1 HDF5 file',
                    )

parser.add_argument('--config_file', '-conf', action='store', type=str,
                    dest='config_file',
                    help='Path to a configuration file for source dependent analysis',
                    default=None
                    )

args = parser.parse_args()

def main():

    dl1_filename = os.path.abspath(args.datafile)

    config = {}
    if args.config_file is not None:
        try:
            config = read_configuration_file(os.path.abspath(args.config_file))
        except("Custom configuration could not be loaded !!!"):                                                                                            
            pass

    dl1_params = pd.read_hdf(dl1_filename, key=dl1_params_lstcam_key)
    src_dep_df = get_source_dependent_parameters(dl1_params, config)
    write_dataframe(src_dep_df, dl1_filename, dl1_params_src_dep_lstcam_key)
 

if __name__ == '__main__':
    main()
 
