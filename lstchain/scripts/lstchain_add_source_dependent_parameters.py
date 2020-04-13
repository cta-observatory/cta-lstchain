# script to add the source dependent parameters to a DL1 file

import os
import argparse
import pandas as pd
from lstchain.reco.dl1_to_dl2 import get_source_dependent_parameters
from lstchain.io import read_configuration_file
from lstchain.io.io import dl1_params_src_dep_lstcam_key, write_dataframe, dl1_params_lstcam_key


def main(dl1_filename, config_filename):
    config = read_configuration_file(config_filename)

    dl1_params = pd.read_hdf(dl1_filename, key=dl1_params_lstcam_key)
    src_dep_df = get_source_dependent_parameters(dl1_params, config=config)
    write_dataframe(src_dep_df, dl1_filename, dl1_params_src_dep_lstcam_key)


if __name__ == '__main__':

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

    config_filename = os.path.abspath(args.config_file)
    dl1_filename = os.path.abspath(args.datafile)

    main(dl1_filename, config_filename)
