import argparse
from ctapipe.utils import get_dataset_path
from lstchain.reco import dl0_to_dl1
from lstchain.io.config import read_configuration_file
import os

parser = argparse.ArgumentParser(description="R0 to DL1")


parser.add_argument('--infile', '-f', type=str,
                    dest='infile',
                    help='path to the file with simtelarray events',
                    default=get_dataset_path('gamma_test_large.simtel.gz'))

parser.add_argument('--outdir', '-o', action='store', type=str,
                    dest='outdir',
                    help='Path where to store the reco dl2 events',
                    default='./dl1_data/')

parser.add_argument('--config_file', '-conf', action='store', type=str,
                    dest='config_file',
                    help='Path to a configuration file. If none is given, a standard configuration is applied',
                    default=None
                    )


args = parser.parse_args()


def main():

    os.makedirs(args.outdir, exist_ok=True)
    
    dl0_to_dl1.allowed_tels = {1, 2, 3, 4}
    output_filename = args.outdir + '/dl1_' + os.path.basename(args.infile).rsplit('.', 1)[0] + '.h5'

    config = {}
    if args.config_file is not None:
        try:
            config = read_configuration_file(os.path.abspath(args.config_file))
        except("Custom configuration could not be loaded !!!"):
            pass

    dl0_to_dl1.r0_to_dl1(args.infile, output_filename=output_filename, custom_config=config)


if __name__ == '__main__':
    main()
