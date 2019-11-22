"""Pipeline for reconstruction of Energy, disp and gamma/hadron
separation of events stored in a simtelarray file.
Result is a dataframe with dl2 data.
Already trained Random Forests are required.

Usage:

$> python lst-recopipe arg1 arg2 ...

"""

from ctapipe.utils import get_dataset_path
import argparse
import os
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

# Optional argument
parser.add_argument('--store_dl1', '-s1', action='store', type=lambda x: bool(strtobool(x)),
                    dest='store_dl1',
                    help='Boolean. True for storing DL1 file'
                    'Default=False, use True otherwise',
                    default=True)

parser.add_argument('--outdir', '-o', action='store', type=str,
                     dest='outdir',
                     help='Path where to store the reco dl2 events',
                     default='./dl2_data')


parser.add_argument('--config_file', '-conf', action='store', type=str,
                    dest='config_file',
                    help='Path to a configuration file. If none is given, a standard configuration is applied',
                    default=None
                    )

args = parser.parse_args()


if __name__ == '__main__':

    dl1_file = args.outdir + '/dl1_' + os.path.basename(args.datafile).rsplit('.', 1)[0] + '.h5'

    cmd_r0_to_dl1 = f'lstchain_mc_r0_to_dl1.py -f {args.datafile} -o {args.outdir}'
    if args.config_file is not None:
        cmd_r0_to_dl1 = cmd_r0_to_dl1 + f' -conf {args.config_file}'

    cmd_dl1_to_dl2 = f'lstchain_mc_dl1_to_dl2.py -f {dl1_file} -p {args.path_models} -o {args.outdir}'
    if args.config_file is not None:
        cmd_dl1_to_dl2 = cmd_dl1_to_dl2 + f' -conf {args.config_file}'

    os.system(cmd_r0_to_dl1)
    os.system(cmd_dl1_to_dl2)

    if not args.store_dl1:
        os.remove(dl1_file)


if __name__ == '__main__':
    main()
