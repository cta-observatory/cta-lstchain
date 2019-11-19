#!/usr//bin/env python

## Code train models from DL1 files onsite (La Palma cluster)


# TODO: find config in logs from prod ID ?


import sys
import os
import shutil
import argparse
from lstchain.io.data_management import *

parser = argparse.ArgumentParser(description="Train models onsite")

parser.add_argument('--gamma_dl1_training_file', '-fg', type=str,
                    help='path to the gamma file',
                    )

parser.add_argument('--proton_dl1_training_file', '-fp', type=str,
                    help='path to the proton file',
                    )

parser.add_argument('--config_file', '-conf', action='store', type=str,
                    dest='config_file',
                    help='Path to a configuration file. If none is given, a standard configuration is applied',
                    default=None
                    )

args = parser.parse_args()

# source env onsite - can be changed for custom install
source_env = 'source /local/home/lstanalyzer/.bashrc; conda activate cta;'


if __name__ == '__main__':

    print("\n ==== START {} ==== \n".format(sys.argv[0]))

    dl1_gamma_dir = os.path.dirname(os.path.abspath(args.gamma_dl1_training_file))
    dl1_proton_dir = os.path.dirname(os.path.abspath(args.proton_dl1_training_file))

    check_prod_id(dl1_gamma_dir, dl1_proton_dir)

    models_dir = dl1_proton_dir.replace('/mc/dl1', '/models')
    models_dir = models_dir.replace('/proton/', '/')

    check_and_make_dir(models_dir)

    base_cmd = ''
    base_cmd += source_env
    base_cmd += 'lstchain_mc_trainpipe.py -fg {} -fp {} -o {}'.format(os.path.abspath(args.gamma_dl1_training_file),
                                                                      os.path.abspath(args.proton_dl1_training_file),
                                                                      models_dir,
                                                                      )
    if args.config_file is not None:
        base_cmd = base_cmd + ' -conf {}'.format(args.config_file)

    jobo = os.path.join(models_dir, "train_job.o")
    jobe = os.path.join(models_dir, "train_job.e")

    cmd = 'sbatch -e {} -o {} --wrap "{}" '.format(jobe, jobo, base_cmd)

    print(cmd)
    os.system(cmd)

    # copy this script itself into logs
    shutil.copyfile(sys.argv[0], os.path.join(models_dir, os.path.basename(sys.argv[0])))
    # copy config file into logs
    if args.config_file is not None:
        shutil.copy(args.config_file, os.path.join(models_dir, os.path.basename(args.config_file)))

    print("\n ==== END {} ==== \n".format(sys.argv[0]))
