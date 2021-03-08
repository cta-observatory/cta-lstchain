#!/usr/bin/env python3

"""
test for rta
"""

import argparse
import os
from lstchain.io import (
    read_configuration_file,
    standard_config,
    replace_config,
)


parser = argparse.ArgumentParser(description="DL2 to DL3 dummy stage for hiperta_stream tests")

# Required arguments
parser.add_argument('--input-file', '-f', type=str,
                    dest='input_file',
                    help='path to a DL1 HDF5 file',
                    default=None, required=True)

parser.add_argument('--output-dir', '-o', action='store', type=str,
                     dest='output_dir',
                     help='Path where to store the reco dl2 events',
                     default='./dl3')

parser.add_argument('--config', '-c', action='store', type=str,
                    dest='config_file',
                    help='Path to a configuration file. If none is given, a standard configuration is applied',
                    default=None, required=False)

args = parser.parse_args()


def main():

    custom_config = {}
    if args.config_file is not None:
        try:
            custom_config = read_configuration_file(os.path.abspath(args.config_file))
        except("Custom configuration could not be loaded !!!"):
            pass

    config = replace_config(standard_config, custom_config)

    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, os.path.basename(args.input_file).replace('dl2', 'dl3'))

    os.system(f'cp {args.input_file} {output_file}')


if __name__ == '__main__':
    main()
