#!/usr/bin/env python3

"""

"""

import argparse
import logging
from pathlib import Path

from lstchain.io.config import (read_configuration_file, dump_config,
                                get_standard_config, get_mc_config, get_srcdep_config
                                )


log = logging.getLogger(__name__)


def update_std_config(new_config):
    std_config = get_standard_config()
    std_config.update(new_config)
    return std_config


def build_parser():
    parser = argparse.ArgumentParser(description="Dump lstchain config in a file.")

    # Required arguments
    parser.add_argument('-o', '--output-file',
                        help='path to the output file containing the config',
                        default='lstchain_config.json',
                        type=Path,
                        )

    parser.add_argument('--update-with',
                        help='Path to a partial config to update the full config with',
                        type=Path
                        )

    parser.add_argument('--mc',
                        action='store_true',
                        help='Use to dump a modified standard configuration for Monte-Carlo analysis')

    parser.add_argument('--src-dep',
                        action='store_true',
                        help='Use to dump a modified standard configuration for source dependent analysis')

    parser.add_argument('--overwrite',
                        action='store_true',
                        help='Overwrite existing output file',
                        )

    return parser


def main():

    args = build_parser().parse_args()

    if args.mc and args.src_dep:
        raise ValueError("--mc and --src-dep can't be used at the same time")

    if args.mc:
        config = get_mc_config()
    elif args.src_dep:
        config = get_srcdep_config()
    else:
        config = get_standard_config()

    if args.update_with:
        if not args.update_with.is_file():
            raise FileNotFoundError(f"Config file {args.update_with} does not exist")
        extra_config = read_configuration_file(args.update_with)
        config.update(extra_config)

    dump_config(config, args.output_file, overwrite=args.overwrite)

    log.info(f"Config dumped in {args.output_file}")


if __name__ == '__main__':
    main()
