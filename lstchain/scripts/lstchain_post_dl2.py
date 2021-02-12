#!/usr/bin/env python3
"""
Run post-DL2 analysis
"""
import argparse
import toml
from lstchain.analysis.post_dl2 import analyze_on_off, analyze_wobble, LOGGER, setup_logging


def get_parser():
    """
    Create command line options parser

    :return: Parser object
    :rtype: class `argparse.ArgumentParser`
    """
    parser = argparse.ArgumentParser()
    required_named = parser.add_argument_group('required named arguments')
    required_named.add_argument('-c', '--config', help='Analysis configuration file name',
                                required=True)
    parser.add_argument('-v', '--verbosity', help='Logging verbosity level', type=int,
                        choices=[0, 1, 2], default=1)
    return parser


def run(config_file_name):
    """
    Parse toml config and run the selected analysis
    """
    config = toml.load(config_file_name)
    LOGGER.info("Loaded configuration: \n%s", toml.dumps(config))
    if config['analysis']['type'] == 'on_off':
        analyze_on_off(config)
    elif config['analysis']['type'] == 'wobble':
        analyze_wobble(config)
    else:
        pass


def main():
    """
    Parse command line input arguments and run the analysis
    """
    parser = get_parser()
    args = parser.parse_args()
    setup_logging(args.verbosity)
    run(args.config)


if __name__ == "__main__":
    main()

