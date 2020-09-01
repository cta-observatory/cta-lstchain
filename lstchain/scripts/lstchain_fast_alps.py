#!/usr/bin/env python3
"""
Run fast-alps analysis
"""
import argparse
import logging
import toml
from lstchain.analysis.fast_alps import analyze_on_off, analyze_wobble


LOGGER = logging.getLogger('fast_alps')
LOGGER.setLevel(logging.DEBUG)
LOGGING_LEVELS = {0: logging.ERROR, 1: logging.INFO, 2: logging.DEBUG}


def setup_logging(verbosity=1):
    """
    Setup logger console and file descriptors

    Two log stream handlers are added, one for file-based logging and one for console output.
    Logging level to file is always set to DEBUG and console verbosity can be controlled.
    Verbosity levels {0,1,2} correspond to {ERROR, INFO, DEBUG}.

    :param int verbosity: Verbosity level used for console output
    """
    fh = logging.FileHandler('/tmp/fast_alps.log')
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    # define a Handler which writes INFO messages or higher to the sys.stderr
    console = logging.StreamHandler()
    console.setLevel(LOGGING_LEVELS[verbosity])
    # tell the handler to use this format
    console.setFormatter(formatter)
    # add the handler to the root logger
    LOGGER.addHandler(console)
    LOGGER.addHandler(fh)


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

