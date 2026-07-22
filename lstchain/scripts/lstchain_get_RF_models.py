#!/usr/bin/env python

import argparse
import logging
from lstchain.io.provenance import read_dl2_provenance

log = logging.getLogger(__name__)
parser = argparse.ArgumentParser(description="Script to find the RF models used in the production of a DL2 file.")
parser.add_argument('-f', '--dl2-file', dest='file', required=True,
                    type=str, help='Input DL2 file')

def main():
    log.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    logging.getLogger().addHandler(handler)

    args = parser.parse_args()
    file = args.file

    prov = read_dl2_provenance(file)
    log.info('Used RF models:')
    log.info(prov['input'][1]['url']+'\n')

if __name__ == '__main__':
    main()
