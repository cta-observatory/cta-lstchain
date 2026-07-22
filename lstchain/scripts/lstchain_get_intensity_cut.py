#!/usr/bin/env python

import argparse
import logging
from lstchain.io.io import dl2_params_lstcam_key
from ctapipe.io import read_table
from lstchain.reco.utils import get_intensity_cut

log = logging.getLogger(__name__)
parser = argparse.ArgumentParser(description="Script to provide recommended minimum intensity cut for the DL3 creation, based on the intensity spectrum observed in the data.")
parser.add_argument('-f', '--dl2-file', dest='file', required=True,
                    type=str, help='Input DL2 file')

def main():
    log.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    logging.getLogger().addHandler(handler)

    args = parser.parse_args()
    file = args.file

    min_intensity = get_intensity_cut(read_table(file, dl2_params_lstcam_key))
    log.info('Recommended minimum intensity cut (p.e.):')
    log.info(f'{int(min_intensity+0.5):d}')

if __name__ == '__main__':
    main()
