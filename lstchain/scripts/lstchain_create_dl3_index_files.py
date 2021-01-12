#!/usr/bin/env python3

"""
A separate script to index the HDU tables and Observation tables of the DL3 files
in a particular directory.
Use a pattern to enter multiple DL3 files as arguments for them to be enlisted

Usage for n files:
$> python lstchain_create_dl3_index_files.py
--input-dl3-files ./DL3[1-n]*fits
"""

from pathlib import Path
import argparse

from lstchain.irf import create_obs_hdu_index

parser = argparse.ArgumentParser(description="Creating DL3 index files")

# Required arguments
parser.add_argument(
    "--input-dl3-files",
    "-f",
    nargs="+",
    dest="input_dl3_files",
    help="Path of DL3 files, listed in a pattern *[1-n]*",
    required=True,
)

args = parser.parse_args()


def main():

    filename_list = []
    dir = Path(args.input_dl3_files[0]).parent

    for f in args.input_dl3_files:
        filename_list.append(Path(f).name)

    create_obs_hdu_index(filename_list, dir)


if __name__ == "__main__":
    main()
