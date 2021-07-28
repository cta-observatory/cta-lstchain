#!/usr/bin/env python3

"""
Script to merge DL3

Usage:
$> python merge_DL3.py
--input-filter "dl3_LST-1.Run04190.*.fits"
--input-directory "./DL3/"
--output-dir "./DL3/"
"""

from astropy.io import fits
import numpy as np
from astropy.table import Table, Column, vstack, QTable
import astropy.units as u
import glob
import argparse
from pathlib import Path

parser = argparse.ArgumentParser(description="DL3 merger")

# Required arguments
parser.add_argument('--input-filter', '-f', type=str,
                    dest='input_data',
                    help='DL3 filter selection, example : "dl3_LST-1.Run04190.*.fits"',
                    default=None, required=True
                    )
                    
parser.add_argument('--input-directory', '-d', type=str,
                    dest='input_dir',
                    help='path to input DL3 files directory',
                    default=None, required=True
                    )

parser.add_argument('--output-dir', '-o', type=str,
                    dest='output_fits_dir',
                    help='path to output files',
                    default=None, required=True
                    )

args = parser.parse_args()




file_filter = args.input_data #"dl3_LST-1.Run04190.*.fits"
directory = args.input_dir #"/home/sami.caroff/cta-lstchain-enrique/cta-lstchain/lstchain/scripts/"
filelist = glob.glob(directory+file_filter)

hdu1 = fits.open(filelist[0])
for file in filelist[1:]:
    hdu2 = fits.open(file)
    nev_hdu1 = len(hdu1['EVENTS'].data)
    nev_hdu2 = len(hdu2['EVENTS'].data)
    hdu1['EVENTS'].header['RA_PNT'] = (hdu1['EVENTS'].header['RA_PNT']*nev_hdu1 + hdu2['EVENTS'].header['RA_PNT']*nev_hdu2)/(nev_hdu1+nev_hdu2)
    hdu1['EVENTS'].header['DEC_PNT'] = (hdu1['EVENTS'].header['DEC_PNT']*nev_hdu1 + hdu2['EVENTS'].header['DEC_PNT']*nev_hdu2)/(nev_hdu1+nev_hdu2)
    hdu1['EVENTS'].header['ALT_PNT'] = (hdu1['EVENTS'].header['ALT_PNT']*nev_hdu1 + hdu2['EVENTS'].header['ALT_PNT']*nev_hdu2)/(nev_hdu1+nev_hdu2)
    hdu1['EVENTS'].header['AZ_PNT'] = (hdu1['EVENTS'].header['AZ_PNT']*nev_hdu1 + hdu2['EVENTS'].header['AZ_PNT']*nev_hdu2)/(nev_hdu1+nev_hdu2)
    hdu1['EVENTS'].header['DEADC'] = (hdu1['EVENTS'].header['DEADC']*nev_hdu1 + hdu2['EVENTS'].header['DEADC']*nev_hdu2)/(nev_hdu1+nev_hdu2)
    if not (hdu1['EVENTS'].header['OBS_ID'] == hdu2['EVENTS'].header['OBS_ID']):
        print("BEWARE ! You are merging DL3 event with different OBS_ID... You should not do that...")
    #print(len(hdu1['EVENTS'].data))
    hdu1['EVENTS'].data = np.append(hdu1['EVENTS'].data,hdu2['EVENTS'].data)
    
#print(len(hdu1['EVENTS'].data))
hdu1['EVENTS'].data = np.sort(hdu1['EVENTS'].data)

#hdu1['EVENTS'].data['TIME'] = hdu1['EVENTS'].data['TIME'] 
#hdu1['EVENTS'].data['RA'] = hdu1['EVENTS'].data['RA'] 
#hdu1['EVENTS'].data['DEC'] = hdu1['EVENTS'].data['DEC'] 
#hdu1['EVENTS'].data['ENERGY'] = hdu1['EVENTS'].data['ENERGY'] 
#hdu1['EVENTS'].data[:][] = hdu1['EVENTS'].data[:][1]*u.s

hdu1['EVENTS'].header['TSTART'] = hdu1['EVENTS'].data[0][1]
hdu1['EVENTS'].header['TSTOP'] = hdu1['EVENTS'].data[-1][1]
hdu1['EVENTS'].header['ONTIME'] = hdu1['EVENTS'].data[-1][1] - hdu1['EVENTS'].data[0][1]
hdu1['EVENTS'].header['LIVETIME'] = hdu1['EVENTS'].header['ONTIME'] * hdu1['EVENTS'].header['DEADC']
hdu1['POINTING'].header['ALT_PNT'] = hdu1['EVENTS'].header['ALT_PNT']
hdu1['POINTING'].header['AZ_PNT'] = hdu1['EVENTS'].header['AZ_PNT']
hdu1['POINTING'].header['TIME'] = hdu1['EVENTS'].header['TSTART']
prefix = (5-len(str(hdu1['EVENTS'].header['OBS_ID'])))*"0"
#print(args.output_fits_dir+'dl3_LST-1.Run'+prefix+str(hdu1['EVENTS'].header['OBS_ID'])+'.fits')
hdu1['EVENTS'].columns['TIME'].unit = "s"
hdu1['EVENTS'].columns['ENERGY'].unit = "TeV"
hdu1['EVENTS'].columns['RA'].unit = "deg"
hdu1['EVENTS'].columns['DEC'].unit = "deg"

hdu1['GTI'].data[0][0] = hdu1['EVENTS'].data[0][1]
hdu1['GTI'].data[0][1] = hdu1['EVENTS'].data[-1][1]

hdu1.writeto(args.output_fits_dir+'dl3_LST-1.Run'+prefix+str(hdu1['EVENTS'].header['OBS_ID'])+'.fits',overwrite=True)
