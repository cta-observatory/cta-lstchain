#!/usr/bin/env python3

"""
Script to create DL3 of single (merged) DL2 run along with IRFs generated from lstchain_mc_dl2_to_irf.py.

The selection cuts applied are the same as those used in generating IRFs.

 - Input: Path where the merged DL2 data HDF5 file is present
          Source name
          IRFs
 - Output: DL3 of the input DL2 data file in fits format.

Usage:
$> python lstchain_dl2_to_dl3.py
--input-data ./DL2/dl2_LST-1.Run*.h5
--output-fits-dir ./DL3/
--add-irf True
--input-irf-file ./IRF/irf.fits.gz
--source-name Crab
--config ../../data/data_selection_cuts.json
"""

import os
import json
from distutils.util import strtobool
import numpy as np
import argparse
from pathlib import Path
import logging
import sys
import pandas as pd
from astropy import table
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
from astropy.table import Table, Column, vstack, QTable


# ~ from lstchain.irf import create_event_list
from lstchain.io import read_configuration_file, get_standard_config
from lstchain.reco.utils import filter_events
from lstchain.paths import run_info_from_filename

from astropy.io import fits
import astropy.units as u
from astropy.coordinates.angle_utilities import angular_separation

from pyirf.utils import calculate_source_fov_offset
log = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description="DL2 to DL3")

# Required arguments
parser.add_argument('--input-file', '-f', type=Path,
                    dest='input_data',
                    help='path to merged DL2 data HDF5 file',
                    default=None, required=True
                    )

parser.add_argument('--output-dir', '-o', type=Path,
                    dest='output_fits_dir',
                    help='path to output fits files',
                    default=None, required=True
                    )
                    
parser.add_argument('--config', '-conf', type=Path,
                    dest='config',
                    help='Config file for selection cuts',
                    default=None, required=False
                    )

parser.add_argument('--add-irf', '-add-irf', action='store',
		            type=lambda x: bool(strtobool(x)), dest='add_irf',
                    help='Boolean: True to add IRF to DL3',
                    default=True, required=False
                    )

parser.add_argument('--input-irf-file', '-irf', type=Path, dest='irf',
                    help='Path to the fits.gz file of IRFs',
                    default="/fefs/aswg/data/mc/IRF/20200629_prod5_trans_80/zenith_20deg/south_pointing/20210416_v0.7.3_prod5_trans_80_local_taicut_8_4/diffuse/irf_20210416_v073_prod5_trans_80_local_taicut_8_4_gamma_diffuse.fits.gz", required=False
                    )

parser.add_argument('--source-name', '-s', type=str,
                    dest='source_name',
                    help='Name of the source',
                    default="None",required=False
                    )


args = parser.parse_args()

DEFAULT_HEADER = fits.Header()
DEFAULT_HEADER["HDUDOC"] = "https://github.com/open-gamma-ray-astro/gamma-astro-data-formats"
DEFAULT_HEADER["HDUVERS"] = "0.2"
DEFAULT_HEADER["HDUCLASS"] = "GADF"

dl2_params_lstcam_key = 'dl2/event/telescope/parameters/LST_LSTCam'

def read_data_dl2_to_QTable(filename):
    """
    Read data DL2 files from lstchain and return QTable format

    Parameters
    ----------
    filename: path to the lstchain DL2 file
    Returns
    -------
    `astropy.table.QTable`
    """
    # Mapping
    name_mapping = {
        'gammaness' : 'gh_score',
        'alt_tel': 'pointing_alt',
        'az_tel': 'pointing_az',

    }
    unit_mapping = {
        'reco_energy': u.TeV,
        'pointing_alt': u.rad,
        'pointing_az': u.rad,
        'reco_alt': u.rad,
        'reco_az': u.rad,
        'trigger_time': u.s,
        'reco_src_x': u.m,
        'reco_src_y': u.m
    }

    data = pd.read_hdf(filename, key = dl2_params_lstcam_key).rename(columns=name_mapping)
    data = table.QTable.from_pandas(data)
	# Make the columns as Quantity
    for k, v in unit_mapping.items():
        data[k] *= v
    return data
    
    
def create_event_list(data, run_number, source_name):
    """
    Create the event_list BinTableHDUs from the given data

    Parameters
    ----------
        Data: DL2 data file
                'astropy.table.QTable'
        Run: Run number
                Int
        Source_name: Name of the source
                Str
    Returns
    -------
        Events HDU:  `astropy.io.fits.BinTableHDU`
        GTI HDU:  `astropy.io.fits.BinTableHDU`
        Pointing HDU:  `astropy.io.fits.BinTableHDU`
    """
    
    # Timing parameters
    lam = 2800 # Average rate of triggered events, taken by hand for now
    t_start = data["trigger_time"].value[0]
    t_stop = data["trigger_time"].value[-1]
    time = Time(data["trigger_time"], format='unix', scale="utc")
    date_obs = time[0].to_value('iso', 'date')
    obs_time = t_stop-t_start # All corrections excluded

    # Position parameters
    location = EarthLocation.from_geodetic(-17.89139 * u.deg, 28.76139 * u.deg, 2184 * u.m)
    reco_alt = data['reco_alt']
    reco_az = data['reco_az']
    pointing_alt = data['pointing_alt']
    pointing_az = data['pointing_az']

    src_sky_pos = SkyCoord(alt=reco_alt, az=reco_az, frame=AltAz(obstime=time,
                location=location)).transform_to(frame='icrs')
    tel_pnt_sky_pos = SkyCoord(alt=pointing_alt[0], az=pointing_az[0], frame=AltAz(
                obstime=time[0], location=location)).transform_to(frame='icrs')

    # ~ try:
        # ~ object_radec=SkyCoord.from_name(source_name)
    # ~ except:
        # ~ log.error('Timeout Error in finding Object in Sesame')
    object_radec = SkyCoord(tel_pnt_sky_pos.icrs)

    # Observation modes
    # ~ source_pointing_diff = object_radec.separation(
                # ~ SkyCoord(tel_pnt_sky_pos.ra, tel_pnt_sky_pos.dec)
                # ~ ).deg
    # Assuming wobble offset is fixed to 0.4
    # ~ if round(source_pointing_diff, 1) == 0.4:
    mode = 'WOBBLE'
    # ~ elif round(source_pointing_diff, 1) > 1:
        # ~ mode = 'OFF'
    # ~ elif round(source_pointing_diff, 1) == 0.:
        # ~ mode = 'ON'
    # ~ else:
        # ~ mode = 'ON-MISPOINTING' # Either this or modify the method of getting ON mode

    # ~ log.error(f'Source pointing difference with camera pointing is {source_pointing_diff:.3f} deg' )

    # Event table
    event_table = QTable(
            {
                "EVENT_ID" : u.Quantity(data['event_id']),
                "TIME" : u.Quantity(data['trigger_time']),
                "RA" : u.Quantity(src_sky_pos.ra.to(u.deg)),
                "DEC" : u.Quantity(src_sky_pos.dec.to(u.deg)),
                "ENERGY" : u.Quantity(data['reco_energy'])
            }
        )
    # GTI table
    gti_table = QTable(
        {
            "START" : u.Quantity(t_start, ndmin=1),
            "STOP" : u.Quantity(t_stop, ndmin=1)
        }
    )
    # Adding the meta data
    # Event table metadata
    ev_header = DEFAULT_HEADER.copy()
    ev_header["HDUCLAS1"] = "EVENTS"

    ev_header["OBS_ID"] = run_number
    ev_header["DATE_OBS"] = date_obs
    ev_header["TSTART"] = t_start
    ev_header["TSTOP"] = t_stop
    ev_header["MJDREFI"] = '40587' # mjd format
    ev_header["MJDREFF"] = '0'
    ev_header["TIMEUNIT"] = 's'
    ev_header["TIMESYS"] = "UTC"
    ev_header["OBJECT"] = source_name
    ev_header["OBS_MODE"] = mode
    ev_header["N_TELS"] = data["tel_id"][0]
    ev_header["TELLIST"] = f'LST-{data["tel_id"][0]}'

    ev_header["RA_PNT"] = tel_pnt_sky_pos.ra.value
    ev_header["DEC_PNT"] = tel_pnt_sky_pos.dec.value
    ev_header["ALT_PNT"] = round(np.rad2deg(data['pointing_alt'].value.mean()),6)
    ev_header["AZ_PNT"] = round(np.rad2deg(data['pointing_az'].value[0]),6)
    ev_header["RA_OBJ"] = object_radec.ra.value
    ev_header["DEC_OBJ"] = object_radec.dec.value
    ev_header["FOVALIGN"] = 'ALTAZ'

    ev_header["ONTIME"] = obs_time
    # Dead time for DRS4 chip is 26 u_sec
    ev_header["DEADC"] = 1/(1+2.6e-5*lam) # 1/(1 + dead_time*lambda)
    ev_header["LIVETIME"] = ev_header["DEADC"]*ev_header["ONTIME"]

    # GTI table metadata
    gti_header = DEFAULT_HEADER.copy()
    gti_header["HDUCLAS1"] = "GTI"

    gti_header["OBS_ID"]=run_number
    gti_header["MJDREFI"] = ev_header["MJDREFI"]
    gti_header["MJDREFF"] = ev_header["MJDREFF"]
    gti_header["TIMESYS"] = ev_header["TIMESYS"]
    gti_header["TIMEUNIT"] = ev_header["TIMEUNIT"]

    # Pointing table metadata
    pnt_header = DEFAULT_HEADER.copy()
    pnt_header["HDUCLAS1"] = "POINTING"

    pnt_header["OBS_ID"] = run_number
    pnt_header["RA_PNT"] = tel_pnt_sky_pos.ra.value
    pnt_header["DEC_PNT"] = tel_pnt_sky_pos.dec.value
    pnt_header["ALT_PNT"] = ev_header["ALT_PNT"]
    pnt_header["AZ_PNT"] = ev_header["AZ_PNT"]
    pnt_header["TIME"] = t_start

    # Create HDUs
    pnt_table = QTable()

    event = fits.BinTableHDU(event_table, header = ev_header, name = 'EVENTS')
    gti = fits.BinTableHDU(gti_table, header = gti_header, name = 'GTI')
    pointing = fits.BinTableHDU(pnt_table, header = pnt_header, name = 'POINTING')
    return event, gti, pointing

def main():
    if not args.input_data.is_file():
        log.error('Input Path does not exist or is not a file')
        sys.exit(1)
    file = str(args.input_data).split('/')[-1]

    output_dir = args.output_fits_dir.absolute()
    output_dir.mkdir(exist_ok=True)

    data = read_data_dl2_to_QTable(args.input_data)
    

    #data['reco_source_fov_offset'] = calculate_source_fov_offset(data, prefix='reco')

    # Get the run_id from the filename if it is -1 in the obs_id column
    #if data['obs_id'][0] <= 0:
    #    run_number=run_info_from_filename(args.input_data)[1]
    #else:
    run_number= data['obs_id'][0]

    log.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    logging.getLogger().addHandler(handler)

    if args.config is None:
        cuts = read_configuration_file(os.path.join(os.path.dirname(__file__), './data_selection_cuts.json'))
    else:
        cuts = read_configuration_file(args.config)
    data = filter_events(data, cuts["events_filters"])
    # Separate cuts for angular separations, for now. Will be included later in filter_events
    data = data[data["gh_score"] > cuts["fixed_cuts"]["gh_score"][0]]

    # ~ data = data[data["reco_source_fov_offset"] < u.Quantity(**cuts["fixed_cuts"]["source_fov_offset"])]

    # Create primary HDU
    events, gti, pointing = create_event_list(data=data, run_number=run_number,source_name=args.source_name)


    name_dl3_file = file.replace('dl2', 'dl3')
    name_dl3_file = name_dl3_file.replace('h5', 'fits')

    if args.add_irf:
        irf = fits.open(args.irf)
        aeff2d = irf['EFFECTIVE AREA']
        edisp2d = irf['ENERGY DISPERSION']
        # bkg2d = irf['BACKGROUND']
        # psf = irf['PSF']
        hdulist = fits.HDUList([fits.PrimaryHDU(), events, gti, pointing, aeff2d, edisp2d])
    else:
        hdulist = fits.HDUList([fits.PrimaryHDU(), events, gti, pointing])
    hdulist.writeto((args.output_fits_dir/name_dl3_file),overwrite=True)

if __name__ == '__main__':
    main()
