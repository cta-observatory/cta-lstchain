#!/usr/bin/env python3

"""
Script to create event list of multiple Runs along with IRF files.
The IRF files are already created by using pyIRF package, which will soon be brought under standard software pipeline.
The gammaness cut have to be the same as those used while creating the IRFs.
For now the selection cuts applied on the real data is given by hand

Problems to solve:
The IRFs maybe present in a separate folder, or by default in the same folder as output
The merged real DL2 files are assumed here to be in the same folder
This can be either solved by taking only 1 Run and then using a separate script to merge the event lists, or
    by specifying the filenames / indices / range in a sorted order
The 2 HDU functions defined here can be moved to another script and imported to use here.

 - Input: Path where the merged DL2 data HDF5 files are present
          Path where the generated IRF files (Effective area and Energy dispersion for now) are present.
          Source name
 - Output: Event lists of the input data files in fits format.
           HDU and Obs index fits files

Usage:
$> python lstchain_create_event_list.py
--input-data-dir ./DL2/
--input-irf-dir ./IRF/
--output-fits-dir ./DL3/
--num-files 1
--source-name Crab
"""

import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import argparse

from lstchain.reco.utils import filter_events, camera_to_altaz
from lstchain.io.io import dl2_params_lstcam_key
from ctapipe.coordinates import CameraFrame

import astropy.units as u
from astropy.table import Table, Column, vstack
from astropy.io import fits
from astropy.coordinates.angle_utilities import angular_separation
from astropy.coordinates import EarthLocation, AltAz, SkyCoord, SkyOffsetFrame
from astropy.time import Time


def make_hdu(hdu_name, table):
    """Create the BinTableHDU from the Table

    Parameters
    ----------
        hdu_name: str
            name of the HDU
        table: `~astropy.table.Table`

    Returns
    -------
       hdu:  `astropy.io.fits.BinTableHDU`
    """
    col_list = [
        fits.Column(col.name, col.format, unit=str(col.unit), array=col.data)
        for col in table.columns.values()
    ]
    hdu = fits.BinTableHDU.from_columns(col_list)
    hdu.header.set("EXTNAME", hdu_name)
    for key in table.meta.keys():
        hdu.header.set(key, table.meta[key])
    return hdu

def filter(data, gammaness_cut):
    cut = (data.leakage2_intensity < 0.2) & \
          (data.intensity > 200) & \
          (data.wl > 0.1) & \
          (data.gammaness > gammaness_cut)
    return data[cut]

def create_obs_hdu_index(list_obs_id, fits_dir):
    """
    Create the obs table and hdu table (below some explanation)

    A two-level index file scheme is used in the IACT community to allow an arbitrary folder structures
    For each directory tree, two files should be present:
    **obs-index.fits.gz**
    (defined in http://gamma-astro-data-formats.readthedocs.io/en/latest/data_storage/obs_index/index.html)
    **hdu-index.fits.gz**
    (defined in http://gamma-astro-data-formats.readthedocs.io/en/latest/data_storage/hdu_index/index.html)

    obs-index contains the single run informations e.g.
    (OBS_ID, RA_PNT, DEC_PNT, ZEN_PNT, ALT_PNT)
    while hdu-index contains the informations about the locations of the other HDU (header data units)
    necessary to the analysis e.g. A_eff, E_disp and so on...
        http://gamma-astro-data-formats.readthedocs.io/en/latest/

    This function will create the necessary data format, starting from the path that contains the DL3
    converted fits file.

    Parameters
    ----------
    list_obs_id : list
        list of obs id
    fits_dir : str
        directory containing the fits file
    """

    # empty lists with the quantities we want
    hdu_tables = []

    obs_id = []
    ra_pnt = []
    dec_pnt = []
    zen_pnt = []
    alt_pnt = []
    az_pnt = []
    ontime = []
    livetime = []
    deadc = []
    tstart = []
    tstop = []
    object = []
    N_TELS = []
    TELLIST = []
    etrue_lo = []
    etrue_hi = []
    theta_lo = []
    theta_hi = []
    migr_lo = []
    migr_hi = []
    matrix = []

    # loop through the files
    for irun,run in enumerate(list_obs_id):
        # Standard naming, maybe modified later
        filename = f"dl3_LST-1_0{run}_merged.fits"
        edisp_file = "energy_dispersion.fits"
        aeff_file = "effective_area.fits"

        if os.path.exists(fits_dir+filename):
            try:
                event_table = Table.read(f"{fits_dir}{filename}", hdu="EVENTS")
                gti_table = Table.read(f"{fits_dir}{filename}", hdu="GTI")
                pointing_table = Table.read(f"{fits_dir}{filename}", hdu="POINTING")
                edisp_table = Table.read(f"{fits_dir}{edisp_file}", hdu="ENERGY DISPERSION")
                aeff_table = Table.read(f"{fits_dir}{aeff_file}", hdu="EFFECTIVE AREA")
            except Exception:
                print(f"fits corrupted for file {filename} or irf files corrupted")
                continue
        else:
            print(f"fits {filename} doesn't exist")
            continue

        #The column names for the table follows the scheme as shown in
        #https://gamma-astro-data-formats.readthedocs.io/en/latest/general/HDU_CLASS.html

        #As RESPONSE class will have EFF_AREA, EDISP, RPSF, BKG and RAD_MAX, the names of
        #the corresponding HDU_CLASS_n can change its name

        ###############################################
        # Event list
        hdu_type_name = ['events']
        hdu_class_1 = ['events']
        hdu_class_2 = ['']
        hdu_class_3 = ['']
        hdu_class_4 = ['']
        obs_id_hdu_table = len(hdu_type_name) * [event_table.meta["OBS_ID"]]
        file_dir = len(hdu_type_name) * [fits_dir]
        file_name = len(hdu_type_name) * [filename]
        hdu_name = ['events']

        t_1 = Table([obs_id_hdu_table, hdu_type_name, hdu_class_1, hdu_class_2, hdu_class_3, hdu_class_4,
                       file_dir, file_name, hdu_name],
                    names=('OBS_ID', 'HDU_TYPE', 'HDU_CLASS', 'HDU_CLASS2', 'HDU_CLASS3', 'HDU_CLASS4',
                           'FILE_DIR', 'FILE_NAME', 'HDU_NAME'),
                    dtype=('>i8', 'S6', 'S10', 'S20', 'S20', 'S20', 'S70', 'S54', 'S20'),
                        meta={'name': 'HDU_INDEX'}
                    )
        hdu_tables.append(t_1)

        ###############################################
        #GTI
        hdu_type_name = ['gti']
        hdu_class_1 = ['gti']
        hdu_class_2 = ['']
        hdu_class_3 = ['']
        hdu_class_4 = ['']
        obs_id_hdu_table = len(hdu_type_name) * [gti_table.meta["OBS_ID"]]
        file_dir = len(hdu_type_name) * [fits_dir]
        file_name = len(hdu_type_name) * [filename]
        hdu_name = ['gti']

        t_2 = Table([obs_id_hdu_table, hdu_type_name, hdu_class_1, hdu_class_2, hdu_class_3, hdu_class_4,
                       file_dir, file_name, hdu_name],
                    names=('OBS_ID', 'HDU_TYPE', 'HDU_CLASS', 'HDU_CLASS2', 'HDU_CLASS3', 'HDU_CLASS4',
                           'FILE_DIR', 'FILE_NAME', 'HDU_NAME'),
                    dtype=('>i8', 'S6', 'S10', 'S20', 'S20', 'S20', 'S70', 'S54', 'S20'),
                    meta={'name': 'HDU_INDEX'}
                    )
        hdu_tables.append(t_2)

        ###############################################
        #POINTING
        hdu_type_name = ['pointing']
        hdu_class_1 = ['pointing']
        hdu_class_2 = ['']
        hdu_class_3 = ['']
        hdu_class_4 = ['']
        obs_id_hdu_table = len(hdu_type_name) * [pointing_table.meta["OBS_ID"]]
        file_dir = len(hdu_type_name) * [fits_dir]
        file_name = len(hdu_type_name) * [filename]
        hdu_name = ['pointing']

        t_3 = Table([obs_id_hdu_table, hdu_type_name, hdu_class_1, hdu_class_2, hdu_class_3, hdu_class_4,
                       file_dir, file_name, hdu_name],
                    names=('OBS_ID', 'HDU_TYPE', 'HDU_CLASS', 'HDU_CLASS2', 'HDU_CLASS3', 'HDU_CLASS4',
                           'FILE_DIR', 'FILE_NAME', 'HDU_NAME'),
                    dtype=('>i8', 'S6', 'S10', 'S20', 'S20', 'S20', 'S70', 'S54', 'S20'),
                    meta={'name': 'HDU_INDEX'}
                    )
        hdu_tables.append(t_3)

        ###############################################
        #Energy Dispersion
        hdu_type_name = ['edisp']
        hdu_class_1 = ['edisp_2d']
        hdu_class_2 = ['EDISP']
        hdu_class_3 = ['POINT-LIKE']
        hdu_class_4 = ['EDISP_2D']
        obs_id_hdu_table = len(hdu_type_name) * [event_table.meta['OBS_ID']]
        file_dir = len(hdu_type_name) * [fits_dir]
        file_name = len(hdu_type_name) * [edisp_file]
        hdu_name = ['ENERGY DISPERSION']

        e = Table([obs_id_hdu_table, hdu_type_name, hdu_class_1, hdu_class_2, hdu_class_3, hdu_class_4,
                   file_dir, file_name, hdu_name],
                names=('OBS_ID', 'HDU_TYPE', 'HDU_CLASS', 'HDU_CLASS2', 'HDU_CLASS3', 'HDU_CLASS4',
                       'FILE_DIR', 'FILE_NAME', 'HDU_NAME'),
                dtype=('>i8', 'S6', 'S10', 'S20', 'S20', 'S20', 'S70', 'S54', 'S20'),
                meta={'name': 'HDU_INDEX'}
                )
        hdu_tables.append(e)

        ###############################################
        #Effective Area
        hdu_type_name = ['aeff']
        hdu_class_1 = ['aeff_2d']
        hdu_class_2 = ['AEFF']
        hdu_class_3 = ['POINT-LIKE']
        hdu_class_4 = ['AEFF_2D']
        obs_id_hdu_table = len(hdu_type_name) * [event_table.meta['OBS_ID']]
        file_dir = len(hdu_type_name) * [fits_dir]
        file_name = len(hdu_type_name) * [aeff_file]
        hdu_name = ['EFFECTIVE AREA']

        a = Table([obs_id_hdu_table, hdu_type_name, hdu_class_1, hdu_class_2, hdu_class_3, hdu_class_4,
                   file_dir, file_name, hdu_name],
                names=('OBS_ID', 'HDU_TYPE', 'HDU_CLASS', 'HDU_CLASS2', 'HDU_CLASS3', 'HDU_CLASS4',
                       'FILE_DIR', 'FILE_NAME', 'HDU_NAME'),
                dtype=('>i8', 'S6', 'S10', 'S20', 'S20', 'S20', 'S70', 'S54', 'S20'),
                meta={'name': 'HDU_INDEX'}
                )
        hdu_tables.append(a)

        ###############################################

        # Filling up the remainng quantities
        obs_id.append(event_table.meta['OBS_ID'])
        object.append(event_table.meta['OBJECT'])
        N_TELS.append(event_table.meta["N_TELS"])
        TELLIST.append(event_table.meta["TELLIST"])

        ontime.append(event_table.meta["ONTIME"])
        livetime.append(event_table.meta["LIVETIME"])
        deadc.append(event_table.meta["DEADC"])
        tstart.append(gti_table["START"])
        tstop.append(gti_table["STOP"])

        ra_pnt.append(pointing_table.meta['RA_PNT'])
        dec_pnt.append(pointing_table.meta['DEC_PNT'])
        zen_pnt.append(90 - float(pointing_table.meta['ALT_PNT']))
        alt_pnt.append(pointing_table.meta['ALT_PNT'])
        az_pnt.append(pointing_table.meta['AZ_PNT'])


    #Complete HDU table
    hdu_table = vstack(hdu_tables)
    hdu_table.meta['EXTNAME'] = 'HDU_INDEX'
    hdu_table.meta["HDUCLASS"] = "GADF"
    hdu_table.meta["HDUDOC"] = "https://github.com/open-gamma-ray-astro/gamma-astro-data-formats"
    hdu_table.meta["HDUVERS"] = "0.2"
    hdu_table.meta["HDUCLAS1"] = "INDEX"
    hdu_table.meta["HDUCLAS2"] = "HDU"

    filename_hdu_table = 'hdu-index.fits.gz'
    hdu = fits.BinTableHDU(hdu_table)

    hdu_table.write(fits_dir+filename_hdu_table, overwrite=True)

    #Complete OBS table
    obs_table = Table(
        [obs_id, ra_pnt, dec_pnt, zen_pnt, alt_pnt, az_pnt, ontime, livetime, deadc, tstart, tstop, object, N_TELS, TELLIST],
        names=(
            'OBS_ID', 'RA_PNT', 'DEC_PNT', 'ZEN_PNT', 'ALT_PNT', 'AZ_PNT', 'ONTIME', 'LIVETIME', 'DEADC', 'TSTART',
            'TSTOP', 'OBJECT','N_TELS', 'TELLIST'),
        dtype=('>i8', '>f4', '>f4', '>f4', '>f4', '>f4', '>f4', '>f4', '>f4', '>f8', '>f8', 'S20','>i8', 'S20'),
        meta={'name': 'OBS_INDEX'}
    )

    obs_table = vstack(obs_table)
    obs_table.meta['EXTNAME'] = 'OBS_INDEX'
    obs_table.meta["HDUCLASS"] = "GADF"
    obs_table.meta["HDUDOC"] = "https://github.com/open-gamma-ray-astro/gamma-astro-data-formats"
    obs_table.meta["HDUVERS"] = "0.2"
    obs_table.meta["HDUCLAS1"] = "INDEX"
    obs_table.meta["HDUCLAS2"] = "OBS"

    obs_table.meta['MJDREFI'] = event_table.meta['MJDREFI']
    obs_table.meta['MJDREFF'] = event_table.meta['MJDREFF']

    filename_obs_table = 'obs-index.fits.gz'
    obs = fits.BinTableHDU(obs_table)
    obs_table.write(fits_dir+filename_obs_table, overwrite=True)



parser = argparse.ArgumentParser(description="DL2 to event_list")

# Required arguments
parser.add_argument('--input-data-dir', '-d', type=str,
                    dest='input_data_dir',
                    help='path to merged DL2 data HDF5 files',
                    default=None, required=True) ### Create a default folder with small DL2 files

parser.add_argument('--output-fits-dir', '-o', type=str,
                    dest='output_fits_dir',
                    help='path to output fits files',
                    default=None, required=True)

parser.add_argument('--source-name', '-s', type=str,
                    dest='source_name',
                    help='Name of the source',
                    default='Crab', required=True)

parser.add_argument('--num-files', '-n', type=int,
                    dest='number_of_files',
                    help='Number of files in sort order',
                    default=1, required=True)

# Optional arguments
parser.add_argument('--input-irf-dir', '-irf', type=str,
                    dest='input_irf_dir',
                    help='path to IRF files',
                    default=None, required=False)


args = parser.parse_args()

def main():

    directory_fits_file=args.output_fits_dir
    if not os.path.isdir(directory_fits_file):
        os.mkdir(directory_fits_file)

    # Search for dl2 files by start or end name
    # Assuming the current standard nomenclature for file name
    # Also assuming that all the required merged DL2 files are kept in a single folder
    # This has to be generalized

    #end_name = 'v0.5.1_v03.h5'
    start_name = 'dl2_Run'
    file_list = []
    Run = []
    for file in os.listdir(args.input_data_dir):
        if file.startswith(start_name):
            file_list.append(file)
            Run.append(int(file.split('_')[2]))
    file_list = (np.sort(file_list))
    Run = (np.sort(Run))

    # Cut to select the gamma-like event
    gammaness_cut = 0.8
    offset_cut = 0.14

    #IRF file names
    edisp_file = "energy_dispersion.fits"
    aeff_file = "effective_area.fits"
    #bkg_file = ""

    #Open the already created IRF fits files
    if args.input_irf_dir is None:
        edisp_fits = fits.open(args.output_fits_dir+edisp_file, hdu="ENERGY DISPERSION")
        aeff_fits = fits.open(args.output_fits_dir+aeff_file, hdu="EFFECTIVE AREA")
        #bkg_fits = fits.open(args.output_fits_dir+bkg_file, hdu="BACKGROUND")
    else:
        edisp_fits = fits.open(args.input_irf_dir+edisp_file, hdu="ENERGY DISPERSION")
        aeff_fits = fits.open(args.input_irf_dir+aeff_file, hdu="EFFECTIVE AREA")
        #bkg_fits = fits.open(args.input_irf_dir+bkg_file, hdu="BACKGROUND")

    #Create primary HDU
    n = np.arange(100) # a simple sequence of floats from 0.0 to 99.9
    primary_hdu = fits.PrimaryHDU(n)

    #Run across multiple runs
    #first indices for the file list
    # Ask for the number of files to read
    range = np.arange(args.number_of_files)

    for i in range:

        real_df = filter(pd.read_hdf(args.input_data_dir+file_list[i], key=dl2_params_lstcam_key), gammaness_cut)

        lam = 1000 #Average rate of triggered events, taken by hand for now

        #Initiate tables
        event_table = Table()
        gti_table = Table()
        pointing_table = Table()

        # Timing parameters
        t_start = real_df.dragon_time.values[0]
        t_stop = real_df.dragon_time.values[-1]
        time = Time(real_df.dragon_time, format='unix', scale="utc")
        mjd = time.mjd.mean()
        obs_time = t_stop-t_start

        #Position parameters
        focal = 28 * u.m
        pos_x = real_df["reco_src_x"].values * u .m
        pos_y = real_df["reco_src_y"].values * u .m
        pointing_alt = real_df["alt_tel"].values * u.rad
        pointing_az = real_df["az_tel"].values *u.rad

        coord = camera_to_altaz(pos_x = pos_x, pos_y=pos_y, focal = focal,
                        pointing_alt = pointing_alt, pointing_az = pointing_az,
                        obstime = time)
        coord_icrs = coord.icrs
        coord_pointing = camera_to_altaz(pos_x = 0 * u.m, pos_y=0 * u.m, focal = focal,
                        pointing_alt = pointing_alt[0], pointing_az = pointing_az[0],
                        obstime = time[0])

        #Number of events selected
        n_event=len(real_df.event_id)

        ##########################################################################
        ### Event table columns
        event_table["EVENT_ID"] = Column(
            real_df.event_id , unit="", description="event_id",
            dtype=np.int64, format="E"
            )
        event_table["TIME"] = Column(
            real_df.dragon_time , unit="s", description="time",
            dtype=np.float64, format="F"
            )
        event_table["RA"] = Column(
            coord_icrs.ra.deg , unit="deg", description="ra",
            dtype=np.float64, format="E"
            )
        event_table["DEC"] = Column(
            coord_icrs.dec.deg  , unit="deg", description="dec",
            dtype=np.float64, format="E"
            )
        event_table["ENERGY"] = Column(
            real_df.reco_energy , unit="TeV",
            description="energy", dtype=np.float64, format="E"
            )
        ##########################################################################
        ### GTI table columns
        gti_table["START"] = Column(
            [t_start], unit='s', description='Start time',
            dtype=np.float64, format="F", length=1
            )
        gti_table["STOP"]=Column(
            [t_stop], unit='s', description='Stop time',
            dtype=np.float64, format="F", length=1
            )
        ##########################################################################
        ### Adding the meta data
        ### Event table metadata
        event_table.meta["OBS_ID"]=Run[i]
        event_table.meta["MJDREFI"]=int(mjd)
        event_table.meta["MJDREFF"]=mjd-int(mjd)
        event_table.meta["OBJECT"]=args.source_name #This should be asked
        event_table.meta["TELLIST"]='LST-1'
        event_table.meta["N_TELS"]=1
        event_table.meta["G_CUT"]=gammaness_cut
        event_table.meta["ANG_CUT"]=offset_cut

        event_table.meta["HDUCLASS"] = "GADF"
        event_table.meta["HDUDOC"] = "https://github.com/open-gamma-ray-astro/gamma-astro-data-formats"
        event_table.meta["HDUVERS"] = "0.2"
        event_table.meta["HDUCLAS1"] = "INDEX"
        event_table.meta["HDUCLAS2"] = "HDU"

        event_table.meta["RA_PNT"]=coord_pointing.icrs.ra.deg
        event_table.meta["DEC_PNT"]=coord_pointing.icrs.dec.deg
        event_table.meta["ALT_PNT"]=round(np.rad2deg(real_df["alt_tel"].values[0]),6)
        event_table.meta["AZ_PNT"]=round(np.rad2deg(real_df["az_tel"].values[0]),6)
        event_table.meta["FOVALIGN"]='ALTAZ'

        event_table.meta["ONTIME"]=obs_time
        #Dead time for DRS4 chip is 26 u_sec
        #Value of lam is taken by hand, to be around the same order of magnitude for now
        event_table.meta["DEADC"]=1/(1+2.6e-5*lam) # 1/(1 + dead_time*lambda)
        event_table.meta["LIVETIME"]=event_table.meta["DEADC"]*event_table.meta["ONTIME"]

        ##########################################################################
        ### GTI table metadata
        gti_table.meta["OBS_ID"]=Run[i]
        gti_table.meta["MJDREFI"] = event_table.meta["MJDREFI"]
        gti_table.meta["MJDREFF"] = event_table.meta["MJDREFF"]

        ##########################################################################
        ### Pointing table metadata
        pointing_table.meta["OBS_ID"]=Run[i]
        pointing_table.meta["TIME"]=t_start
        pointing_table.meta["RA_PNT"]=coord_pointing.icrs.ra.deg
        pointing_table.meta["DEC_PNT"]=coord_pointing.icrs.dec.deg
        pointing_table.meta["ALT_PNT"]=event_table.meta["ALT_PNT"]
        pointing_table.meta["AZ_PNT"]=event_table.meta["AZ_PNT"]

        #########################################################################
        ### Create HDUs
        event = make_hdu(hdu_name = "EVENTS", table = event_table)
        gti = make_hdu(hdu_name= "GTI", table = gti_table)
        pointing = make_hdu(hdu_name = "POINTING", table = pointing_table)

        #HDUlist from the primary, the event hdu and the IRFs
        hdulist = fits.HDUList([primary_hdu, event, gti, pointing, edisp_fits[1], aeff_fits[1]])
        name_dl3_file = f"dl3_LST-1_0{Run[i]}_merged.fits"
        hdulist.writeto(f"{directory_fits_file}{name_dl3_file}",overwrite=True)

    ### Create the Obs and HDU index tables
    list_obs_id=Run[range]
    create_obs_hdu_index(list_obs_id = list_obs_id, fits_dir = directory_fits_file)

if __name__ == '__main__':
    main()
