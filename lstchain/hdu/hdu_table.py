import numpy as np
import os
from lstchain.reco.utils import camera_to_altaz

import astropy.units as u
from astropy.table import Table, Column, vstack, QTable
from astropy.io import fits
from astropy.time import Time

__all__ = [
    'create_obs_hdu_index',
    'create_event_list'
    ]


DEFAULT_HEADER = fits.Header()
DEFAULT_HEADER["HDUDOC"] = "https://github.com/open-gamma-ray-astro/gamma-astro-data-formats"
DEFAULT_HEADER["HDUVERS"] = "0.2"
DEFAULT_HEADER["HDUCLASS"] = "GADF"

#IRF should be separate?
def create_obs_hdu_index(filename_list, fits_dir):
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
    filename_list : list
        list of filenames
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
    source = []
    N_TELS = []
    TELLIST = []

    # loop through the files
    for file in filename_list:

        if os.path.exists(str(fits_dir)+"/"+file):
            try:
                event_table = Table.read(str(fits_dir)+"/"+file, hdu="EVENTS")
                gti_table = Table.read(str(fits_dir)+"/"+file, hdu="GTI")
                pointing_table = Table.read(str(fits_dir)+"/"+file, hdu="POINTING")
            except Exception:
                print(f"fits corrupted for file {file}")
                continue
        else:
            print(f"fits {file} doesn't exist")
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
        obs_id_hdu_table = [event_table.meta["OBS_ID"]]
        file_dir = ['']
        file_name = [file]
        hdu_name = ['events']

        #hdu_table = QTable({})
        t_events = Table([obs_id_hdu_table, hdu_type_name, hdu_class_1, hdu_class_2, hdu_class_3, hdu_class_4,
                    file_dir, file_name, hdu_name],
                names=('OBS_ID', 'HDU_TYPE', 'HDU_CLASS', 'HDU_CLASS2', 'HDU_CLASS3', 'HDU_CLASS4',
                    'FILE_DIR', 'FILE_NAME', 'HDU_NAME'),
                dtype=('>i8', 'S6', 'S10', 'S20', 'S20', 'S20', 'S70', 'S54', 'S20'),
                meta={'name': 'HDU_INDEX'}
                )
        hdu_tables.append(t_events)

        ###############################################
        #GTI
        t_gti = t_events.copy()

        t_gti['HDU_TYPE'] = ['gti']
        t_gti['HDU_CLASS'] = ['gti']
        t_gti['HDU_NAME'] = ['gti']

        hdu_tables.append(t_gti)

        ###############################################
        #POINTING
        t_pnt = t_events.copy()
        t_pnt['HDU_TYPE'] = ['pointing']
        t_pnt['HDU_CLASS'] = ['pointing']
        t_pnt['HDU_NAME'] = ['pointing']

        hdu_tables.append(t_pnt)

        ###############################################
        #Energy Dispersion
        if Table.read(str(fits_dir)+"/"+file, hdu="EDISP"):
            t_edisp = t_events.copy()
            t_edisp['HDU_TYPE'] = ['edisp']
            t_edisp['HDU_CLASS'] = ['edisp_2d']
            t_edisp['HDU_CLASS2'] = ['EDISP']
            t_edisp['HDU_CLASS3'] = ['POINT-LIKE']
            t_edisp['HDU_CLASS4'] = ['EDISP_2D']
            t_edisp['HDU_NAME'] = ['ENERGY DISPERSION']

            hdu_tables.append(t_edisp)
        else:
            print('Energy Dispersion HDU not found')

        ###############################################
        #Effective Area
        if Table.read(str(fits_dir)+"/"+file, hdu="EFFECTIVE AREA"):
            t_aeff = t_edisp.copy()
            t_aeff['HDU_TYPE'] = ['aeff']
            t_aeff['HDU_CLASS'] = ['aeff_2d']
            t_aeff['HDU_CLASS2'] = ['AEFF']
            t_aeff['HDU_CLASS4'] = ['AEFF_2D']
            t_aeff['HDU_NAME'] = ['EFFECTIVE AREA']

            hdu_tables.append(t_aeff)
        else:
            print('Effective Area HDU not found')

        ###############################################

        # Filling up the remainng quantities for obs_index
        obs_id.append(event_table.meta['OBS_ID'])
        source.append(event_table.meta['OBJECT'])
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
    hdu_table.meta["TELESCOP"] = "CTA"
    hdu_table.meta["INSTRUME"] = "LST-1"

    filename_hdu_table = 'hdu-index.fits.gz'
    hdu = fits.BinTableHDU(hdu_table)
    #hdu_name='HDU_INDEX'
    hdu_list = fits.HDUList([fits.PrimaryHDU(), hdu])
    #hdu_list.append(hdu)
    hdu_list.writeto(str(fits_dir)+"/"+filename_hdu_table, overwrite=True)

    #Complete OBS table
    obs_table = Table(
        [obs_id, ra_pnt, dec_pnt, zen_pnt, alt_pnt, az_pnt, ontime, livetime, deadc, tstart, tstop, source, N_TELS, TELLIST],
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
    obs_table.meta["TELESCOP"] = "CTA"
    obs_table.meta["INSTRUME"] = "LST-1"

    obs_table.meta['MJDREFI'] = event_table.meta['MJDREFI']
    obs_table.meta['MJDREFF'] = event_table.meta['MJDREFF']

    filename_obs_table = 'obs-index.fits.gz'
    obs = fits.BinTableHDU(obs_table)
    hdu_list = fits.HDUList([fits.PrimaryHDU(), obs])
    hdu_list.writeto(str(fits_dir)+"/"+filename_obs_table, overwrite=True) #Make it to update

    return

def create_event_list(data, run_number, Source_name):
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
    name=Source_name

    lam = 1000 #Average rate of triggered events, taken by hand for now
    # Timing parameters
    t_start = data['dragon_time'][0]
    t_stop = data['dragon_time'][-1]
    time = Time(data['dragon_time'], format='unix', scale="utc")

    obs_time = t_stop-t_start

    #Position parameters
    focal = 28 * u.m
    pos_x = data['reco_src_x'] * u .m
    pos_y = data['reco_src_y'] * u .m
    pointing_alt = data['alt_tel'] * u.rad
    pointing_az = data['az_tel'] *u.rad

    coord = camera_to_altaz(pos_x = pos_x, pos_y=pos_y, focal = focal,
                    pointing_alt = pointing_alt, pointing_az = pointing_az,
                    obstime = time)
    coord_icrs = coord.icrs
    coord_pointing = camera_to_altaz(pos_x = 0 * u.m, pos_y=0 * u.m, focal = focal,
                pointing_alt = pointing_alt[0], pointing_az = pointing_az[0],
                obstime = time[0])

    ##########################################################################
    ### Event table
    event_table = QTable(
            {
                "EVENT_ID" : u.Quantity(data['event_id'], ndmin=2),
                "TIME" : u.Quantity(data['dragon_time'] * u.s, ndmin=2),
                "RA" : u.Quantity(coord_icrs.ra.deg * u.deg, ndmin=2),
                "DEC" : u.Quantity(coord_icrs.dec.deg * u.deg, ndmin=2),
                "ENERGY" : u.Quantity(data['reco_energy'] * u.TeV, ndmin=2)
            }
        )
    ##########################################################################
    ### GTI table
    gti_table = QTable(
        {
            "START" : u.Quantity(t_start * u.s, ndmin=2),
            "STOP" : u.Quantity(t_stop * u.s, ndmin=2)
        }
    )
    ##########################################################################
    ### Adding the meta data
    ### Event table metadata
    ev_header = DEFAULT_HEADER.copy()
    ev_header["HDUCLAS1"] = "EVENTS"

    ev_header["OBS_ID"] = run_number
    ev_header["TSTART"] = t_start
    ev_header["TSTOP"] = t_stop
    ev_header["MJDREFI"] = '40587' #Time('', format='mjd', scale="utc") # Unix 01/01/1970 0h0m0
    ev_header["MJDREFF"] = '0' #Time('0',format='mjd',scale='utc')
    ev_header["TIMEUNIT"] = 's'
    ev_header["TIMESYS"] = "UTC"
    ev_header["OBJECT"] = name
    ev_header["TELLIST"] = 'LST-1'
    ev_header["N_TELS"] = 1

    ev_header["RA_PNT"]=float(coord_pointing.icrs.ra.deg)
    ev_header["DEC_PNT"]=float(coord_pointing.icrs.dec.deg)
    ev_header["ALT_PNT"]=round(np.rad2deg(data['alt_tel'][0]),6)
    ev_header["AZ_PNT"]=round(np.rad2deg(data['az_tel'][0]),6)
    ev_header["FOVALIGN"]='ALTAZ'
    ev_header["ONTIME"]=obs_time

    #Dead time for DRS4 chip is 26 u_sec
    #Value of lam is taken by hand, to be around the same order of magnitude for now
    ev_header["DEADC"]=1/(1+2.6e-5*lam) # 1/(1 + dead_time*lambda)
    ev_header["LIVETIME"]=ev_header["DEADC"]*ev_header["ONTIME"]

    ##########################################################################
    ### GTI table metadata
    gti_header = DEFAULT_HEADER.copy()
    gti_header["HDUCLAS1"] = "GTI"

    gti_header["OBS_ID"]=run_number
    gti_header["MJDREFI"] = ev_header["MJDREFI"]
    gti_header["MJDREFF"] = ev_header["MJDREFF"]
    gti_header["TIMESYS"] = ev_header["TIMESYS"]
    gti_header["TIMEUNIT"] = ev_header["TIMEUNIT"]

    ##########################################################################
    ### Pointing table metadata
    pnt_header = DEFAULT_HEADER.copy()
    pnt_header["HDUCLAS1"] = "POINTING"

    pnt_header["OBS_ID"]=run_number
    pnt_header["RA_PNT"]=float(coord_pointing.icrs.ra.deg)
    pnt_header["DEC_PNT"]=float(coord_pointing.icrs.dec.deg)
    pnt_header["ALT_PNT"]=ev_header["ALT_PNT"]
    pnt_header["AZ_PNT"]=ev_header["AZ_PNT"]
    pnt_header["TIME"]=t_start

    ### Create HDUs
    #########################################################################
    pnt_table = QTable()

    event = fits.BinTableHDU(event_table, header = ev_header, name = 'EVENTS')
    gti = fits.BinTableHDU(gti_table, header = gti_header, name = 'GTI')
    pointing = fits.BinTableHDU(pnt_table, header = pnt_header, name = 'POINTING')

    return event, gti, pointing
