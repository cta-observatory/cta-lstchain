import numpy as np
import os
from lstchain.reco.utils import camera_to_altaz

import astropy.units as u
from astropy.table import Table, Column, vstack
from astropy.io import fits
from astropy.time import Time

__all__ = [
    'make_hdu',
    'create_obs_hdu_index',
    'create_event_list'
    ]


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
            #edisp_table = Table.read(edisp, hdu="ENERGY DISPERSION")
            #aeff_table = Table.read(aeff, hdu="EFFECTIVE AREA")
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
        file_dir = [fits_dir]
        file_name = [file]
        hdu_name = ['events']

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

        hdu_tables.append(t_gti)

        ###############################################
        #Energy Dispersion
        if Table.read(str(fits_dir)+"/"+file, hdu="ENERGY DISPERSION"):
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

        # Filling up the remainng quantities
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
    hdu_name='HDU_INDEX'
    hdu_list = fits.HDUList()
    hdu_list.append(hdu)
    #fits.append(fits_dir+filename_hdu_table, hdu_table, header='HDU_INDEX')
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
    hdu_list = fits.HDUList()
    hdu_list.append(obs)
    #fits.append(fits_dir+filename_obs_table, obs_table, header='OBS_INDEX')
    hdu_list.writeto(str(fits_dir)+"/"+filename_obs_table, overwrite=True) #Make it to update

    return



def create_event_list(data, run_number, Source_name):

    """Create the event_list HDUs from the given data and fill it up with the relevant values

    Parameters
    ----------
        Data: DL2 data file
                HDF5 file
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

    #Initiate tables
    event_table = Table()
    gti_table = Table()
    pointing_table = Table()

    lam = 1000 #Average rate of triggered events, taken by hand for now
    # Timing parameters
    t_start = data.dragon_time.values[0]
    t_stop = data.dragon_time.values[-1]
    time = Time(data.dragon_time, format='unix', scale="utc")

    obs_time = t_stop-t_start

    #Position parameters
    focal = 28 * u.m
    pos_x = data.reco_src_x.values * u .m
    pos_y = data.reco_src_y.values * u .m
    pointing_alt = data.alt_tel.values * u.rad
    pointing_az = data.az_tel.values *u.rad

    coord = camera_to_altaz(pos_x = pos_x, pos_y=pos_y, focal = focal,
                    pointing_alt = pointing_alt, pointing_az = pointing_az,
                    obstime = time)
    coord_icrs = coord.icrs
    coord_pointing = camera_to_altaz(pos_x = 0 * u.m, pos_y=0 * u.m, focal = focal,
                pointing_alt = pointing_alt[0], pointing_az = pointing_az[0],
                obstime = time[0])

    ##########################################################################
    ### Event table columns
    event_table["EVENT_ID"] = Column(
        data.event_id , unit="", description="event_id",
        dtype=np.int64, format="E"
        )
    event_table["TIME"] = Column(
        data.dragon_time , unit="s", description="time",
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
        data.reco_energy , unit="TeV",
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

    event_table.meta["OBS_ID"] = run_number
    event_table.meta["TSTART"] = t_start
    event_table.meta["TSTOP"] = t_stop
    event_table.meta["MJDREFI"] = Time(40587, format='unix', scale="utc") # Unix 01/01/1970 0h0m0
    event_table.meta["MJDREFF"] = Time(0,format='unix',scale='utc')
    event_table.meta["TIMEUNIT"] = 's'
    event_table.meta["TIMESYS"] = "UTC"
    event_table.meta["OBJECT"] = name
    event_table.meta["TELLIST"] = 'LST-1'
    event_table.meta["N_TELS"] = 1

    event_table.meta["HDUCLASS"] = "GADF"
    event_table.meta["HDUDOC"] = "https://github.com/open-gamma-ray-astro/gamma-astro-data-formats"
    event_table.meta["HDUVERS"] = "0.2"
    event_table.meta["HDUCLAS1"] = "INDEX"
    event_table.meta["HDUCLAS2"] = "HDU"

    event_table.meta["RA_PNT"]=float(coord_pointing.icrs.ra.deg)
    event_table.meta["DEC_PNT"]=float(coord_pointing.icrs.dec.deg)
    event_table.meta["ALT_PNT"]=round(np.rad2deg(data.alt_tel.values[0]),6)
    event_table.meta["AZ_PNT"]=round(np.rad2deg(data.az_tel.values[0]),6)
    event_table.meta["FOVALIGN"]='ALTAZ'
    event_table.meta["ONTIME"]=obs_time

    #Dead time for DRS4 chip is 26 u_sec
    #Value of lam is taken by hand, to be around the same order of magnitude for now
    event_table.meta["DEADC"]=1/(1+2.6e-5*lam) # 1/(1 + dead_time*lambda)
    event_table.meta["LIVETIME"]=event_table.meta["DEADC"]*event_table.meta["ONTIME"]

    ##########################################################################
    ### GTI table metadata
    gti_table.meta["OBS_ID"]=run_number#data.obs_id
    gti_table.meta["MJDREFI"] = event_table.meta["MJDREFI"]
    gti_table.meta["MJDREFF"] = event_table.meta["MJDREFF"]
    gti_table.meta["TIMESYS"] = event_table.meta["TIMESYS"]
    gti_table.meta["TIMEUNIT"] = event_table.meta["TIMEUNIT"]

    ##########################################################################
    ### Pointing table metadata
    pointing_table.meta["OBS_ID"]=run_number#data.obs_id
    pointing_table.meta["RA_PNT"]=float(coord_pointing.icrs.ra.deg)
    pointing_table.meta["DEC_PNT"]=float(coord_pointing.icrs.dec.deg)
    pointing_table.meta["ALT_PNT"]=event_table.meta["ALT_PNT"]
    pointing_table.meta["AZ_PNT"]=event_table.meta["AZ_PNT"]
    pointing_table.meta["TIME"]=t_start

    ### Create HDUs
    #########################################################################
    event = make_hdu(hdu_name = "EVENTS", table = event_table)
    gti = make_hdu(hdu_name= "GTI", table = gti_table)
    pointing = make_hdu(hdu_name = "POINTING", table = pointing_table)
    return event, gti, pointing
