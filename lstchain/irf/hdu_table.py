import numpy as np
import logging

import astropy.units as u
from astropy.table import Table, QTable
from astropy.io import fits
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
from astropy.time import Time

from lstchain.__init__ import __version__

__all__ = ["create_obs_hdu_index", "create_event_list"]

log = logging.getLogger(__name__)

DEFAULT_HEADER = fits.Header()
DEFAULT_HEADER["CREATOR"] = f"lstchain v{__version__}"
DEFAULT_HEADER["HDUDOC"] = "https://github.com/open-gamma-ray-astro/gamma-astro-data-formats"
DEFAULT_HEADER["HDUVERS"] = "0.2"
DEFAULT_HEADER["HDUCLASS"] = "GADF"
DEFAULT_HEADER["ORIGIN"] = "CTA"
DEFAULT_HEADER["TELESCOP"] = "CTA-N"
DEFAULT_HEADER["CREATED"] = Time.now().utc.iso

location = EarthLocation.from_geodetic(-17.89139 * u.deg, 28.76139 * u.deg, 2184 * u.m)

wobble_offset = 0.4


def create_obs_hdu_index(
    filename_list,
    fits_dir,
    hdu_index_filename,
    obs_index_filename,
    add_fits_dir_meta=False,
):
    """
    Create the obs table and hdu table

    A two-level index file scheme is used in the IACT community to allow
    an arbitrary folder structures. For each directory tree, two files should be present:
    1. obs-index.fits.gz
    (http://gamma-astro-data-formats.readthedocs.io/en/latest/data_storage/obs_index/index.html)
    2. hdu-index.fits.gz
    (http://gamma-astro-data-formats.readthedocs.io/en/latest/data_storage/hdu_index/index.html)

    obs-index contains the single run informations e.g.
    (OBS_ID, RA_PNT, DEC_PNT, ZEN_PNT, ALT_PNT)
    while hdu-index contains the informations about the locations of the other
    HDU (header data units) necessary to the analysis.

    File directory information is currently kept optional in the HDU index table

    Parameters
    ----------
    filename_list : list
        list of filenames
    fits_dir : Path
        directory containing the fits file
    hdu_index_filename : str
        filename for HDU index file
    obs_index_filename : str
        filename for OBS index file
    add_fits_dir_meta : Bool
        True for adding directory path of FITS file in HDU index table

    Returns
    -------
    hdu_index_list : 'astropy.io.fits.HDUList'
        HDU list file for HDU index tables
    obs_index_list : 'astropy.io.fits.HDUList'
        HDU list file for Obs index table
    """

    hdu_index_tables = []
    obs_index_tables = []

    # loop through the files
    for file in filename_list:
        filepath = fits_dir / file
        if filepath.is_file():
            try:
                event_table = Table.read(filepath, hdu="EVENTS")
            except Exception:
                log.error(f"fits corrupted for file {file}")
                continue
        else:
            log.error(f"fits {file} doesn't exist")
            continue

        # The column names for the table follows the scheme as shown in
        # https://gamma-astro-data-formats.readthedocs.io/en/latest/general/HDU_CLASS.html
        # Event list
        t_events = {
            "OBS_ID": event_table.meta["OBS_ID"],
            "HDU_TYPE": "events",
            "HDU_CLASS": "events",
            "HDU_CLASS1": "EVENTS",
            "HDU_CLASS2": "",
            "HDU_CLASS3": "",
            "HDU_CLASS4": "",
            "FILE_DIR": "",
            "FILE_NAME": file,
            "HDU_NAME": "EVENTS",
        }
        if add_fits_dir_meta:
            t_events["FILE_DIR"] = str(fits_dir)

        hdu_index_tables.append(t_events)

        # GTI
        t_gti = t_events.copy()

        t_gti["HDU_TYPE"] = "gti"
        t_gti["HDU_CLASS"] = "gti"
        t_gti["HDU_CLASS1"] = "GTI"
        t_gti["HDU_NAME"] = "GTI"

        hdu_index_tables.append(t_gti)

        # POINTING
        t_pnt = t_events.copy()

        t_pnt["HDU_TYPE"] = "pointing"
        t_pnt["HDU_CLASS"] = "pointing"
        t_pnt["HDU_CLASS1"] = "POINTING"
        t_pnt["HDU_NAME"] = "POINTING"

        hdu_index_tables.append(t_pnt)

        # Energy Dispersion
        try:
            edisp = Table.read(filepath, hdu="ENERGY DISPERSION")
            t_edisp = t_events.copy()

            t_edisp["HDU_TYPE"] = "edisp"
            t_edisp["HDU_CLASS"] = "edisp_2d"
            t_edisp["HDU_CLASS1"] = edisp.meta["HDUCLAS1"]
            t_edisp["HDU_CLASS2"] = edisp.meta["HDUCLAS2"]
            t_edisp["HDU_CLASS3"] = edisp.meta["HDUCLAS3"]
            t_edisp["HDU_CLASS4"] = edisp.meta["HDUCLAS4"]
            t_edisp["HDU_NAME"] = "ENERGY DISPERSION"

            hdu_index_tables.append(t_edisp)
        except KeyError as err:
            log.error("Run {0}:{1}".format(t_events["OBS_ID"], err))

        # Effective Area
        try:
            aeff = Table.read(filepath, hdu="EFFECTIVE AREA")
            t_aeff = t_events.copy()
            t_aeff["HDU_TYPE"] = "aeff"
            t_aeff["HDU_CLASS"] = "aeff_2d"
            t_aeff["HDU_CLASS1"] = aeff.meta["HDUCLAS1"]
            t_aeff["HDU_CLASS2"] = aeff.meta["HDUCLAS2"]
            t_aeff["HDU_CLASS3"] = aeff.meta["HDUCLAS3"]
            t_aeff["HDU_CLASS4"] = aeff.meta["HDUCLAS4"]
            t_aeff["HDU_NAME"] = "EFFECTIVE AREA"

            hdu_index_tables.append(t_aeff)
        except KeyError as err:
            log.error("Run {0}:{1}".format(t_events["OBS_ID"], err))

        # Background
        try:
            bkg = Table.read(filepath, hdu="BACKGROUND")
            t_bkg = t_events.copy()
            t_bkg["HDU_TYPE"] = "bkg"
            t_bkg["HDU_CLASS"] = "bkg_2d"
            t_bkg["HDU_CLASS1"] = bkg.meta["HDUCLAS1"]
            t_bkg["HDU_CLASS2"] = bkg.meta["HDUCLAS2"]
            t_bkg["HDU_CLASS3"] = bkg.meta["HDUCLAS3"]
            t_bkg["HDU_CLASS4"] = bkg.meta["HDUCLAS4"]
            t_bkg["HDU_NAME"] = "BACKGROUND"

            hdu_index_tables.append(t_bkg)
        except KeyError as err:
            log.error("Run {0}:{1}".format(t_events["OBS_ID"], err))

        # PSF
        try:
            psf = Table.read(filepath, hdu="PSF")
            t_psf = t_events.copy()
            t_psf["HDU_TYPE"] = "psf"
            t_psf["HDU_CLASS"] = "psf_table"
            t_psf["HDU_CLASS1"] = psf.meta["HDUCLAS1"]
            t_psf["HDU_CLASS2"] = psf.meta["HDUCLAS2"]
            t_psf["HDU_CLASS3"] = psf.meta["HDUCLAS3"]
            t_psf["HDU_CLASS4"] = psf.meta["HDUCLAS4"]
            t_psf["HDU_NAME"] = "PSF"

            hdu_index_tables.append(t_psf)
        except KeyError as err:
            log.error("Run {0}:{1}".format(t_events["OBS_ID"], err))

        # Obs_table
        t_obs = {
            "OBS_ID": event_table.meta["OBS_ID"],
            "DATE_OBS": event_table.meta["DATE_OBS"],
            "RA_PNT": event_table.meta["RA_PNT"] * u.deg,
            "DEC_PNT": event_table.meta["DEC_PNT"] * u.deg,
            "ZEN_PNT": (90 - float(event_table.meta["ALT_PNT"])) * u.deg,
            "ALT_PNT": event_table.meta["ALT_PNT"] * u.deg,
            "AZ_PNT": event_table.meta["AZ_PNT"] * u.deg,
            "RA_OBJ": event_table.meta["RA_OBJ"] * u.deg,
            "DEC_OBJ": event_table.meta["DEC_OBJ"] * u.deg,
            "TSTART": event_table.meta["TSTART"] * u.s,
            "TSTOP": event_table.meta["TSTOP"] * u.s,
            "ONTIME": event_table.meta["ONTIME"] * u.s,
            "TELAPSE": event_table.meta["TELAPSE"] * u.s,
            "LIVETIME": event_table.meta["LIVETIME"] * u.s,
            "DEADC": event_table.meta["DEADC"],
            "OBJECT": event_table.meta["OBJECT"],
            "OBS_MODE": event_table.meta["OBS_MODE"],
            "N_TELS": event_table.meta["N_TELS"],
            "TELLIST": event_table.meta["TELLIST"],
            "INSTRUME": event_table.meta["INSTRUME"],
        }
        obs_index_tables.append(t_obs)

    hdu_index_table = Table(hdu_index_tables)

    hdu_index_header = DEFAULT_HEADER.copy()
    hdu_index_header["HDUCLAS1"] = "INDEX"
    hdu_index_header["HDUCLAS2"] = "HDU"
    hdu_index_header["INSTRUME"] = t_obs["INSTRUME"]

    hdu_index = fits.BinTableHDU(
        hdu_index_table, header=hdu_index_header, name="HDU INDEX"
    )
    hdu_index_list = fits.HDUList([fits.PrimaryHDU(), hdu_index])

    obs_index_table = QTable(obs_index_tables)

    obs_index_header = hdu_index_header.copy()
    obs_index_header["HDUCLAS2"] = "OBS"
    obs_index_header["MJDREFI"] = event_table.meta["MJDREFI"]
    obs_index_header["MJDREFF"] = event_table.meta["MJDREFF"]

    obs_index = fits.BinTableHDU(
        obs_index_table, header=obs_index_header, name="OBS INDEX"
    )
    obs_index_list = fits.HDUList([fits.PrimaryHDU(), obs_index])

    return hdu_index_list, obs_index_list


def create_event_list(data, run_number, source_ra, source_dec, effective_time, elapsed_time):
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
        Effective_time: Effective time of triggered events of the run
                Float
        Elapsed_time: Total elapsed time of triggered events of the run
                Float
    Returns
    -------
        Events HDU:  `astropy.io.fits.BinTableHDU`
        GTI HDU:  `astropy.io.fits.BinTableHDU`
        Pointing HDU:  `astropy.io.fits.BinTableHDU`
    """

    tel_list = np.unique(data["tel_id"])

    # Timing parameters
    t_start = data["dragon_time"].value[0]
    t_stop = data["dragon_time"].value[-1]
    time = Time(data["dragon_time"], format="unix", scale="utc")
    date_obs = time[0].to_value("iso", "date")

    # Position parameters
    reco_alt = data["reco_alt"]
    reco_az = data["reco_az"]
    pointing_alt = data["pointing_alt"]
    pointing_az = data["pointing_az"]

    src_sky_pos = SkyCoord(
        alt=reco_alt, az=reco_az, frame=AltAz(obstime=time, location=location)
    ).transform_to(frame="icrs")
    tel_pnt_sky_pos = SkyCoord(
        alt=pointing_alt.mean(),
        az=pointing_az.mean(),
        frame=AltAz(obstime=time.mean(), location=location),
    ).transform_to(frame="icrs")

    try:
        object_radec = SkyCoord.from_name(source_name)
    except Exception:
        log.error("Name resolve error in finding the Object in Sesame")
        object_radec = SkyCoord(tel_pnt_sky_pos.icrs)

    # Observation modes
    source_pointing_diff = object_radec.separation(tel_pnt_sky_pos)

    if round(source_pointing_diff, 1) == wobble_offset:
        mode = "WOBBLE"
    elif round(source_pointing_diff, 1) > 1:
        mode = "OFF"
    elif round(source_pointing_diff, 1) == 0.0:
        mode = "ON"
    else:
        # Nomenclature is to be worked out or have a separate way to mark mispointings
        mode = "UNDETERMINED"

    log.info(
        f"Source pointing difference with camera pointing is {source_pointing_diff:.3f} deg"
    )

    event_table = QTable(
        {
            "EVENT_ID": data["event_id"],
            "TIME": data["dragon_time"],
            "RA": src_sky_pos.ra.to(u.deg),
            "DEC": src_sky_pos.dec.to(u.deg),
            "ENERGY": data["reco_energy"],
        }
    )
    gti_table = QTable(
        {"START": u.Quantity(t_start, ndmin=1), "STOP": u.Quantity(t_stop, ndmin=1)}
    )
    pnt_table = QTable()

    # Adding the meta data
    # Event table metadata
    ev_header = DEFAULT_HEADER.copy()
    ev_header["HDUCLAS1"] = "EVENTS"

    ev_header["OBS_ID"] = run_number

    ev_header["DATE_OBS"] = date_obs
    ev_header["TSTART"] = t_start
    ev_header["TSTOP"] = t_stop
    ev_header["MJDREFI"] = "40587"  # mjd format for date 01-01-1970
    ev_header["MJDREFF"] = "0"
    ev_header["TIMEUNIT"] = "s"
    ev_header["TIMESYS"] = "UTC"
    ev_header["ONTIME"] = elapsed_time
    ev_header["TELAPSE"] = t_stop - t_start
    ev_header["DEADC"] = effective_time / elapsed_time
    ev_header["LIVETIME"] = effective_time

    ev_header["OBJECT"] = source_name
    ev_header["OBS_MODE"] = mode

    ev_header["N_TELS"] = len(tel_list)
    ev_header["MULTIP"] = ev_header["N_TELS"]
    ev_header["TELLIST"] = "LST-" + " ".join(map(str, tel_list))
    ev_header["INSTRUME"] = f"{ev_header['TELLIST']}"

    ev_header["RA_PNT"] = tel_pnt_sky_pos.ra.value
    ev_header["DEC_PNT"] = tel_pnt_sky_pos.dec.value
    ev_header["ALT_PNT"] = data["pointing_alt"].mean().to_value(u.deg)
    ev_header["AZ_PNT"] = data["pointing_az"].mean().to_value(u.deg)
    ev_header["RA_OBJ"] = object_radec.ra.value
    ev_header["DEC_OBJ"] = object_radec.dec.value
    ev_header["FOVALIGN"] = "RADEC"

    # GTI table metadata
    gti_header = DEFAULT_HEADER.copy()
    gti_header["HDUCLAS1"] = "GTI"

    gti_header["OBS_ID"] = run_number
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
    event = fits.BinTableHDU(event_table, header=ev_header, name="EVENTS")
    gti = fits.BinTableHDU(gti_table, header=gti_header, name="GTI")
    pointing = fits.BinTableHDU(pnt_table, header=pnt_header, name="POINTING")

    return event, gti, pointing
