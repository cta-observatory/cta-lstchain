import numpy as np
import logging
import os

import astropy.units as u
from astropy.table import Table, QTable
from astropy.io import fits
from astropy.coordinates import SkyCoord, AltAz
from astropy.coordinates.erfa_astrom import ErfaAstromInterpolator, erfa_astrom
from astropy.time import Time

from lstchain.__init__ import __version__
from lstchain.reco.utils import location

__all__ = ["create_obs_index_hdu", "create_hdu_index_hdu", "create_event_list"]

log = logging.getLogger(__name__)

DEFAULT_HEADER = fits.Header()
DEFAULT_HEADER["CREATOR"] = f"lstchain v{__version__}"
DEFAULT_HEADER["HDUDOC"] = "https://github.com/open-gamma-ray-astro/gamma-astro-data-formats"
DEFAULT_HEADER["HDUVERS"] = "0.2"
DEFAULT_HEADER["HDUCLASS"] = "GADF"
DEFAULT_HEADER["ORIGIN"] = "CTA"
DEFAULT_HEADER["TELESCOP"] = "CTA-N"

wobble_offset = 0.4 * u.deg


def create_obs_index_hdu(filename_list, fits_dir, obs_index_file, overwrite):
    """
    Create the obs index table and write it to the given file.
    The Index table is created as per,
    http://gamma-astro-data-formats.readthedocs.io/en/latest/data_storage/obs_index/index.html

    Parameters
    ----------
    filename_list : list
        list of filenames of the fits files
    fits_dir : Path
        Path of the fits files
    obs_index_file : Path
        Path for the OBS index file
    overwrite : Bool
        Boolean to overwrite existing file
    """
    obs_index_tables = []

    # loop through the files
    for file in filename_list:
        filepath = fits_dir / file
        if filepath.is_file():
            try:
                hdu_list = fits.open(filepath)
                evt_hdr = hdu_list["EVENTS"].header
            except Exception:
                log.error(f"fits corrupted for file {file}")
                continue
        else:
            log.error(f"fits {file} doesn't exist")
            continue

        # Obs_table
        t_obs = {
            "OBS_ID": evt_hdr["OBS_ID"],
            "DATE-OBS": evt_hdr["DATE-OBS"],
            "TIME-OBS": evt_hdr["TIME-OBS"],
            "DATE-END": evt_hdr["DATE-END"],
            "TIME-END": evt_hdr["TIME-END"],
            "RA_PNT": evt_hdr["RA_PNT"] * u.deg,
            "DEC_PNT": evt_hdr["DEC_PNT"] * u.deg,
            "ZEN_PNT": (90 - float(evt_hdr["ALT_PNT"])) * u.deg,
            "ALT_PNT": evt_hdr["ALT_PNT"] * u.deg,
            "AZ_PNT": evt_hdr["AZ_PNT"] * u.deg,
            "RA_OBJ": evt_hdr["RA_OBJ"] * u.deg,
            "DEC_OBJ": evt_hdr["DEC_OBJ"] * u.deg,
            "TSTART": evt_hdr["TSTART"] * u.s,
            "TSTOP": evt_hdr["TSTOP"] * u.s,
            "ONTIME": evt_hdr["ONTIME"] * u.s,
            "TELAPSE": evt_hdr["TELAPSE"] * u.s,
            "LIVETIME": evt_hdr["LIVETIME"] * u.s,
            "DEADC": evt_hdr["DEADC"],
            "OBJECT": evt_hdr["OBJECT"],
            "OBS_MODE": evt_hdr["OBS_MODE"],
            "N_TELS": evt_hdr["N_TELS"],
            "TELLIST": evt_hdr["TELLIST"],
            "INSTRUME": evt_hdr["INSTRUME"],
        }
        obs_index_tables.append(t_obs)

    obs_index_table = QTable(obs_index_tables)

    obs_index_header = DEFAULT_HEADER.copy()
    obs_index_header["CREATED"] = Time.now().utc.iso
    obs_index_header["HDUCLAS1"] = "INDEX"
    obs_index_header["HDUCLAS2"] = "OBS"
    obs_index_header["INSTRUME"] = t_obs["INSTRUME"]
    obs_index_header["MJDREFI"] = evt_hdr["MJDREFI"]
    obs_index_header["MJDREFF"] = evt_hdr["MJDREFF"]

    obs_index = fits.BinTableHDU(
        obs_index_table, header=obs_index_header, name="OBS INDEX"
    )
    obs_index_list = fits.HDUList([fits.PrimaryHDU(), obs_index])
    obs_index_list.writeto(obs_index_file, overwrite=overwrite)


def create_hdu_index_hdu(
    filename_list,
    fits_dir,
    hdu_index_file,
    overwrite=False
):
    """
    Create the hdu index table and write it to the given file.
    The Index table is created as per,
    http://gamma-astro-data-formats.readthedocs.io/en/latest/data_storage/hdu_index/index.html

    Parameters
    ----------
    filename_list : list
        list of filenames of the fits files
    fits_dir : Path
        Path of the fits files
    hdu_index_file : Path
        Path for HDU index file
    overwrite : Bool
        Boolean to overwrite existing file
    """

    hdu_index_tables = []

    base_dir = os.path.commonpath(
        [
            hdu_index_file.parent.absolute().resolve(),
            fits_dir.absolute().resolve()
        ]
    )
    # loop through the files
    for file in filename_list:
        filepath = fits_dir / file
        if filepath.is_file():
            try:
                hdu_list = fits.open(filepath)
                evt_hdr = hdu_list["EVENTS"].header
                gti_hdr = hdu_list["GTI"].header
                pnt_hdr = hdu_list["POINTING"].header
            except Exception:
                log.error(f"fits corrupted for file {file}")
                continue
        else:
            log.error(f"fits {file} doesn't exist")
            continue

        # The column names for the table follows the scheme as shown in
        # https://gamma-astro-data-formats.readthedocs.io/en/latest/general/hduclass.html
        # Event list
        t_events = {
            "OBS_ID": evt_hdr["OBS_ID"],
            "HDU_TYPE": "events",
            "HDU_CLASS": "events",
            "FILE_DIR": str(os.path.relpath(fits_dir, hdu_index_file.parent)),
            "FILE_NAME": file,
            "HDU_NAME": "EVENTS",
            "SIZE": filepath.stat().st_size,
        }
        hdu_index_tables.append(t_events)

        # GTI
        t_gti = t_events.copy()

        t_gti["HDU_TYPE"] = "gti"
        t_gti["HDU_CLASS"] = "gti"
        t_gti["HDU_NAME"] = "GTI"

        hdu_index_tables.append(t_gti)

        # POINTING
        t_pnt = t_events.copy()

        t_pnt["HDU_TYPE"] = "pointing"
        t_pnt["HDU_CLASS"] = "pointing"
        t_pnt["HDU_NAME"] = "POINTING"

        hdu_index_tables.append(t_pnt)

        # Energy Dispersion
        try:
            edisp = hdu_list["ENERGY DISPERSION"]
            t_edisp = t_events.copy()

            t_edisp["HDU_TYPE"] = "edisp"
            t_edisp["HDU_CLASS"] = "edisp_2d"
            t_edisp["HDU_NAME"] = "ENERGY DISPERSION"

            hdu_index_tables.append(t_edisp)
        except KeyError:
            log.error(
                f"Run {t_events['OBS_ID']} does not contain HDU 'ENERGY DISPERSION'"
            )

        # Effective Area
        try:
            aeff = hdu_list["EFFECTIVE AREA"]
            t_aeff = t_events.copy()
            t_aeff["HDU_TYPE"] = "aeff"
            t_aeff["HDU_CLASS"] = "aeff_2d"
            t_aeff["HDU_NAME"] = "EFFECTIVE AREA"

            hdu_index_tables.append(t_aeff)
        except KeyError:
            log.error(
                f"Run {t_events['OBS_ID']} does not contain HDU 'EFFECTIVE AREA'"
            )

        # Background
        try:
            bkg = hdu_list["BACKGROUND"]
            t_bkg = t_events.copy()
            t_bkg["HDU_TYPE"] = "bkg"
            t_bkg["HDU_CLASS"] = "bkg_2d"
            t_bkg["HDU_NAME"] = "BACKGROUND"

            hdu_index_tables.append(t_bkg)
        except KeyError:
            log.error(
                f"Run {t_events['OBS_ID']} does not contain HDU 'BACKGROUND'"
            )

        # PSF
        try:
            psf = hdu_list["PSF"]
            t_psf = t_events.copy()
            t_psf["HDU_TYPE"] = "psf"
            t_psf["HDU_CLASS"] = "psf_table"
            t_psf["HDU_NAME"] = "PSF"

            hdu_index_tables.append(t_psf)
        except KeyError:
            log.error(
                f"Run {t_events['OBS_ID']} does not contain HDU 'PSF'"
            )

    hdu_index_table = Table(hdu_index_tables)

    hdu_index_header = DEFAULT_HEADER.copy()
    hdu_index_header["CREATED"] = Time.now().utc.iso
    hdu_index_header["HDUCLAS1"] = "INDEX"
    hdu_index_header["HDUCLAS2"] = "HDU"
    hdu_index_header["INSTRUME"] = evt_hdr["INSTRUME"]
    hdu_index_header["BASE_DIR"] = base_dir

    hdu_index = fits.BinTableHDU(
        hdu_index_table, header=hdu_index_header, name="HDU INDEX"
    )
    hdu_index_list = fits.HDUList([fits.PrimaryHDU(), hdu_index])
    hdu_index_list.writeto(hdu_index_file, overwrite=overwrite)


def create_event_list(
    data, run_number, source_name, source_pos, effective_time, elapsed_time
):
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
        Source_pos: Ra/Dec position of the source
                'astropy.coordinates.SkyCoord'
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
    time_utc = Time(data["dragon_time"], format="unix", scale="utc")
    t_start_iso = time_utc[0].to_value("iso", "date_hms")
    t_stop_iso = time_utc[-1].to_value("iso", "date_hms")
    date_obs = t_start_iso[:10]
    time_obs = t_start_iso[11:]
    date_end = t_stop_iso[:10]
    time_end = t_stop_iso[11:]

    MJDREF = Time("1970-01-01T00:00", scale="utc")

    # Position parameters
    reco_alt = data["reco_alt"]
    reco_az = data["reco_az"]
    pointing_alt = data["pointing_alt"]
    pointing_az = data["pointing_az"]

    reco_altaz = SkyCoord(
        alt=reco_alt, az=reco_az, frame=AltAz(obstime=time_utc, location=location)
    )
    pnt_icrs = SkyCoord(
        alt=pointing_alt[0],
        az=pointing_az[0],
        frame=AltAz(obstime=time_utc[0], location=location),
    ).transform_to(frame="icrs")

    with erfa_astrom.set(ErfaAstromInterpolator(30 * u.s)):
        reco_icrs = reco_altaz.transform_to(frame="icrs")

    # Observation modes
    source_pointing_diff = source_pos.separation(pnt_icrs)
    if np.around(source_pointing_diff, 1) == wobble_offset:
        mode = "WOBBLE"
    elif np.around(source_pointing_diff, 1) > 1 * u.deg:
        mode = "OFF"
    elif np.around(source_pointing_diff, 1) == 0.0 * u.deg:
        mode = "ON"
    else:
        # Nomenclature is to be worked out or have a separate way to mark mispointings
        mode = "UNDETERMINED"

    log.info(
        "Source pointing difference with camera pointing"
        f" is {source_pointing_diff:.3f}"
    )

    event_table = QTable(
        {
            "EVENT_ID": data["event_id"],
            "TIME": data["dragon_time"],
            "RA": reco_icrs.ra.to(u.deg),
            "DEC": reco_icrs.dec.to(u.deg),
            "ENERGY": data["reco_energy"],
            # Optional columns
            "GAMMANESS": data["gh_score"],
            "MULTIP": u.Quantity(np.repeat(len(tel_list), len(data)), dtype=int),
            "GLON": reco_icrs.galactic.l.to(u.deg),
            "GLAT": reco_icrs.galactic.b.to(u.deg),
            "ALT": reco_alt.to(u.deg),
            "AZ": reco_az.to(u.deg),
        }
    )
    gti_table = QTable(
        {
            "START": u.Quantity(t_start, unit=u.s, ndmin=1),
            "STOP": u.Quantity(t_stop, unit=u.s, ndmin=1),
        }
    )
    pnt_table = QTable(
        {
            "TIME": u.Quantity(t_start, unit=u.s, ndmin=1),
            "RA_PNT": u.Quantity(pnt_icrs.ra.to(u.deg), ndmin=1),
            "DEC_PNT": u.Quantity(pnt_icrs.dec.to(u.deg), ndmin=1),
            "ALT_PNT": u.Quantity(pointing_alt[0].to(u.deg), ndmin=1),
            "AZ_PNT": u.Quantity(pointing_az[0].to(u.deg), ndmin=1),
        }
    )

    # Adding the meta data
    # Comments can be added later for relevant metadata
    # Event table metadata
    ev_header = DEFAULT_HEADER.copy()
    ev_header["CREATED"] = Time.now().utc.iso
    ev_header["HDUCLAS1"] = "EVENTS"

    ev_header["OBS_ID"] = run_number

    ev_header["DATE-OBS"] = date_obs
    ev_header["TIME-OBS"] = time_obs
    ev_header["DATE-END"] = date_end
    ev_header["TIME-END"] = time_end
    ev_header["TSTART"] = t_start
    ev_header["TSTOP"] = t_stop
    ev_header["MJDREFI"] = int(MJDREF.mjd)
    ev_header["MJDREFF"] = MJDREF.mjd - int(MJDREF.mjd)
    ev_header["TIMEUNIT"] = "s"
    ev_header["TIMESYS"] = "UTC"
    ev_header["TIMEREF"] = "TOPOCENTER"
    ev_header["ONTIME"] = elapsed_time
    ev_header["TELAPSE"] = t_stop - t_start
    ev_header["DEADC"] = effective_time / elapsed_time
    ev_header["LIVETIME"] = effective_time

    ev_header["OBJECT"] = source_name
    ev_header["OBS_MODE"] = mode

    ev_header["N_TELS"] = len(tel_list)
    ev_header["TELLIST"] = "LST-" + " ".join(map(str, tel_list))
    ev_header["INSTRUME"] = f"{ev_header['TELLIST']}"

    ev_header["RA_PNT"] = pnt_icrs.ra.to_value()
    ev_header["DEC_PNT"] = pnt_icrs.dec.to_value()
    ev_header["ALT_PNT"] = data["pointing_alt"].mean().to_value(u.deg)
    ev_header["AZ_PNT"] = data["pointing_az"].mean().to_value(u.deg)
    ev_header["RA_OBJ"] = source_pos.ra.to_value()
    ev_header["DEC_OBJ"] = source_pos.dec.to_value()
    ev_header["FOVALIGN"] = "RADEC"

    # GTI table metadata
    gti_header = DEFAULT_HEADER.copy()
    gti_header["CREATED"] = Time.now().utc.iso
    gti_header["HDUCLAS1"] = "GTI"

    gti_header["OBS_ID"] = run_number
    gti_header["MJDREFI"] = ev_header["MJDREFI"]
    gti_header["MJDREFF"] = ev_header["MJDREFF"]
    gti_header["TIMESYS"] = ev_header["TIMESYS"]
    gti_header["TIMEUNIT"] = ev_header["TIMEUNIT"]
    gti_header["TIMEREF"] = ev_header["TIMEREF"]

    # Pointing table metadata
    pnt_header = DEFAULT_HEADER.copy()
    pnt_header["CREATED"] = Time.now().utc.iso
    pnt_header["HDUCLAS1"] = "POINTING"

    pnt_header["OBS_ID"] = run_number
    pnt_header["MJDREFI"] = ev_header["MJDREFI"]
    pnt_header["MJDREFF"] = ev_header["MJDREFF"]
    pnt_header["TIMEUNIT"] = ev_header["TIMEUNIT"]
    pnt_header["TIMESYS"] = ev_header["TIMESYS"]
    pnt_header["OBSGEO-L"] = (
        location.lon.to_value(u.deg),
        "Geographic longitude of telescope (deg)",
    )
    pnt_header["OBSGEO-B"] = (
        location.lat.to_value(u.deg),
        "Geographic latitude of telescope (deg)",
    )
    pnt_header["OBSGEO-H"] = (
        round(location.height.to_value(u.m), 2),
        "Geographic latitude of telescope (m)",
    )

    pnt_header["TIMEREF"] = ev_header["TIMEREF"]

    # Create HDUs
    event = fits.BinTableHDU(event_table, header=ev_header, name="EVENTS")
    gti = fits.BinTableHDU(gti_table, header=gti_header, name="GTI")
    pointing = fits.BinTableHDU(pnt_table, header=pnt_header, name="POINTING")

    return event, gti, pointing
