import numpy as np
import logging

import astropy.units as u
from astropy.table import Table, QTable
from astropy.io import fits

from pyirf.io.gadf import (
    create_aeff2d_hdu,
    create_energy_dispersion_hdu,
)
from pyirf.interpolation import (
    interpolate_effective_area_per_energy_and_fov,
    interpolate_energy_dispersion
)


def compare_irfs(irfs):
    """
    Compare the given IRFs with various selection cuts/ data binning/ metadata
    """
    bin_sim = False
    meta_sim = False

    params = []
    meta = []
    select_meta = ["HDUCLAS3", "INSTRUME", "GH_CUT", "RAD_MAX", "G_OFFSET"]

    for i, irf in enumerate(irfs):
        e_table = Table.read(irf, hdu="ENERGY DISPERSION")
        for j, col in enumerate(e_table.columns[:-1]):
            params.append(e_table[col].quantity[0])
        for m in select_meta:
            if m in e_table.meta:
                meta.append(e_table.meta[m])

    meta_sim = (np.unique(meta) == meta[:int(len(meta)/len(irfs))]).all()

    for i in np.arange(6):
        a = [params[6*j + i] for j in np.arange(len(irfs))]
        bim_sim = (np.unique(a) == params[i]).all()

    return bin_sim * meta_sim

def load_irf_grid(irfs, extname, interp_col):
    """
    From a given list of IRFs, load the list of IRF data values that can be
    interpolated (Effective Area and Energy Dispersion for now) and
    check the other parameters to be in the same bins and the selection cuts
    used while creating the IRFs

    Parameters
    ------------
    irfs: List of IRFs to use to interpolate
        List
    extname: Name of the IRF to be extracted
        Str
    interp_col: Name of the column whose values are to be interpolated
        Str

    Returns
    ----------
    interp_params: List of columns of the IRF from each file
        'numpy.stack'
    """
    interp_params = []
    for irf in irfs:
        interp_params.append(
            QTable.read(irf, hdu=extname)[interp_col][0].T
        )

    return np.stack(interp_params)

def interpolate_irf(irfs, data_pars):
    """
    Using pyirf functions with a list of IRFs and parameters to compare with
    data, to interpolate over, to get the closest match

    For now only Effective Area and Energy Dispersion is interpolated over.
    For the rest of IRFs, we can include the ones from IRF closest to data,
    over the given parameters.

    For now only binning in zenith is done.

    Parameters
    ------------
    irfs: List of IRFs to use to interpolate
        List
    data_pars: Dict of arrays of range of parameters of the observed data
        in the event list, to check for interpolation.
        'Dict'

    Returns
    ------------
    irf_interp: Final interpolated IRF
        'astropy.io.fits'
    """

    # Gather the parameters to use for interpolation
    params = list(data_pars.keys())
    n_grid = len(irfs) * len(params)
    irf_pars = np.empty((n_grid, len(params)))
    mc_params = np.empty((len(params), len(irfs)))

    for i, par in enumerate(params):
        # Assuming that the header values have ' deg' after the float value
        mc_params[i, :] = np.array(
            [float(fits.open(irf)[1].header[par][:-4]) for irf in irfs]
        )
    print(mc_params)

    extra_keys = ["TELESCOP", "INSTRUME", "FOVALIGN", "GH_CUT", "G_OFFSET", "RAD_MAX"]
    main_headers = fits.open(irfs[0])[1].header

    if main_headers["HDUCLAS3"] == "POINT-LIKE":
        point_like = True
    else:
        point_like = False

    extra_headers = dict((k, main_headers[k]) for k in extra_keys if k in main_headers)

    # Read the IRFs into lists and extract the necessary columns
    effarea_list = load_irf_grid(irfs, extname="EFFECTIVE AREA", interp_col="EFFAREA")
    edisp_list = load_irf_grid(irfs, extname="ENERGY DISPERSION", interp_col="MATRIX")
    temp_e = QTable.read(irfs[0], extname="ENERGY DISPERSION")
    e_true = temp_e["ENERG_LO"].quantity[0].insert(
        len(temp_e["ENERG_LO"][0]), temp_e["ENERG_HI"].quantity[0][-1]
    )
    e_migra = temp_e["MIGRA_LO"].quantity[0].insert(
        len(temp_e["MIGRA_LO"][0]), temp_e["MIGRA_HI"].quantity[0][-1]
    )
    fov_off = temp_e["THETA_LO"].quantity[0].insert(
        len(temp_e["THETA_LO"][0]), temp_e["THETA_HI"].quantity[0][-1]
    )

    interp_pars = []

    #for k in data_pars.keys():
    i_grid = 0
    for i in np.arange(len(params)):
        for j in np.arange(len(irfs)):
            if params[i] == "ZEN_PNT":
                #Assuming the first column of params is ZEN_PNT
                irf_pars[i_grid, ] = np.array(np.cos(j * np.pi/180.))
                interp_pars.append(np.array(np.cos(data_pars[i][0] * np.pi/180.)))
            else:
                irf_pars[i_grid, i+j] = [mc_params[i][j]]
                interp_pars.append(data_pars[i])
            i_grid += 1

    print(irf_pars)
    ## Final desired interpolated value
    ## Zenith of data - start or mean?
    extra_headers["ZEN_PNT"] = str(data_pars["ZEN_PNT"][0] * u.deg)

    ## Check and compare cuts applied in each IRF using pyirf.cuts.compare_irf_cuts

    aeff_interp = interpolate_effective_area_per_energy_and_fov(
        effarea_list, irf_pars, interp_pars
    )
    aeff_hdu_interp = create_aeff2d_hdu(
        aeff_interp.T,
        e_true,
        fov_off,
        point_like=point_like,
        extname="EFFECTIVE AREA",
        **extra_headers,
    )

    edisp_interp = interpolate_energy_dispersion(edisp_list, irf_pars, interp_pars)

    edisp_hdu_interp = create_energy_dispersion_hdu(
        edisp_interp,
        e_true,
        e_migra,
        fov_off,
        point_like=point_like,
        extname="ENERGY DISPERSION",
        **extra_headers,
    )

    irf_interp = [fits.PrimaryHDU(), ]
    irf_interp.append(aeff_hdu_interp)
    irf_interp.append(edisp_hdu_interp)

    ## For Background, PSF?
    return irf_interp
