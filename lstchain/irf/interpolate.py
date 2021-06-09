import numpy as np

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

    # For fixed gammaness/theta cuts
    select_meta = ["HDUCLAS3", "INSTRUME", "GH_CUT", "RAD_MAX", "G_OFFSET"]
    cols = Table.read(irfs[0], hdu="ENERGY DISPERSION").columns[:-1]

    for i, irf in enumerate(irfs):
        e_table = Table.read(irf, hdu="ENERGY DISPERSION")
        for j, col in enumerate(cols):
            params.append(e_table[col].quantity[0])
        for m in select_meta:
            if m in e_table.meta:
                meta.append(e_table.meta[m])

    # Comparing metadata
    meta_2, meta_ind = np.unique(meta, return_index=True)

    if len(meta_2) == int(len(meta)/len(irfs)):
        if (meta_2[np.argsort(meta_ind)] == meta[:int(len(meta)/len(irfs))]).all():
            meta_sim = True

    # Comparing other paramater axes in IRFs
    for i in np.arange(len(cols)):
        a = [params[len(cols)*j + i] for j in np.arange(len(irfs))]
        a2, a_ind = np.unique(a, return_index=True)
        if len(a2) == len(params[i]):
            if (a2[np.argsort(a_ind)] == params[i].value).all():
                bin_sim = True

    return bin_sim * meta_sim


def load_irf_grid(irfs, extname, interp_col):
    """
    From a given list of IRFs, load the list of IRF data values that can be
    interpolated (Effective Area and Energy Dispersion for now)

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
    irf_list: List of columns of the IRF from each file
        'numpy.stack'
    """
    irf_list = []
    for irf in irfs:
        irf_list.append(
            QTable.read(irf, hdu=extname)[interp_col][0].T
        )
    return np.stack(irf_list)


def interpolate_irf(irfs, data_pars):
    """
    Using pyirf functions with a list of IRFs and parameters to compare with
    data, to interpolate over, to get the closest match

    For now only Effective Area and Energy Dispersion is interpolated over.

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
    n_grid = len(irfs)
    irf_pars = np.empty((n_grid, len(params)))
    mc_params = np.empty((len(params), len(irfs)))
    interp_pars = list()

    for i, par in enumerate(params):
        # Assuming that the header values have ' deg' after the float value
        mc_params[i, :] = np.array(
            [float(fits.open(irf)[1].header[par][:-4]) for irf in irfs]
        )
        # Modifying parameter values to right ones, for interpolation
        if par == "ZEN_PNT":
            interp_pars.append(np.cos(data_pars[par] * np.pi/180.))
        else:
            interp_pars.append(data_pars[par])
    # Keep interp_pars as a tuple to keep the right dimensions in interpolation
    interp_pars = tuple(interp_pars)

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
    temp_e = QTable.read(irfs[0], hdu="ENERGY DISPERSION")

    # Check the units as well
    e_true = np.append(temp_e["ENERG_LO"][0], temp_e["ENERG_HI"][0][-1])
    e_migra = np.append(temp_e["MIGRA_LO"][0], temp_e["MIGRA_HI"][0][-1])
    fov_off = np.append(temp_e["THETA_LO"][0], temp_e["THETA_HI"][0][-1])

    i_grid = 0
    for i in np.arange(n_grid):
        for j, par in enumerate(params):
            if par == "ZEN_PNT":
                irf_pars[i_grid, j] = np.cos(mc_params[j][i] * np.pi/180.)
            else:
                irf_pars[i_grid, j] = mc_params[j][i]
        i_grid += 1

    for par in params:
        extra_headers[par] = str(data_pars[par] * u.deg)

    ## Check and compare cuts applied in each IRF using pyirf.cuts.compare_irf_cuts
    aeff_interp = interpolate_effective_area_per_energy_and_fov(
        effarea_list, irf_pars, interp_pars
    )

    aeff_hdu_interp = create_aeff2d_hdu(
        aeff_interp.T,
        true_energy_bins=e_true,
        fov_offset_bins=fov_off,
        point_like=point_like,
        extname="EFFECTIVE AREA",
        **extra_headers,
    )

    edisp_interp = interpolate_energy_dispersion(edisp_list, irf_pars, interp_pars)

    edisp_hdu_interp = create_energy_dispersion_hdu(
        edisp_interp,
        true_energy_bins=e_true,
        migration_bins=e_migra,
        fov_offset_bins=fov_off,
        point_like=point_like,
        extname="ENERGY DISPERSION",
        **extra_headers,
    )

    irf_interp = fits.HDUList([fits.PrimaryHDU(), ])
    irf_interp.append(aeff_hdu_interp)
    irf_interp.append(edisp_hdu_interp)

    return irf_interp
