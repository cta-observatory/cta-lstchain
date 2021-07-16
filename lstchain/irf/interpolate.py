import numpy as np
from pathlib import Path

import astropy.units as u
from astropy.table import Table, QTable
from astropy.io import fits

# from lstchain.io.io import get_geomagnetic_delta

from pyirf.io.gadf import (
    create_aeff2d_hdu,
    create_energy_dispersion_hdu,
)
from pyirf.interpolation import (
    interpolate_effective_area_per_energy_and_fov,
    interpolate_energy_dispersion
)
from scipy.spatial import Delaunay, distance


def duplicate_irfs(irfs):
    """
    Duplicate IRFs which have azimuth pointing as 0 deg, with the metadata
    value of the azimuth pointing as 360 deg, for facilitating interpolation.
    """
    new_irfs = []
    for i, irf in enumerate(irfs):
        new_irfs.append(irf)

        f = fits.open(irf)[1]
        if round(u.Quantity(f.header["AZ_PNT"]).to_value(u.deg), 2) == 0.0:
            # Copy the IRF to create the new IRF
            f2 = fits.open(irf).copy()

            # Get the name of the new IRF file
            p = Path(irf)
            p2 = p.with_name(
                p.name.replace(".fits.gz", "_duplicate_az359.fits.gz")
            )

            # Change the header value of AZ_PNT for each BinTableHDU
            for h in f2[1:]:
                h.header["AZ_PNT"] = "359.99 deg" # Check for the threshold value

            # Save the duplicated IRF with new name
            f2.writeto(p2, overwrite="True")
            new_irfs.append(p2)

    return new_irfs


def interp_params(params_list, data, use_b_delta):
    """
    From a given list of angular parameters, to be used for interpolation,
    take values from a given data table/dict.
    If AZ_PNT has to be replaced by B_DELTA value, take the boolean flag.

    Returns the neccessary values with applied functions as need be for the
    interpolation, for each parameter as a list.
    """
    mc_pars = []
    if "ZEN_PNT" in params_list:
        mc_pars.append(
            np.cos(u.Quantity(data["ZEN_PNT"]).to_value(u.rad))
        )
    if "AZ_PNT" in params_list or "B_DELTA" in params_list:
        if use_b_delta:
            #print('using B delta instead of az pnt')
            mc_pars.append(
                np.sin(
                    u.Quantity(data["B_DELTA"]).to_value(u.rad)
                )
            )
        else:
            # Using a sine wave approximation for azimuth dependence,
            # with using half angles. This is an ok approximation for
            # zenith less than 40 deg
            mc_pars.append(
                np.sin(
                    u.Quantity(data["AZ_PNT"]).to_value(u.rad)
                )
            )
    return mc_pars


def check_in_delaunay_triangle(irfs, data_params, use_b_delta):
    """
    From a given list of IRFs as grid points used for interpolation, retrieve
    the Delaunay triangulation list of IRFs, where the simplex includes the
    target points in data_params.

    If the target point does not exist inside the simplex, the IRF
    corresponding to the nearest grid point to the target value.

    If the list of given IRFs are not enough for calculating the Delaunay
    triangulation, an empty list is returned.

    Parameters
    ------------
    irfs: List of IRFs to check for Delaunay triangulation.
        'List'
    data_pars: Dict of arrays of range of parameters of the observed data
        in the event list, to check for interpolation.
        'Dict'
    use_b_delta: Bool to replace the azimuth angle with the angle between
        geomagnetic field with the shower axis.
        'Bool'

    Returns
    ----------
    irf_list: Revised list of IRFs after the checks.
        'List'
    """
    #print(data_params)
    if not use_b_delta:
        data_pars = [*data_params.keys()]
    else:
        # Exclude AZ_PNT as target interpolation parameter
        d = data_params.copy()
        d.pop("AZ_PNT", None)
        data_pars = [*d.keys()]

    new_irfs = []

    mc_params = np.empty((len(irfs), len(data_pars)))

    for i, irf in enumerate(irfs):
        f = fits.open(irf)[1].header

        mc_pars = interp_params(data_pars, f, use_b_delta)
        #print(mc_pars, use_b_delta)
        mc_params[i, :] = np.array(mc_pars)

    data_val = interp_params(data_pars, data_params, use_b_delta)
    #print(mc_params, data_val)
    try:
        tri = Delaunay(mc_params)
    except ValueError:
        print('Not enough grid values for Delaunay triangulation')
        return new_irfs
    ## Check list
    target_in_simplex = tri.find_simplex(data_val)

    if target_in_simplex == -1:
        # The target values are not contained in any Delaunay triangle formed
        # by the paramters of the list of IRFs provided.
        # So just include the IRF with the closest parameter values
        # to the target values
        index = distance.cdist([data_val], mc_params).argmin()
        print("Target value is outside interpolation. Using the nearest IRF.")
        new_irfs.append(irfs[index])
    else:
        # Just select the IRFs that are needed for the Delaunay triangulation
        for i in tri.simplices[target_in_simplex]:
            new_irfs.append(irfs[i])

    return new_irfs


def compare_irfs(irfs):
    """
    Compare the given list of IRFs with various selection cuts, data binning
    and relevant metadata values.
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


def interpolate_irf(irfs, data_pars, interp_method, use_b_delta):
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
    interp_method: Method of interpolation to be used by
        scipy.interpolate.griddata. Values can be "linear", "nearest", "cubic"
        'Str'
    use_b_delta: Bool to replace the azimuth angle with the angle between
        geomagnetic field with the shower axis.
        'Bool'

    Returns
    ------------
    irf_interp: Final interpolated IRF
        'astropy.io.fits'
    """

    # Gather the parameters to use for interpolation

    if not use_b_delta:
        params = [*data_pars.keys()]
    else:
        # Exclude AZ_PNT as target interpolation parameter
        d = data_pars.copy()
        d.pop("AZ_PNT", None)
        params = [*d.keys()]
    n_grid = len(irfs)
    irf_pars = np.empty((n_grid, len(params)))
    #mc_params = np.empty((len(params), len(irfs)))
    interp_pars = list()

    extra_keys = [
        "TELESCOP", "INSTRUME", "FOVALIGN",
        "GH_CUT", "G_OFFSET", "RAD_MAX", "B_TOTAL", "B_INC"
        ]
    main_headers = fits.open(irfs[0])[1].header

    if main_headers["HDUCLAS3"] == "POINT-LIKE":
        point_like = True
    else:
        point_like = False

    extra_headers = dict(
        (k, main_headers[k]) for k in extra_keys if k in main_headers
    )

    # The interpolation over angular parameters will be carried in radians
    """for i, par in enumerate(params):
        if par == "AZ_PNT":
            par = "B_DELTA"
            b_delta = get_geomagnetic_delta(
                u.Quantity(main_headers["B_TOTAL"]).to_value(u.uT),
                u.Quantity(main_headers["B_INC"]).to_value(u.rad),
                data_pars["ZEN_PNT"].to_value(u.rad),
                data_pars["AZ_PNT"].to_value(u.rad),
            )
            interp_pars.append(b_delta)

        mc_params[i, :] = np.array(
            [
                u.Quantity(
                    fits.open(irf)[1].header[par]
                ).to_value(u.rad) for irf in irfs
            ]
        )
        if par == "ZEN_PNT":
            interp_pars.append(np.cos(data_pars[par].to_value(u.rad)))
    """
    interp_pars = interp_params(params, data_pars, use_b_delta)
    # Keep interp_pars as a tuple to keep the right dimensions in interpolation
    interp_pars = tuple(interp_pars)

    # Read the IRFs into lists and extract the necessary columns
    effarea_list = load_irf_grid(
        irfs, extname="EFFECTIVE AREA", interp_col="EFFAREA"
    )
    edisp_list = load_irf_grid(
        irfs, extname="ENERGY DISPERSION", interp_col="MATRIX"
    )
    temp_e = QTable.read(irfs[0], hdu="ENERGY DISPERSION")

    # Check the units as well
    e_true = np.append(temp_e["ENERG_LO"][0], temp_e["ENERG_HI"][0][-1])
    e_migra = np.append(temp_e["MIGRA_LO"][0], temp_e["MIGRA_HI"][0][-1])
    fov_off = np.append(temp_e["THETA_LO"][0], temp_e["THETA_HI"][0][-1])

    for i in np.arange(n_grid):
        """for j, par in enumerate(params):
            if par == "ZEN_PNT":
                irf_pars[i, j] = np.cos(mc_params[j][i])
            else:
                irf_pars[i, j] = np.sin(mc_params[j][i])"""

        f = fits.open(irfs[i])[1].header
        mc_pars = interp_params(params, f, use_b_delta)
        irf_pars[i, :] = np.array(mc_pars)

        i += 1

    for par in data_pars.keys():
        #print(par)
        extra_headers[par] = str(data_pars[par].to(u.deg))
    #extra_headers["B_DELTA"] = str(b_delta * 180/np.pi * u.deg)

    ## Check and compare cuts applied in each IRF using pyirf.cuts.compare_irf_cuts
    aeff_interp = interpolate_effective_area_per_energy_and_fov(
        effarea_list, irf_pars, interp_pars, method=interp_method
    )

    aeff_hdu_interp = create_aeff2d_hdu(
        aeff_interp.T,
        true_energy_bins=e_true,
        fov_offset_bins=fov_off,
        point_like=point_like,
        extname="EFFECTIVE AREA",
        **extra_headers,
    )

    edisp_interp = interpolate_energy_dispersion(
        edisp_list, irf_pars, interp_pars, method=interp_method
    )

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
