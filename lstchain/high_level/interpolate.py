import numpy as np
import logging
import astropy.units as u
from astropy.table import Table, QTable
from astropy.io import fits
from astropy.time import Time

from lstchain.__init__ import __version__

from pyirf.io.gadf import (
    create_aeff2d_hdu,
    create_energy_dispersion_hdu,
    create_psf_table_hdu,
)
from pyirf.interpolation import (
    interpolate_effective_area_per_energy_and_fov,
    interpolate_energy_dispersion,
    interpolate_psf_table, # interpolate_rad_max_table
)
from scipy.spatial import Delaunay, distance
from scipy.interpolate import griddata

log = logging.getLogger(__name__)


def interp_params(params_list, data):
    """
    From a given list of angular parameters, to be used for interpolation,
    take values from a given data table/dict.

    Returns the neccessary values with applied functions as need be for the
    interpolation, for each parameter as a list.
    """
    mc_pars = []
    if "ZEN_PNT" in params_list:
        mc_pars.append(
            np.cos(u.Quantity(data["ZEN_PNT"]).to_value(u.rad))
        )

    if "B_DELTA" in params_list:
        mc_pars.append(
            np.sin(
                u.Quantity(data["B_DELTA"]).to_value(u.rad)
            )
        )

    return mc_pars


def check_in_delaunay_triangle(irfs, data_params):
    """
    From a given list of IRFs as grid points used for interpolation, retrieve
    the Delaunay triangulation list of IRFs, where the simplex includes the
    target points in data_params.

    If the target point does not exist inside the simplex, the IRF
    corresponding to the nearest grid point to the target value.

    If the list of given IRFs are not enough for calculating the Delaunay
    triangulation, an empty list is returned.

    Parameters
    ----------
    irfs: List of IRFs to check for Delaunay triangulation.
        'List'
    data_pars: Dict of arrays of range of parameters of the observed data
        in the event list, to check for interpolation.
        'Dict'

    Returns
    -------
    irf_list: Revised list of IRFs after the checks.
        'List'
    """
    # Exclude AZ_PNT as target interpolation parameter
    d = data_params.copy()
    d.pop("AZ_PNT", None)
    data_pars = [*d.keys()]

    new_irfs = []
    mc_params = np.empty((len(irfs), len(data_pars)))

    for i, irf in enumerate(irfs):
        f = fits.open(irf)[1].header

        mc_pars = interp_params(data_pars, f)
        mc_params[i, :] = np.array(mc_pars)

    data_val = interp_params(data_pars, data_params)

    try:
        tri = Delaunay(mc_params)
    except ValueError:
        print('Not enough grid values for Delaunay triangulation')
        return new_irfs

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
    Compare the given list of IRFs with data binning and relevant
    metadata values.

    Parameters
    ----------
    irfs: List of IRFs to compare
        'List'
    Returns
    -------
    : Boolean indicating whether the given list of IRFs are comparable
    """
    bin_bool = False
    meta_bool = True

    params = []

    # Collect the list of metadata and column names to compare the values/data
    select_meta = ["HDUCLAS3", "INSTRUME", "G_OFFSET"]
    try:
        h = Table.read(irfs[0], hdu="EFFECTIVE AREA")
        h_meta = h.meta
        cols = h.columns[:-1]

        energy_dependent_gh = "GH_EFF" in h_meta
        point_like = h_meta["HDUCLAS3"] == "POINT-LIKE"
        energy_dependent_theta = "TH_CONT" in h_meta
        source_dep = "AL_CUT" in h_meta

        if energy_dependent_gh:
            select_meta.append("GH_EFF")
        else:
            select_meta.append("GH_CUT")
        if point_like:
            if energy_dependent_theta:
                select_meta.append("TH_CONT")
            else:
                if source_dep:
                    select_meta.append("AL_CUT")
                else:
                    select_meta.append("RAD_MAX")
    except:
        print(f"Effective Area is not present in {irfs[0]}")

    # Comparing the metadata information
    for k in select_meta:
        meta = [
            Table.read(i, hdu="ENERGY DISPERSION").meta[k] for i in irfs
        ]
        m = np.unique(meta)
        if len(m) != 1:
            meta_bool *= False

    print(f"The metadata are{' ' if meta_bool else ' not '}comparable")
    for irf in irfs:
        e_table = Table.read(irf, hdu="ENERGY DISPERSION")
        for col in cols:
            params.append(e_table[col].quantity[0])

    # Comparing other paramater columns in IRFs
    for i in np.arange(len(cols)):
        a = [params[len(cols)*j + i] for j in np.arange(len(irfs))]
        a2, a_ind = np.unique(a, return_index=True)

        if len(a2) == len(params[i]):
            if (a2[np.argsort(a_ind)] == params[i].value).all():
                bin_bool = True
    print(
        "The other parameter axes data "
        f"are{' ' if bin_bool else ' not '}comparable"
    )

    return bin_bool * meta_bool


def load_irf_grid(irfs, extname, interp_col, gadf_irf=True):
    """
    From a given list of IRFs, load the list of IRF data values that can be
    interpolated. For GH_CUTS which is not GADF approved, the HDU is stored
    differently and so requires a different way of loading the data.

    Parameters
    ----------
    irfs: List of IRFs to use to interpolate
        List
    extname: Name of the IRF to be extracted
        Str
    interp_col: Name of the column whose values are to be interpolated
        Str
    gadf_irf: IRF being as per GADF standard or custom
        Bool

    Returns
    -------
    irf_list: List of columns of the IRF from each file
        'numpy.stack'
    """
    irf_list = []
    for irf in irfs:
        if gadf_irf:
            irf_list.append(
                QTable.read(irf, hdu=extname)[interp_col][0].T
            )
        else:
            irf_list.append(
                QTable.read(irf, hdu=extname)[interp_col]
            )
    return np.stack(irf_list)


def interpolate_gh_table(
    gh_cuts, grid_points, target_point, method="linear",
):
    """
    Interpolates a grid of GH CUTS tables to a target-point.
    Wrapper around scipy.interpolate.griddata [1].
    Parameters
    ----------
    gh_cuts: numpy.ndarray, shape=(N, M, ...)
        Gammaness-cuts for all combinations of grid-points, energy and fov_offset.
        Shape (N:n_grid_points, M:n_energy_bins, n_fov_offset_bins)
    grid_points: numpy.ndarray, shape=(N, O)
        Array of the N O-dimensional morphing parameter values corresponding to the N input templates.
    target_point: numpy.ndarray, shape=(O)
        Value for which the interpolation is performed (target point)
    method: 'linear’, ‘nearest’, ‘cubic’
        Interpolation method for scipy.interpolate.griddata [1]. Defaults to 'linear'.
    Returns
    -------
    gh_cuts_interp: numpy.ndarray, shape=(1, M, ...)
        Gammaness-cuts for the target grid-point, shape (1, M:n_energy_bins, n_fov_offset_bins)
    """
    return griddata(grid_points, gh_cuts, target_point, method=method)


def interpolate_irf(irfs, data_pars, interp_method="linear"):
    """
    Using pyirf functions with a list of IRFs and parameters to compare with
    data, to interpolate over, to get the closest match

    For now only Effective Area and Energy Dispersion is interpolated over.

    Parameters
    ----------
    irfs: List of IRFs to use to interpolate
        List
    data_pars: Dict of arrays of range of parameters of the observed data
        in the event list, to check for interpolation.
        'Dict'
    interp_method: Method of interpolation to be used by
        scipy.interpolate.griddata. Values can be "linear", "nearest", "cubic"
        'Str'

    Returns
    -------
    irf_interp: Final interpolated IRF
        'astropy.io.fits'
    """

    # Gather the parameters to use for interpolation

    # Exclude AZ_PNT as target interpolation parameter - Hard-coded
    d = data_pars.copy()
    d.pop("AZ_PNT", None)
    params = [*d.keys()]
    n_grid = len(irfs)
    irf_pars = np.empty((n_grid, len(params)))
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

    # Update headers to be added to the final IRFs
    extra_headers = dict(
        (k, main_headers[k]) for k in extra_keys if k in main_headers
    )
    for par in data_pars.keys():
        extra_headers[par] = str(data_pars[par].to(u.deg))

    for i in np.arange(n_grid):
        f = fits.open(irfs[i])[1].header
        mc_pars = interp_params(params, f)
        irf_pars[i, :] = np.array(mc_pars)

    interp_pars = interp_params(params, data_pars)
    # Keep interp_pars as a tuple to keep the right dimensions in interpolation
    interp_pars = tuple(interp_pars)
    irf_interp = fits.HDUList([fits.PrimaryHDU(), ])

    # Read select IRFs into lists and extract the necessary columns
    hdus_interp = fits.open(irfs[0])

    try:
        hdus_interp["EFFECTIVE AREA"]
        effarea_list = load_irf_grid(
            irfs, extname="EFFECTIVE AREA", interp_col="EFFAREA"
        )

        temp_irf = QTable.read(irfs[0], hdu="EFFECTIVE AREA")
        e_true = np.append(temp_irf["ENERG_LO"][0], temp_irf["ENERG_HI"][0][-1])
        fov_off = np.append(temp_irf["THETA_LO"][0], temp_irf["THETA_HI"][0][-1])

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

        irf_interp.append(aeff_hdu_interp)

    except KeyError:
        log.error("Effective Area not present for IRF interpolation")

    try:
        hdus_interp["ENERGY DISPERSION"]
        edisp_list = load_irf_grid(
            irfs, extname="ENERGY DISPERSION", interp_col="MATRIX"
        )
        temp_irf = QTable.read(irfs[0], hdu="ENERGY DISPERSION")

        # Check the units as well
        e_true = np.append(temp_irf["ENERG_LO"][0], temp_irf["ENERG_HI"][0][-1])
        e_migra = np.append(temp_irf["MIGRA_LO"][0], temp_irf["MIGRA_HI"][0][-1])
        fov_off = np.append(temp_irf["THETA_LO"][0], temp_irf["THETA_HI"][0][-1])

        edisp_interp = interpolate_energy_dispersion(
            e_migra,
            edisp_list,
            irf_pars,
            interp_pars,
            quantile_resolution=1e-3
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

        irf_interp.append(edisp_hdu_interp)

    except KeyError:
        log.error("Energy Dispersion not present for IRF interpolation")

    try:
        hdus_interp["GH_CUTS"]
        gh_cuts_list = load_irf_grid(
            irfs, extname="GH_CUTS", interp_col="cut", gadf_irf=False
        )

        temp_irf = QTable.read(irfs[0], hdu="GH_CUTS")

        gh_cut_interp = interpolate_gh_table(
            gh_cuts_list, irf_pars, interp_pars, method=interp_method
        )

        gh_header = fits.Header()
        gh_header["CREATOR"] = f"lstchain v{__version__}"
        gh_header["DATE"] = Time.now().utc.iso

        for k, v in extra_headers.items():
            gh_header[k] = v

        temp_irf["cut"] = gh_cut_interp

        gh_cut_hdu_interp = fits.BinTableHDU(
            temp_irf, header=gh_header, name="GH_CUTS"
        )

        irf_interp.append(gh_cut_hdu_interp)

    except KeyError:
        log.error("GH CUTS not present for IRF interpolation")

    """
    try:
        hdus_interp["RAD_MAX"]
        radmax_list = load_irf_grid(
            irfs, extname="RAD_MAX", interp_col="RAD_MAX"
        )
        temp_irf = QTable.read(irfs[0], hdu="RAD_MAX")

        rad_max_interp = interpolate_rad_max_table(
            radmax_list, irf_pars, interp_pars, method=interp_method
        )

        temp_irf["RAD_MAX"] = rad_max_interp.T[np.newaxis, ...] * u.deg

        radmax_header = fits.Header()
        radmax_header["CREATOR"] = f"lstchain v{__version__}"
        radmax_header["DATE"] = Time.now().utc.iso

        for k, v in extra_headers.items():
            radmax_header[k] = v

        rad_max_hdu_interp = fits.BinTableHDU(
            temp_irf, header=radmax_header, name="RAD_MAX"
        )
        irf_interp.append(rad_max_hdu_interp)

    except KeyError:
        log.error("RAD_MAX not present for IRF interpolation")
    """

    if not point_like:
        try:
            hdus_interp["PSF"]
            psf_list = load_irf_grid(
                irfs, extname="PSF", interp_col="RPSF"
            )
            temp_irf = QTable.read(irfs[0], hdu="PSF")

            e_true = np.append(temp_irf["ENERG_LO"][0], temp_irf["ENERG_HI"][0][-1])
            src_bins = np.append(temp_irf["RAD_LO"][0], temp_irf["RAD_HI"][0][-1])
            fov_off = np.append(temp_irf["THETA_LO"][0], temp_irf["THETA_HI"][0][-1])

            psf_interp = interpolate_psf_table(
                src_bins,
                psf_list,
                irf_pars,
                interp_pars,
                quantile_resolution=1e-3
            )
            psf_hdu_interp = create_psf_table_hdu(
                psf_interp,
                true_energy=e_true,
                source_offset_bins=src_bins,
                fov_offset_bins=fov_off,
                extname="PSF",
                **extra_headers
            )

            irf_interp.append(psf_hdu_interp)
        except KeyError:
            log.error("PSF HDU not present for IRF interpolation")

    return irf_interp
