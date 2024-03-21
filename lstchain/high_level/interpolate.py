import numpy as np
import logging
import astropy.units as u
from astropy.table import QTable, Table
from astropy.io import fits
from astropy.time import Time

from lstchain.__init__ import __version__

from pyirf.io.gadf import (
    create_aeff2d_hdu,
    create_energy_dispersion_hdu,
    create_psf_table_hdu,
)
from pyirf.interpolation import (
    GridDataInterpolator,
    EnergyDispersionEstimator,
    EffectiveAreaEstimator,
    PSFTableEstimator,
    RadMaxEstimator,
)
from scipy.spatial import Delaunay, distance, QhullError

__all__ = [
    "check_in_delaunay_triangle",
    "compare_irfs",
    "get_nearest_az_node",
    "interp_params",
    "interpolate_al_cuts",
    "interpolate_gh_cuts",
    "interpolate_irf",
    "load_irf_grid",
]

log = logging.getLogger(__name__)


def interp_params(params_list, data):
    """
    From a given list of angular parameters, to be used for interpolation,
    take values from a given data dict.

    Parameters
    ----------
    params_list: List of basic angular parameters to get interpolation
        parameters information .
        'list'
    data: dict containing the required basic angular parameters.
        'dict'

    Returns
    -------
    mc_pars: List of interpolation parameters
        'list'
    """
    mc_pars = []
    if "ZEN_PNT" in params_list:
        mc_pars.append(np.cos(u.Quantity(data["ZEN_PNT"], "deg").to_value(u.rad)))

    if "B_DELTA" in params_list:
        mc_pars.append(np.sin(u.Quantity(data["B_DELTA"], "deg").to_value(u.rad)))

    if "AZ_PNT" in params_list:
        mc_pars.append(u.Quantity(data["AZ_PNT"], "deg").to_value(u.rad))

    return np.array(mc_pars)


def check_in_delaunay_triangle(irfs, data_params, use_nearest_irf_node=False):
    """
    From a given list of IRFs as grid points used for interpolation, retrieve
    the Delaunay triangulation list of IRFs, where the simplex includes the
    target points in data_params.

    If the target point does not exist inside the simplex, the IRF
    corresponding to the nearest grid point to the target value is used. This
    is also achieved if use_nearest_irf_node is passed as True.

    Fetching the nearest IRF node is also performed by checking the azimuth
    values, as there might be more than a single node, with the same position
    in the interpolation parameter space.

    If the list of given IRFs are not enough for calculating the Delaunay
    triangulation, an empty list is returned.

    Parameters
    ----------
    irfs: List of IRFs to check for Delaunay triangulation.
        'List'
    data_pars: Dict of arrays of range of parameters of the observed data
        in the event list, to check for interpolation.
        'Dict'
    use_nearest_irf_node: Boolean if we need only the nearest IRF node.
        'Bool'

    Returns
    -------
    irf_list: Revised list of IRFs after the checks.
        'List'
    """
    # Exclude AZ_PNT as target interpolation parameter
    # For the interpolation parameters only
    data_pars_sel = [d for d in data_params.keys() if d != "AZ_PNT"]
    az_idx = list(data_params.keys()).index("AZ_PNT")

    new_irfs = []
    mc_params_sel = np.empty((len(irfs), len(data_pars_sel)))
    mc_params_full = np.empty((len(irfs), len(data_params)))

    for i, irf in enumerate(irfs):
        f = fits.open(irf)[1].header

        mc_pars_sel = interp_params(data_pars_sel, f)
        mc_params_sel[i, :] = np.array(mc_pars_sel)

        mc_pars_full = interp_params(data_params, f)
        mc_params_full[i, :] = np.array(mc_pars_full)

    data_val_sel = interp_params(data_pars_sel, data_params)
    data_val_full = interp_params(data_params, data_params)

    try:
        tri = Delaunay(mc_params_sel)
    except QhullError:
        print("Not enough grid values for Delaunay triangulation")
        # Fetch the nearest IRF node
        index = distance.cdist([data_val_sel], mc_params_sel).argmin()
        index = get_nearest_az_node(
            mc_params_sel, index, mc_params_full, data_val_full, az_idx
        )
        return [irfs[index]]

    target_in_simplex = tri.find_simplex(data_val_sel)

    if not use_nearest_irf_node:
        if target_in_simplex == -1:
            # The target values are not contained in any Delaunay triangle formed
            # by the paramters of the list of IRFs provided.
            # So just include the IRF with the closest parameter values
            # to the target values

            index = distance.cdist([data_val_sel], mc_params_sel).argmin()
            print("Target value is outside interpolation. Using the nearest IRF.")
            index = get_nearest_az_node(
                mc_params_sel, index, mc_params_full, data_val_full, az_idx
            )
            new_irfs.append(irfs[index])
        else:
            # Just select the IRFs that are needed for the Delaunay triangulation
            for i in tri.simplices[target_in_simplex]:
                i_sel = get_nearest_az_node(
                    mc_params_sel, i, mc_params_full, data_val_full, az_idx
                )
                new_irfs.append(irfs[i_sel])
    else:
        index = distance.cdist([data_val_sel], mc_params_sel).argmin()
        print("Using the nearest IRF.")
        index = get_nearest_az_node(
            mc_params_sel, index, mc_params_full, data_val_full, az_idx
        )
        new_irfs.append(irfs[index])

    return new_irfs


def get_nearest_az_node(
    irf_params_sel, index, irf_params_full, target_params_full, az_idx
):
    """
    Check to see if a given IRF node overlaps with another node, in the
    interpolation parameter space, and to select based on the azimuth angle,
    the nearest node to the target.

    All interp_params objects are assumed to generated with the same IRF list.

    Parameters
    ----------
    irf_params_sel: interp_params object for the interpolation parameters-based
        information.
        'dict'
    index: index of the list of IRF node information to compare.
        'int'
    irf_params_full: interp_params object for all parameters information.
        'dict'
    target_params_full: interp_params object for all parameters information of
        the target.
        'dict'
    az_idx: index of azimuth information (AZ_PNT) in the main dict information
        used for the target.
        'int'

    Returns
    -------
    idx: index of the IRF node information with the closest azimuth value as
        with the target.
        'int'
    """
    # Remove the numerical variations by shortening the precision of values
    irf_params_short = np.around(irf_params_sel, 3)

    # Find the uniques set of nodes, to compare and choose the indices
    irf_unique, irf_num = np.unique(irf_params_short, axis=0, return_counts=True)
    idx = np.flatnonzero((irf_unique == irf_params_short[index]).all(1))

    if irf_num[idx] > 1:
        # Only using the overlapping nodes
        idx_list = np.flatnonzero((irf_params_short == irf_params_short[index]).all(1))
        # Finding the shortest distance with respect to target azimuth
        diff = np.abs(irf_params_full[idx_list, az_idx] - target_params_full[az_idx])
        idx = idx_list[np.where(diff == diff.min())[0]][0]
    else:
        # if there are no overlapping nodes
        idx = index
    return idx


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
    select_meta = ["HDUCLAS3", "INSTRUME"]
    try:
        h = Table.read(irfs[0], hdu="EFFECTIVE AREA")
        h_meta = h.meta
        cols = h.columns[:-1]

        energy_dependent_gh = "GH_EFF" in h_meta
        point_like = h_meta["HDUCLAS3"] == "POINT-LIKE"
        energy_dependent_theta = "TH_CONT" in h_meta
        source_dep = ("AL_CUT" in h_meta) or ("AL_CONT" in h_meta)
        energy_dependent_al = "AL_CONT" in h_meta

        if energy_dependent_gh:
            select_meta.append("GH_EFF")
        else:
            select_meta.append("GH_CUT")

        if point_like:
            if not source_dep:
                if energy_dependent_theta:
                    select_meta.append("TH_CONT")
                else:
                    select_meta.append("RAD_MAX")
            else:
                if energy_dependent_al:
                    select_meta.append("AL_CONT")
                else:
                    select_meta.append("AL_CUT")

    except KeyError:
        print(f"Effective Area is not present in {irfs[0]}")

    # Comparing the metadata information
    for k in select_meta:
        meta = [Table.read(i, hdu="ENERGY DISPERSION").meta[k] for i in irfs]
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
        a = [params[len(cols) * j + i] for j in np.arange(len(irfs))]
        a2, a_ind = np.unique(a, return_index=True)

        if len(a2) == len(params[i]):
            if (a2[np.argsort(a_ind)] == params[i].value).all():
                bin_bool = True
    print(
        "The other parameter axes data " f"are{' ' if bin_bool else ' not '}comparable"
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
            irf_list.append(QTable.read(irf, hdu=extname)[interp_col][0].T)
        else:
            irf_list.append(QTable.read(irf, hdu=extname)[interp_col])
    return np.stack(irf_list)


def interpolate_gh_cuts(
    gh_cuts,
    grid_points,
    target_point,
    method="linear",
):
    """
    Interpolates a grid of GH_CUTS tables to a target-point.

    Wrapper around scipy.interpolate.griddata [1].

    Parameters
    ----------
    gh_cuts: numpy.ndarray, shape=(N, M, ...)
        Gammaness-cuts for all combinations of grid-points, like energy.
        Shape (N:n_grid_points, M:n_energy_bins)
    grid_points: numpy.ndarray, shape=(N, O)
        Array of the N O-dimensional morphing parameter values corresponding
        to the N input templates.
    target_point: numpy.ndarray, shape=(O)
        Value for which the interpolation is performed (target point)
    method: 'linear', 'nearest', 'cubic'
        Interpolation method for scipy.interpolate.griddata [1].
        Defaults to 'linear'.

    Returns
    -------
    gh_cuts_interp: numpy.ndarray, shape=(1, M, ...)
        Gammaness-cuts for the target grid-point, shape (1, M:n_energy_bins)

    References
    ----------
    .. [1] https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.griddata.html
    """
    interp = GridDataInterpolator(
        grid_points=grid_points, params=gh_cuts, method=method
    )
    gh_cuts_interp = interp(target_point)

    return gh_cuts_interp


def interpolate_al_cuts(
    al_cuts,
    grid_points,
    target_point,
    method="linear",
):
    """
    Interpolates a grid of AL_CUTS tables to a target-point.

    Wrapper around scipy.interpolate.griddata [1].

    Parameters
    ----------
    al_cuts: numpy.ndarray, shape=(N, M, ...)
        Alpha-cuts for all combinations of grid-points, like energy.
        Shape (N:n_grid_points, M:n_energy_bins)
    grid_points: numpy.ndarray, shape=(N, O)
        Array of the N O-dimensional morphing parameter values corresponding
        to the N input templates.
    target_point: numpy.ndarray, shape=(O)
        Value for which the interpolation is performed (target point)
    method: 'linear', 'nearest', 'cubic'
        Interpolation method for scipy.interpolate.griddata [1].
        Defaults to 'linear'.

    Returns
    -------
    al_cuts_interp: numpy.ndarray, shape=(1, M, ...)
        Alpha-cuts for the target grid-point, shape (1, M:n_energy_bins)

    References
    ----------
    .. [1] https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.griddata.html
    """
    interp = GridDataInterpolator(
        grid_points=grid_points, params=al_cuts, method=method
    )
    al_cuts_interp = interp(target_point)

    return al_cuts_interp


def interpolate_irf(irfs, data_pars, interp_method="linear"):
    """
    Using pyirf functions with a list of IRFs and parameters to compare with
    data, to interpolate over in a selected interpolation parameter space.

    Currently for the single telescope, we are using only cos zenith and
    sin delta as the interpolation parameters for the IRFs, and not including
    the contributions from azimuth direction (of the shower) as that becomes
    significant only when considering a stereo system of telescopes. Hence,
    we have to perform the selection criteria only for the selected
    interpolation parameters.

    Along with IRFs, energy-dependent theta and gammaness cuts, RAD_MAX and
    GH_CUTS respectively, are also interpolated if they exist in the list of
    IRFs provided.

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

    # Exclude AZ_PNT as target interpolation parameter
    # For the interpolation parameters only
    params_sel = [d for d in data_pars.keys() if d != "AZ_PNT"]
    n_grid = len(irfs)
    irf_pars_sel = np.empty((n_grid, len(params_sel)))
    interp_pars_sel = list()

    extra_keys = [
        "TELESCOP",
        "INSTRUME",
        "FOVALIGN",
        "GH_CUT",
        "G_OFFSET",
        "RAD_MAX",
        "AL_CUT",
        "B_TOTAL",
        "B_INC",
    ]
    main_headers = fits.open(irfs[0])[1].header

    if main_headers["HDUCLAS3"] == "POINT-LIKE":
        point_like = True
    else:
        point_like = False

    # Update headers to be added to the final IRFs
    extra_headers = dict()

    for k in extra_keys:
        if k in main_headers:
            if main_headers.comments[k]:
                extra_headers[k] = (main_headers[k], main_headers.comments[k])
            else:
                extra_headers[k] = main_headers[k]

    for par in data_pars.keys():
        extra_headers[par] = (data_pars[par].to_value(u.deg), "deg")

    for i in np.arange(n_grid):
        f = fits.open(irfs[i])[1].header
        mc_pars_sel = interp_params(params_sel, f)
        irf_pars_sel[i, :] = np.array(mc_pars_sel)

    interp_pars_sel = interp_params(params_sel, data_pars)

    # Keep interp_pars as a tuple to keep the right dimensions in interpolation
    irf_interp = fits.HDUList(
        [
            fits.PrimaryHDU(),
        ]
    )

    # Read select IRFs, for which interpolation is supported into lists and
    # extract the necessary columns
    hdus_interp = fits.open(irfs[0])

    try:
        hdus_interp["EFFECTIVE AREA"]
        effarea_list = load_irf_grid(
            irfs, extname="EFFECTIVE AREA", interp_col="EFFAREA"
        )

        temp_irf = QTable.read(irfs[0], hdu="EFFECTIVE AREA")
        e_true = np.append(temp_irf["ENERG_LO"][0], temp_irf["ENERG_HI"][0][-1])
        fov_off = np.append(temp_irf["THETA_LO"][0], temp_irf["THETA_HI"][0][-1])

        aeff_estimator = EffectiveAreaEstimator(
            grid_points=irf_pars_sel,
            effective_area=effarea_list,
            interpolator_kwargs={"method": interp_method},
        )
        aeff_interp = aeff_estimator(interp_pars_sel)

        aeff_hdu_interp = create_aeff2d_hdu(
            effective_area=aeff_interp.T[0],
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

        edisp_estimator = EnergyDispersionEstimator(
            grid_points=irf_pars_sel,
            migra_bins=e_migra,
            energy_dispersion=edisp_list,
        )
        edisp_interp = edisp_estimator(interp_pars_sel)

        edisp_hdu_interp = create_energy_dispersion_hdu(
            energy_dispersion=edisp_interp[0],
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

        gh_cut_interp = interpolate_gh_cuts(
            gh_cuts=gh_cuts_list,
            grid_points=irf_pars_sel,
            target_point=interp_pars_sel,
            method=interp_method,
        )

        gh_header = fits.Header()
        gh_header["CREATOR"] = f"lstchain v{__version__}"
        gh_header["DATE"] = Time.now().utc.iso

        for k, v in extra_headers.items():
            gh_header[k] = v

        temp_irf["cut"] = gh_cut_interp

        gh_cut_hdu_interp = fits.BinTableHDU(temp_irf, header=gh_header, name="GH_CUTS")

        irf_interp.append(gh_cut_hdu_interp)

    except KeyError:
        log.error("GH CUTS not present for IRF interpolation")

    try:
        hdus_interp["RAD_MAX"]
        radmax_list = load_irf_grid(irfs, extname="RAD_MAX", interp_col="RAD_MAX")
        temp_irf = QTable.read(irfs[0], hdu="RAD_MAX")

        rad_max_estimator = RadMaxEstimator(
            grid_points=irf_pars_sel,
            rad_max=radmax_list,
            interpolator_kwargs={"method": interp_method},
        )
        rad_max_interp = rad_max_estimator(interp_pars_sel)

        temp_irf["RAD_MAX"] = rad_max_interp[0].T[np.newaxis, ...] * u.deg

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

    try:
        hdus_interp["AL_CUTS"]
        al_cuts_list = load_irf_grid(
            irfs, extname="AL_CUTS", interp_col="cut", gadf_irf=False
        )

        temp_irf = QTable.read(irfs[0], hdu="AL_CUTS")

        al_cut_interp = interpolate_al_cuts(
            al_cuts_list, irf_pars_sel, interp_pars_sel, method=interp_method
        )

        al_header = fits.Header()
        al_header["CREATOR"] = f"lstchain v{__version__}"
        al_header["DATE"] = Time.now().utc.iso

        for k, v in extra_headers.items():
            al_header[k] = v

        temp_irf["cut"] = al_cut_interp * u.deg

        al_cut_hdu_interp = fits.BinTableHDU(temp_irf, header=al_header, name="AL_CUTS")

        irf_interp.append(al_cut_hdu_interp)

    except KeyError:
        log.error("AL CUTS not present for IRF interpolation")

    if not point_like:
        try:
            hdus_interp["PSF"]
            psf_list = load_irf_grid(irfs, extname="PSF", interp_col="RPSF")
            temp_irf = QTable.read(irfs[0], hdu="PSF")

            e_true = np.append(temp_irf["ENERG_LO"][0], temp_irf["ENERG_HI"][0][-1])
            src_bins = np.append(temp_irf["RAD_LO"][0], temp_irf["RAD_HI"][0][-1])
            fov_off = np.append(temp_irf["THETA_LO"][0], temp_irf["THETA_HI"][0][-1])

            psf_estimator = PSFTableEstimator(
                grid_points=irf_pars_sel,
                source_offset_bins=src_bins,
                psf=psf_list,
            )
            psf_interp = psf_estimator(interp_pars_sel)

            psf_hdu_interp = create_psf_table_hdu(
                psf=psf_interp[0],
                true_energy_bins=e_true,
                source_offset_bins=src_bins,
                fov_offset_bins=fov_off,
                extname="PSF",
                **extra_headers,
            )

            irf_interp.append(psf_hdu_interp)
        except KeyError:
            log.error("PSF HDU not present for IRF interpolation")

    return irf_interp
