import astropy.units as u
import numpy as np
from astropy.io import fits
from astropy.table import Table


def test_compare_irfs(simulated_irf_file, simulated_dl2_file):
    from lstchain.high_level.interpolate import compare_irfs
    from lstchain.scripts.tests.test_lstchain_scripts import run_program

    # Creating a IRF with a different gammaness cut to check the comparison
    irf_file_2 = simulated_dl2_file.parent / "diff_cut_irf.fits.gz"
    run_program(
        "lstchain_create_irf_files",
        "--input-gamma-dl2",
        simulated_dl2_file,
        "--input-proton-dl2",
        simulated_dl2_file,
        "--input-electron-dl2",
        simulated_dl2_file,
        "--output-irf-file",
        irf_file_2,
        "--global-gh-cut=0.7",
    )

    irfs_1 = [simulated_irf_file, irf_file_2]
    irfs_2 = [simulated_irf_file, simulated_irf_file]

    assert compare_irfs(irfs_1) == 0
    assert compare_irfs(irfs_2)


def test_load_irf_grid(simulated_irf_file):
    from lstchain.high_level.interpolate import load_irf_grid

    aeff_list = load_irf_grid([simulated_irf_file], "EFFECTIVE AREA", "EFFAREA")

    assert aeff_list.shape == (1, 23, 8)


def test_interp_irf(simulated_irf_file):
    from lstchain.high_level.interpolate import interpolate_irf

    # Create another IRFs with different zenith and azimuth parameter
    irf_file_3 = simulated_irf_file.parent / "irf_interp_0.fits.gz"
    irf_file_4 = simulated_irf_file.parent / "irf_interp_1.fits.gz"
    irf_file_5 = simulated_irf_file.parent / "irf_interp_final.fits.gz"

    # Change the effective area for different angular pointings
    aeff_1 = Table.read(simulated_irf_file, hdu="EFFECTIVE AREA")
    aeff_1_meta = fits.open(simulated_irf_file)["EFFECTIVE AREA"].header
    aeff_2 = aeff_1.copy()
    aeff_2_meta = aeff_1_meta.copy()

    zen_1 = u.Quantity(aeff_1_meta["ZEN_PNT"], "deg").to_value(u.rad)
    del_1 = u.Quantity(aeff_1_meta["B_DELTA"], "deg").to_value(u.rad)

    zen_2 = 2 * zen_1
    del_2 = 1.2 * del_1

    factor_zd = (np.cos(zen_2))/np.cos(zen_1)
    factor_del = (np.sin(del_2))/np.sin(del_1)

    aeff_1["EFFAREA"][0] *= factor_zd
    aeff_2["EFFAREA"][0] *= factor_zd * factor_del

    aeff_1_meta["ZEN_PNT"] = (zen_2 * 180/np.pi, "deg")
    aeff_1_meta["B_DELTA"] = (del_1 * 180/np.pi, "deg")
    aeff_2_meta["ZEN_PNT"] = (zen_2 * 180/np.pi, "deg")
    aeff_2_meta["B_DELTA"] = (del_2 * 180/np.pi, "deg")

    aeff_hdu_1 = fits.BinTableHDU(
        aeff_1, header=aeff_1_meta, name="EFFECTIVE AREA"
    )
    aeff_hdu_2 = fits.BinTableHDU(
        aeff_2, header=aeff_2_meta, name="EFFECTIVE AREA"
    )

    # Change the energy migration for different angular pointings
    edisp_1 = Table.read(simulated_irf_file, hdu="ENERGY DISPERSION")
    edisp_2 = edisp_1.copy()

    edisp_1["MATRIX"][0] *= factor_zd
    edisp_2["MATRIX"][0] *= factor_zd * factor_del

    edisp_hdu_1 = fits.BinTableHDU(
        edisp_1, header=aeff_1_meta, name="ENERGY DISPERSION"
    )
    edisp_hdu_2 = fits.BinTableHDU(
        edisp_2, header=aeff_2_meta, name="ENERGY DISPERSION"
    )

    fits.HDUList(
        [fits.PrimaryHDU(), aeff_hdu_1, edisp_hdu_1]
    ).writeto(irf_file_3, overwrite=True)

    fits.HDUList(
        [fits.PrimaryHDU(), aeff_hdu_2, edisp_hdu_2]
    ).writeto(irf_file_4, overwrite=True)

    irfs = [simulated_irf_file, irf_file_3, irf_file_4]
    data_pars = {
        "ZEN_PNT": 30 * u.deg,
        "B_DELTA": (del_1 * 0.8 * u.rad).to(u.deg)
    }
    print(data_pars)
    hdu = interpolate_irf(irfs, data_pars)
    hdu.writeto(irf_file_5, overwrite=True)

    assert hdu[1].header["ZEN_PNT"] == 30
    assert irf_file_3.exists()
    assert irf_file_4.exists()
    assert irf_file_5.exists()


def test_check_delaunay_triangles(simulated_irf_file):
    from lstchain.high_level.interpolate import check_in_delaunay_triangle

    irf_file_3 = simulated_irf_file.parent / "irf_interp_0.fits.gz"
    irf_file_4 = simulated_irf_file.parent / "irf_interp_1.fits.gz"
    irf_file_5 = simulated_irf_file.parent / "irf_interp_final.fits.gz"

    irfs = [simulated_irf_file, irf_file_3, irf_file_4, irf_file_5]

    # Check on target being inside or outside Delaunay simplex
    data_pars = {
        "ZEN_PNT": 25 * u.deg,
        "B_DELTA": 45 * u.deg
    }
    data_pars2 = {
        "ZEN_PNT": 58 * u.deg,
        "B_DELTA": 70 * u.deg
    }

    new_irfs = check_in_delaunay_triangle(irfs, data_pars)
    new_irfs2 = check_in_delaunay_triangle(irfs, data_pars2)
    new_irfs3 = check_in_delaunay_triangle(irfs, data_pars, use_nearest_irf_node=True)

    t3 = Table.read(new_irfs3[0], hdu=1).meta

    assert len(new_irfs) == 3
    assert len(new_irfs2) == 1
    assert t3["ZEN_PNT"] == 20


def test_get_nearest_az_node():
    from scipy.spatial import distance

    from lstchain.high_level.interpolate import get_nearest_az_node, interp_params

    # 2 coincident nodes in IRF interpolation grid (cos zenith, sin delta)
    data_0  = {
        "ZEN_PNT": 10 * u.deg,
        "B_DELTA": 50.3607 * u.deg,
        "AZ_PNT": 102.199 * u.deg,
    }
    data_1  = {
        "ZEN_PNT": 10 * u.deg,
        "B_DELTA": 50.3607 * u.deg,
        "AZ_PNT": 248.117 * u.deg,
    }
    # Target node, with same b_delta (sin delta) value as the above nodes
    # AZ_PNT and ZEN_PNT are random close values, not physical.
    data_target  = {
        "ZEN_PNT": 12 * u.deg,
        "B_DELTA": 50.3607 * u.deg,
        "AZ_PNT": 150 * u.deg,
    }
    params_list = ["ZEN_PNT", "B_DELTA"]
    params_list_full = ["ZEN_PNT", "B_DELTA", "AZ_PNT"]

    # Use interp_params to have appropriate lists for the test
    # Without AZ_PNT, for IRF interpolation space
    check_params_0 = interp_params(params_list, data_0)
    check_params_1 = interp_params(params_list, data_1)
    check_params_target = interp_params(params_list, data_target)
    check_params = np.array([check_params_0, check_params_1])

    # With AZ_PNT
    check_params_0_full = interp_params(params_list_full, data_0)
    check_params_1_full = interp_params(params_list_full, data_1)
    check_params_target_full = interp_params(params_list_full, data_target)
    check_params_full = np.array([check_params_0_full, check_params_1_full])

    # Get the distance between the target and given nodes, in IRF interpolation grid
    index = distance.cdist([check_params_target], check_params).argmin()

    # Fetch the index of node, closest to the target
    index2 = get_nearest_az_node(
        check_params, index, check_params_full, check_params_target_full
    )

    # Closest node
    mc_closest = check_params_full[index2]

    # Values to compare
    mc_closest_az = round(mc_closest[-1], 3)
    az_check = round(102.199 * np.pi/180, 3)

    assert mc_closest_az == az_check

def test_interpolate_gh_cuts():
    from lstchain.high_level.interpolate import interpolate_gh_cuts

    # Similar function as interpolate_th_cuts, hence no need for extra test
    # linear test case
    gh_cuts_1 = np.array([[0, 0], [0.1, 0], [0.2, 0.1], [0.3, 0.2]])
    gh_cuts_2 = 2 * gh_cuts_1
    gh_cut = np.array([gh_cuts_1, gh_cuts_2])

    grid_points = np.array([[0], [0.1]])
    target_point = np.array([0.05])

    interp = interpolate_gh_cuts(
        gh_cuts=gh_cut,
        grid_points=grid_points,
        target_point=target_point,
        method="linear",
    )

    assert interp.shape == (1, *gh_cuts_1.shape)
    assert np.allclose(interp, 1.5 * gh_cuts_1)
