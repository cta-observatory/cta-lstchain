import pytest


def test_duplicate_irfs(simulated_irf_file):
    from lstchain.irf.interpolate import duplicate_irfs
    irfs = [simulated_irf_file]
    assert duplicate_irfs(irfs)


def test_compare_irfs(simulated_irf_file, simulated_dl2_file):
    from lstchain.irf.interpolate import compare_irfs
    from lstchain.scripts.tests.test_lstchain_scripts import run_program

    # Creating a IRF with a different gammaness cut to check the comparison
    irf_file_2 = simulated_dl2_file.parent / "irf2.fits.gz"
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
        "--fixed-gh-cut=0.7",
    )

    irfs_1 = [simulated_irf_file, irf_file_2]
    irfs_2 = [simulated_irf_file, simulated_irf_file]

    assert compare_irfs(irfs_1) == 0
    assert compare_irfs(irfs_2)


def test_load_irf_grid(simulated_irf_file):
    from lstchain.irf.interpolate import load_irf_grid

    aeff_list = load_irf_grid([simulated_irf_file], "EFFECTIVE AREA", "EFFAREA")

    assert aeff_list.shape == (1, 21, 8)


def test_interp_irf(simulated_irf_file):
    from lstchain.irf.interpolate import interpolate_irf

    import numpy as np
    from astropy.table import Table
    import astropy.units as u
    from astropy.io import fits

    # Create another IRFs with different zenith and azimuth parameter
    irf_file_3 = simulated_irf_file.parent / "irf3.fits.gz"
    irf_file_4 = simulated_irf_file.parent / "irf4.fits.gz"
    irf_file_5 = simulated_irf_file.parent / "irf5.fits.gz"

    # Change the effective area for different angular pointings
    aeff_1 = Table.read(simulated_irf_file, hdu="EFFECTIVE AREA")
    aeff_2 = aeff_1.copy()

    zen_1 = u.Quantity(aeff_1.meta["ZEN_PNT"]).to_value(u.deg)
    az_1 = u.Quantity(aeff_1.meta["AZ_PNT"]).to_value(u.deg)

    if az_1 == 0:
        az_1 += 1

    factor_zd = (np.cos(2 * zen_1 * np.pi/180))/np.cos(zen_1 * np.pi/180)
    factor_az = (np.pi - az_1 * np.pi/180)/(az_1 * np.pi/180)

    aeff_1["EFFAREA"][0] *= factor_zd
    aeff_2["EFFAREA"][0] *= factor_zd * factor_az

    zen_2 = 2 * zen_1
    az_2 = 180 - az_1

    aeff_1.meta["ZEN_PNT"] = str(zen_2) + ' deg'
    aeff_1.meta["AZ_PNT"] = str(az_1) + ' deg'
    aeff_2.meta["ZEN_PNT"] = str(zen_2) + ' deg'
    aeff_2.meta["AZ_PNT"] = str(az_2) + ' deg'

    aeff_hdu_1 = fits.BinTableHDU(
        aeff_1, header=aeff_1.meta, name="EFFECTIVE AREA"
    )
    aeff_hdu_2 = fits.BinTableHDU(
        aeff_2, header=aeff_2.meta, name="EFFECTIVE AREA"
    )

    # Change the energy migration for different angular pointings
    edisp_1 = Table.read(simulated_irf_file, hdu="ENERGY DISPERSION")
    edisp_2 = edisp_1.copy()

    edisp_1["MATRIX"][0] *= factor_zd
    edisp_2["MATRIX"][0] *= factor_zd * factor_az

    edisp_1.meta["ZEN_PNT"] = str(zen_2) + ' deg'
    edisp_1.meta["AZ_PNT"] = str(az_1) + ' deg'
    edisp_2.meta["ZEN_PNT"] = str(zen_2) + ' deg'
    edisp_2.meta["AZ_PNT"] = str(az_2) + ' deg'

    edisp_hdu_1 = fits.BinTableHDU(
        edisp_1, header=edisp_1.meta, name="ENERGY DISPERSION"
    )
    edisp_hdu_2 = fits.BinTableHDU(
        edisp_2, header=edisp_2.meta, name="ENERGY DISPERSION"
    )

    fits.HDUList(
        [fits.PrimaryHDU(), aeff_hdu_1, edisp_hdu_1]
    ).writeto(irf_file_3, overwrite=True)

    fits.HDUList(
        [fits.PrimaryHDU(), aeff_hdu_2, edisp_hdu_2]
    ).writeto(irf_file_4, overwrite=True)

    irfs = [simulated_irf_file, irf_file_3, irf_file_4]
    data_pars = {"ZEN_PNT": 30 * u.deg, "AZ_PNT": 70 * u.deg}
    hdu = interpolate_irf(irfs, data_pars)
    hdu.writeto(irf_file_5, overwrite=True)

    assert hdu[1].header["ZEN_PNT"] == '30.0 deg'
    assert irf_file_3.exists()
    assert irf_file_4.exists()
    assert irf_file_5.exists()


def test_check_delaunay_triangles(simulated_irf_file):
    from lstchain.irf.interpolate import check_in_delaunay_triangle
    import astropy.units as u

    irf_file_3 = simulated_irf_file.parent / "irf3.fits.gz"
    irf_file_4 = simulated_irf_file.parent / "irf4.fits.gz"
    irf_file_5 = simulated_irf_file.parent / "irf5.fits.gz"

    irfs = [simulated_irf_file, irf_file_3, irf_file_4, irf_file_5]

    # Check on target being inside or outside Delaunay simplex
    data_pars = {"ZEN_PNT": 35 * u.deg, "AZ_PNT": 40 * u.deg}
    data_pars2 = {"ZEN_PNT": 58 * u.deg, "AZ_PNT": 35 * u.deg}

    new_irfs = check_in_delaunay_triangle(irfs, data_pars)
    new_irfs2 = check_in_delaunay_triangle(irfs, data_pars2)

    assert len(new_irfs) == 3
    assert len(new_irfs2) == 1
