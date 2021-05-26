import pytest


def test_compare_irfs(simulated_irf_file, simulated_dl2_file):
    from lstchain.irf.interpolate import compare_irfs
    from lstchain.scripts.tests.test_lstchain_scripts import run_program

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

    assert compare_irfs(irfs_1) == False
    assert compare_irfs(irfs_2)


def test_load_irf_grid(simulated_irf_file):
    from lstchain.irf.interpolate import load_irf_grid

    aeff_list = load_irf_grid([simulated_irf_file], "EFFECTIVE AREA", "EFFAREA")

    assert aeff_list.shape == (1, 21, 8)


def test_interp_irf(simulated_irf_file):
    from lstchain.irf.interpolate import interpolate_irf
    import numpy as np
    from astropy.table import Table
    from astropy.io import fits

    # Create another IRF with different zenith parameter
    irf_file_3 = simulated_irf_file.parent / "irf3.fits.gz"

    # Change the effective area for different zenith pointing
    aeff_temp = Table.read(simulated_irf_file, hdu="EFFECTIVE AREA")

    zen_1 = float(aeff_temp.meta["ZEN_PNT"][:-4])
    factor = np.exp(
        np.cos(zen_1 * np.pi/180.)/(1 - np.cos(zen_1 * np.pi/180.))
    )
    aeff_temp["EFFAREA"][0] *= factor
    zen_2 = round(np.arccos(1 - np.cos(zen_1 * np.pi/180.)) * 180/np.pi, 2)
    aeff_temp.meta["ZEN_PNT"] = str(zen_2) + ' deg'
    aeff_hdu = fits.BinTableHDU(aeff_temp, header=aeff_temp.meta, name="EFFECTIVE AREA")

    # Change the energy migration for different zenith pointing
    edisp_temp = Table.read(simulated_irf_file, hdu="ENERGY DISPERSION")
    edisp_temp["MATRIX"][0] *= factor
    edisp_temp.meta["ZEN_PNT"] = str(zen_2) + ' deg'
    edisp_hdu = fits.BinTableHDU(edisp_temp, header=edisp_temp.meta, name="ENERGY DISPERSION")

    fits.HDUList([fits.PrimaryHDU(), aeff_hdu, edisp_hdu]).writeto(irf_file_3, overwrite=True)

    irfs = [simulated_irf_file, irf_file_3]
    data_pars = {"ZEN_PNT": 60}
    hdu = interpolate_irf(irfs, data_pars)

    assert hdu[1].header["ZEN_PNT"] == '60.0 deg'
