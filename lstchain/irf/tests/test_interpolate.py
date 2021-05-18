import pytest
from astropy.table import Table


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

    assert aeff_list.shape == (1, 8, 21)


"""
def test_interp_irf(simulated_irf_file):
    from lstchain.irf.interpolate import interpolate_irf

    irf_file_2 = simulated_irf_file.parent / "irf2.fits.gz"

    irfs = [simulated_irf_file, irf_file_2]
    #print(Table.read(simulated_irf_file, hdu=1).meta)

    data_pars = {"AZ_PNT": 0}

    hdu = interpolate_irf(irfs, data_pars)
    #print(hdu[1].header)

    assert hdu[1].header["ZEN_PNT"] == 20
"""
