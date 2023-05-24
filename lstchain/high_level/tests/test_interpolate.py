import astropy.units as u
import numpy as np
from astropy.io import fits
from astropy.table import Table


def test_load_irf_grid(simulated_irf_file):
    from lstchain.high_level.interpolate import load_irf_grid

    aeff_list = load_irf_grid([simulated_irf_file], "EFFECTIVE AREA", "EFFAREA")

    assert aeff_list.shape == (1, 25, 8)


def test_interp_irf(
    simulated_irf_file,
    simulated_srcdep_irf_file,
    simulated_dl2_file,
    simulated_srcdep_dl2_file,
):
    from lstchain.high_level.interpolate import interpolate_irf
    from lstchain.scripts.tests.test_lstchain_scripts import run_program

    # Create another IRFs with different zenith and azimuth parameter
    # Global cuts

    irf_file_g_2 = simulated_irf_file.parent / "irf_interp_0.fits.gz"
    irf_file_g_3 = simulated_irf_file.parent / "irf_interp_1.fits.gz"
    irf_file_g_final = simulated_irf_file.parent / "irf_interp_final.fits.gz"

    hdus_2_g = [fits.PrimaryHDU(), ]
    hdus_3_g = [fits.PrimaryHDU(), ]

    # Global cuts (srcdep)
    irf_file_g_srcdep_2 = (
        simulated_srcdep_irf_file.parent / "irf_interp_srcdep_0.fits.gz"
    )
    irf_file_g_srcdep_3 = (
        simulated_srcdep_irf_file.parent / "irf_interp_srcdep_1.fits.gz"
    )
    irf_file_g_srcdep_final = (
        simulated_srcdep_irf_file.parent / "irf_interp_srcdep_final.fits.gz"
    )

    hdus_2_g_srcdep = [
        fits.PrimaryHDU(),
    ]
    hdus_3_g_srcdep = [
        fits.PrimaryHDU(),
    ]

    # En-dep, point-like cuts IRF
    irf_file_en_1 = simulated_dl2_file.parent / "en_dep_cut_irf_1.fits.gz"
    irf_file_en_2 = simulated_dl2_file.parent / "en_dep_cut_irf_2.fits.gz"
    irf_file_en_3 = simulated_dl2_file.parent / "en_dep_cut_irf_3.fits.gz"
    irf_file_en_final = simulated_dl2_file.parent / "interp_en_dep_cut_irf.fits.gz"

    run_program(
        "lstchain_create_irf_files",
        "--input-gamma-dl2",
        simulated_dl2_file,
        "--output-irf-file",
        irf_file_en_1,
        "--energy-dependent-gh",
        "--energy-dependent-theta",
        "--point-like",
        "--gh-efficiency=0.8",
        "--theta-containment=0.8",
        "--DL3Cuts.min_event_p_en_bin=2",
    )

    hdus_2_en = [fits.PrimaryHDU(), ]
    hdus_3_en = [fits.PrimaryHDU(), ]

    # En-dep, point-like cuts IRF (srcdep)
    irf_file_en_srcdep_1 = (
        simulated_srcdep_dl2_file.parent / "en_dep_cut_irf_srcdep_1.fits.gz"
    )
    irf_file_en_srcdep_2 = (
        simulated_srcdep_dl2_file.parent / "en_dep_cut_irf_srcdep_2.fits.gz"
    )
    irf_file_en_srcdep_3 = (
        simulated_srcdep_dl2_file.parent / "en_dep_cut_irf_srcdep_3.fits.gz"
    )
    irf_file_en_srcdep_final = (
        simulated_srcdep_dl2_file.parent / "interp_en_dep_cut_irf_srcdep.fits.gz"
    )

    run_program(
        "lstchain_create_irf_files",
        "--input-gamma-dl2",
        simulated_srcdep_dl2_file,
        "--output-irf-file",
        irf_file_en_srcdep_1,
        "--energy-dependent-gh",
        "--energy-dependent-alpha",
        "--point-like",
        "--source-dep",
        "--gh-efficiency=0.8",
        "--alpha-containment=0.8",
        "--DL3Cuts.min_event_p_en_bin=2",
    )

    hdus_2_en_srcdep = [fits.PrimaryHDU(), ]
    hdus_3_en_srcdep = [fits.PrimaryHDU(), ]

    for irf in [
        simulated_irf_file,
        simulated_srcdep_irf_file,
        irf_file_en_1,
        irf_file_en_srcdep_1,
    ]:
        # Change the effective area for different angular pointings
        aeff_1 = Table.read(irf, hdu="EFFECTIVE AREA")
        aeff_1_meta = fits.open(irf)["EFFECTIVE AREA"].header
        aeff_2 = aeff_1.copy()
        aeff_2_meta = aeff_1_meta.copy()

        zen_1 = u.Quantity(aeff_1_meta["ZEN_PNT"], "deg").to_value(u.rad)
        az_1 = u.Quantity(aeff_1_meta["AZ_PNT"], "deg").to_value(u.rad)
        del_1 = u.Quantity(aeff_1_meta["B_DELTA"], "deg").to_value(u.rad)

        factor_czd = 0.7
        factor_sd = 1.2

        zen_2 = np.arccos(np.cos(zen_1) * factor_czd)
        az_2 = az_1 * factor_czd
        del_2 = np.arcsin(np.sin(del_1) * factor_sd)

        aeff_1["EFFAREA"][0] *= factor_czd
        aeff_2["EFFAREA"][0] *= factor_czd * factor_sd

        aeff_1_meta["ZEN_PNT"] = (zen_2 * 180 / np.pi, "deg")
        aeff_1_meta["AZ_PNT"] = (az_1 * 180 / np.pi, "deg")
        aeff_1_meta["B_DELTA"] = (del_1 * 180 / np.pi, "deg")
        aeff_2_meta["ZEN_PNT"] = (zen_2 * 180 / np.pi, "deg")
        aeff_1_meta["AZ_PNT"] = (az_2 * 180 / np.pi, "deg")
        aeff_2_meta["B_DELTA"] = (del_2 * 180 / np.pi, "deg")

        aeff_hdu_1 = fits.BinTableHDU(aeff_1, header=aeff_1_meta, name="EFFECTIVE AREA")
        aeff_hdu_2 = fits.BinTableHDU(aeff_2, header=aeff_2_meta, name="EFFECTIVE AREA")

        # Change the energy migration for different angular pointings
        edisp_1 = Table.read(irf, hdu="ENERGY DISPERSION")
        edisp_1_meta = fits.open(irf)["ENERGY DISPERSION"].header
        edisp_2 = edisp_1.copy()
        edisp_2_meta = edisp_1_meta.copy()

        edisp_1["MATRIX"][0] *= factor_czd
        edisp_2["MATRIX"][0] *= factor_czd * factor_sd

        edisp_1_meta["ZEN_PNT"] = (zen_2 * 180 / np.pi, "deg")
        edisp_1_meta["AZ_PNT"] = (az_1 * 180 / np.pi, "deg")
        edisp_1_meta["B_DELTA"] = (del_1 * 180 / np.pi, "deg")
        edisp_2_meta["ZEN_PNT"] = (zen_2 * 180 / np.pi, "deg")
        edisp_2_meta["AZ_PNT"] = (az_2 * 180 / np.pi, "deg")
        edisp_2_meta["B_DELTA"] = (del_2 * 180 / np.pi, "deg")

        edisp_hdu_1 = fits.BinTableHDU(
            edisp_1, header=edisp_1_meta, name="ENERGY DISPERSION"
        )
        edisp_hdu_2 = fits.BinTableHDU(
            edisp_2, header=edisp_2_meta, name="ENERGY DISPERSION"
        )

        if "en_dep" in irf.name:
            # For GH CUTS, apply the factors for cuts, lower than the max value
            gh_1 = Table.read(irf, hdu="GH_CUTS")
            gh_1_meta = fits.open(irf)["GH_CUTS"].header
            gh_2 = gh_1.copy()
            gh_2_meta = gh_1_meta.copy()

            mask = gh_1["cut"] < gh_1["cut"].max()

            gh_1["cut"][mask] *= factor_czd
            gh_2["cut"][mask] *= factor_czd * factor_sd

            gh_1_meta["ZEN_PNT"] = (zen_2 * 180 / np.pi, "deg")
            gh_1_meta["AZ_PNT"] = (az_1 * 180 / np.pi, "deg")
            gh_1_meta["B_DELTA"] = (del_1 * 180 / np.pi, "deg")
            gh_2_meta["ZEN_PNT"] = (zen_2 * 180 / np.pi, "deg")
            gh_2_meta["AZ_PNT"] = (az_2 * 180 / np.pi, "deg")
            gh_2_meta["B_DELTA"] = (del_2 * 180 / np.pi, "deg")

            gh_hdu_1 = fits.BinTableHDU(gh_1, header=gh_1_meta, name="GH_CUTS")
            gh_hdu_2 = fits.BinTableHDU(gh_2, header=gh_2_meta, name="GH_CUTS")

            if not "srcdep" in irf.name:
                # For RAD_MAX CUTS, apply the factors for cuts, lower than the max value
                th_1 = Table.read(irf, hdu="RAD_MAX")
                th_1_meta = fits.open(irf)["RAD_MAX"].header
                th_2 = th_1.copy()
                th_2_meta = th_1_meta.copy()

                mask = th_1["RAD_MAX"] < th_1["RAD_MAX"].max()

                th_1["RAD_MAX"][mask] *= factor_czd
                th_2["RAD_MAX"][mask] *= factor_czd * factor_sd

                th_1_meta["ZEN_PNT"] = (zen_2 * 180 / np.pi, "deg")
                th_1_meta["AZ_PNT"] = (az_1 * 180 / np.pi, "deg")
                th_1_meta["B_DELTA"] = (del_1 * 180 / np.pi, "deg")
                th_2_meta["ZEN_PNT"] = (zen_2 * 180 / np.pi, "deg")
                th_2_meta["AZ_PNT"] = (az_2 * 180 / np.pi, "deg")
                th_2_meta["B_DELTA"] = (del_2 * 180 / np.pi, "deg")

                th_hdu_1 = fits.BinTableHDU(th_1, header=th_1_meta, name="RAD_MAX")
                th_hdu_2 = fits.BinTableHDU(th_2, header=th_2_meta, name="RAD_MAX")

                hdus_2_en.append(aeff_hdu_1)
                hdus_2_en.append(edisp_hdu_1)
                hdus_2_en.append(gh_hdu_1)
                hdus_2_en.append(th_hdu_1)

                hdus_3_en.append(aeff_hdu_2)
                hdus_3_en.append(edisp_hdu_2)
                hdus_3_en.append(gh_hdu_2)
                hdus_3_en.append(th_hdu_2)

            else:
                # For AL CUTS, apply the factors for cuts, lower than the max value
                al_1 = Table.read(irf, hdu="AL_CUTS")
                al_1_meta = fits.open(irf)["AL_CUTS"].header
                al_2 = al_1.copy()
                al_2_meta = al_1_meta.copy()

                mask = al_1["cut"] < al_1["cut"].max()

                al_1["cut"][mask] *= factor_czd
                al_2["cut"][mask] *= factor_czd * factor_sd

                al_1_meta["ZEN_PNT"] = (zen_2 * 180 / np.pi, "deg")
                al_1_meta["AZ_PNT"] = (az_1 * 180 / np.pi, "deg")
                al_1_meta["B_DELTA"] = (del_1 * 180 / np.pi, "deg")
                al_2_meta["ZEN_PNT"] = (zen_2 * 180 / np.pi, "deg")
                al_2_meta["AZ_PNT"] = (az_2 * 180 / np.pi, "deg")
                al_2_meta["B_DELTA"] = (del_2 * 180 / np.pi, "deg")

                al_hdu_1 = fits.BinTableHDU(al_1, header=al_1_meta, name="AL_CUTS")
                al_hdu_2 = fits.BinTableHDU(al_2, header=al_2_meta, name="AL_CUTS")

                hdus_2_en_srcdep.append(aeff_hdu_1)
                hdus_2_en_srcdep.append(edisp_hdu_1)
                hdus_2_en_srcdep.append(gh_hdu_1)
                hdus_2_en_srcdep.append(al_hdu_1)

                hdus_3_en_srcdep.append(aeff_hdu_2)
                hdus_3_en_srcdep.append(edisp_hdu_2)
                hdus_3_en_srcdep.append(gh_hdu_2)
                hdus_3_en_srcdep.append(al_hdu_2)

        else:
            if not "srcdep" in irf.name:
                hdus_2_g.append(aeff_hdu_1)
                hdus_2_g.append(edisp_hdu_1)

                hdus_3_g.append(aeff_hdu_2)
                hdus_3_g.append(edisp_hdu_2)

            else:
                hdus_2_g_srcdep.append(aeff_hdu_1)
                hdus_2_g_srcdep.append(edisp_hdu_1)

                hdus_3_g_srcdep.append(aeff_hdu_2)
                hdus_3_g_srcdep.append(edisp_hdu_2)

    fits.HDUList(hdus_2_g).writeto(irf_file_g_2, overwrite=True)
    fits.HDUList(hdus_3_g).writeto(irf_file_g_3, overwrite=True)
    fits.HDUList(hdus_2_en).writeto(irf_file_en_2, overwrite=True)
    fits.HDUList(hdus_3_en).writeto(irf_file_en_3, overwrite=True)

    fits.HDUList(hdus_2_g_srcdep).writeto(irf_file_g_srcdep_2, overwrite=True)
    fits.HDUList(hdus_3_g_srcdep).writeto(irf_file_g_srcdep_3, overwrite=True)
    fits.HDUList(hdus_2_en_srcdep).writeto(irf_file_en_srcdep_2, overwrite=True)
    fits.HDUList(hdus_3_en_srcdep).writeto(irf_file_en_srcdep_3, overwrite=True)

    irfs_g = [simulated_irf_file, irf_file_g_2, irf_file_g_3]
    irfs_en = [irf_file_en_1, irf_file_en_2, irf_file_en_3]
    irfs_g_srcdep = [
        simulated_srcdep_irf_file,
        irf_file_g_srcdep_2,
        irf_file_g_srcdep_3,
    ]
    irfs_en_srcdep = [irf_file_en_srcdep_1, irf_file_en_srcdep_2, irf_file_en_srcdep_3]

    factor_czd_t = 1 - (1 - factor_czd) / 2  # as factor_czd < 1
    factor_sd_t = 1 + (factor_sd - 1) / 2

    zen_t = np.arccos(np.cos(zen_1) * factor_czd_t) * 180 / np.pi
    del_t = np.arcsin(np.sin(del_1) * factor_sd_t) * 180 / np.pi
    az_t = az_1 * factor_czd_t * 180 / np.pi
    data_pars = {
        "ZEN_PNT": zen_t * u.deg,
        "B_DELTA": del_t * u.deg,
        "AZ_PNT": az_t * u.deg,
    }

    hdu_g = interpolate_irf(irfs_g, data_pars)
    hdu_g.writeto(irf_file_g_final, overwrite=True)

    hdu_en = interpolate_irf(irfs_en, data_pars)
    hdu_en.writeto(irf_file_en_final, overwrite=True)

    hdu_g_srcdep = interpolate_irf(irfs_g_srcdep, data_pars)
    hdu_g_srcdep.writeto(irf_file_g_srcdep_final, overwrite=True)

    hdu_en_srcdep = interpolate_irf(irfs_en_srcdep, data_pars)
    hdu_en_srcdep.writeto(irf_file_en_srcdep_final, overwrite=True)

    assert hdu_g[1].header["ZEN_PNT"] == zen_t
    assert irf_file_g_2.exists()
    assert irf_file_g_3.exists()
    assert irf_file_g_final.exists()

    assert hdu_en[1].header["ZEN_PNT"] == zen_t
    assert irf_file_en_2.exists()
    assert irf_file_en_3.exists()
    assert irf_file_en_final.exists()

    assert hdu_g_srcdep[1].header["ZEN_PNT"] == zen_t
    assert irf_file_g_srcdep_2.exists()
    assert irf_file_g_srcdep_3.exists()
    assert irf_file_g_srcdep_final.exists()

    assert hdu_en_srcdep[1].header["ZEN_PNT"] == zen_t
    assert irf_file_en_srcdep_2.exists()
    assert irf_file_en_srcdep_3.exists()
    assert irf_file_en_srcdep_final.exists()


def test_compare_irfs(
    simulated_irf_file,
    simulated_srcdep_irf_file,
    simulated_dl2_file,
    simulated_srcdep_dl2_file,
):
    from lstchain.high_level.interpolate import compare_irfs
    from lstchain.scripts.tests.test_lstchain_scripts import run_program

    # Create IRF with different global cuts for comparison
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
    # IRFs with same and different efficiency values for energy-dependent cuts,
    # and same direction pointing values.
    irf_file_en_1 = simulated_dl2_file.parent / "en_dep_cut_irf_1.fits.gz"
    irf_file_en_1_cut2 = simulated_dl2_file.parent / "en_dep_diff_cut_irf_1.fits.gz"

    run_program(
        "lstchain_create_irf_files",
        "--input-gamma-dl2",
        simulated_dl2_file,
        "--output-irf-file",
        irf_file_en_1_cut2,
        "--energy-dependent-gh",
        "--energy-dependent-theta",
        "--point-like",
        "--gh-efficiency=0.7",
        "--theta-containment=0.7",
        "--DL3Cuts.min_event_p_en_bin=2",
    )

    # IRFs with same efficiency values for energy-dependent cuts,
    # and different direction pointing values.
    irf_file_en_2 = simulated_dl2_file.parent / "en_dep_cut_irf_2.fits.gz"

    # Src-dep IRF with different global alpha cut
    srcdep_irf_file_2 = simulated_srcdep_dl2_file.parent / "irf_srcdep_2.fits.gz"
    run_program(
        "lstchain_create_irf_files",
        "--input-gamma-dl2",
        simulated_srcdep_dl2_file,
        "--output-irf-file",
        srcdep_irf_file_2,
        "--point-like",
        "--source-dep",
        "--global-alpha-cut=11",
    )

    # IRFs with same and different efficiency values for energy-dependent cuts
    srcdep_irf_file_en_1 = (
        simulated_srcdep_dl2_file.parent / "en_dep_cut_irf_srcdep_1.fits.gz"
    )
    srcdep_irf_file_en_1_cut2 = (
        simulated_srcdep_dl2_file.parent / "en_dep_diff_cut_irf_srcdep_1.fits.gz"
    )

    run_program(
        "lstchain_create_irf_files",
        "--input-gamma-dl2",
        simulated_srcdep_dl2_file,
        "--output-irf-file",
        srcdep_irf_file_en_1_cut2,
        "--energy-dependent-gh",
        "--energy-dependent-alpha",
        "--point-like",
        "--source-dep",
        "--gh-efficiency=0.7",
        "--alpha-containment=0.7",
        "--DL3Cuts.min_event_p_en_bin=2",
    )

    # IRFs with same efficiency values for energy-dependent cuts
    srcdep_irf_file_en_2 = (
        simulated_srcdep_dl2_file.parent / "en_dep_cut_irf_srcdep_2.fits.gz"
    )

    irfs_diff_global_cuts = [simulated_irf_file, irf_file_2]
    irfs_same_file = [simulated_irf_file, simulated_irf_file]
    irfs_same_en_dep_cuts_diff_dir = [irf_file_en_1, irf_file_en_2]
    irfs_diff_en_dep_cuts = [irf_file_en_1, irf_file_en_1_cut2]
    irfs_srcdep_diff_global_cuts = [simulated_srcdep_irf_file, srcdep_irf_file_2]
    irfs_srcdep_same_file = [simulated_srcdep_irf_file, simulated_srcdep_irf_file]
    irfs_srcdep_same_en_dep_cuts_diff_dir = [srcdep_irf_file_en_1, srcdep_irf_file_en_2]
    irfs_srcdep_diff_en_dep_cuts = [srcdep_irf_file_en_1, srcdep_irf_file_en_1_cut2]

    assert compare_irfs(irfs_diff_global_cuts) == 0
    assert compare_irfs(irfs_same_file)
    assert compare_irfs(irfs_diff_en_dep_cuts) == 0
    assert compare_irfs(irfs_same_en_dep_cuts_diff_dir)
    assert compare_irfs(irfs_srcdep_diff_global_cuts) == 0
    assert compare_irfs(irfs_srcdep_same_file)
    assert compare_irfs(irfs_srcdep_diff_en_dep_cuts) == 0
    assert compare_irfs(irfs_srcdep_same_en_dep_cuts_diff_dir)


def test_check_delaunay_triangles(simulated_irf_file):
    from lstchain.high_level.interpolate import check_in_delaunay_triangle

    irf_file_3 = simulated_irf_file.parent / "irf_interp_0.fits.gz"
    irf_file_4 = simulated_irf_file.parent / "irf_interp_1.fits.gz"
    irf_file_5 = simulated_irf_file.parent / "irf_interp_final.fits.gz"

    irfs = [simulated_irf_file, irf_file_3, irf_file_4, irf_file_5]

    # Fetch the grid parameters to get target points inside and outside grids.
    czd_list = []
    sd_list = []
    az_list = []

    for i in irfs:
        meta_ = fits.open(i)["EFFECTIVE AREA"].header
        czd_list.append(np.cos(meta_["ZEN_PNT"] * np.pi/180))
        sd_list.append(np.sin(meta_["B_DELTA"] * np.pi/180))
        az_list.append(meta_["AZ_PNT"])
    czd_list = np.array(czd_list)
    sd_list = np.array(sd_list)
    az_list = np.array(az_list)

    # Check on target being inside or outside Delaunay simplex
    zen_t1 = np.arccos(np.mean(czd_list)) * 180 / np.pi
    zen_t2 = np.arccos(np.min(czd_list) - 0.1) * 180 / np.pi
    del_t = np.arcsin(np.mean(sd_list)) * 180 / np.pi
    az_close_t = az_list[np.where(czd_list == np.min(czd_list))[0][0]]

    data_pars = {
        "ZEN_PNT": zen_t1 * u.deg,
        "B_DELTA": del_t * u.deg,
        "AZ_PNT": 100 * u.deg
    }
    data_pars2 = {
        "ZEN_PNT": zen_t2 * u.deg,
        "B_DELTA": del_t * u.deg,
        "AZ_PNT": az_close_t * u.deg
    }

    new_irfs_in_grid = check_in_delaunay_triangle(irfs, data_pars)
    new_irfs_out_grid = check_in_delaunay_triangle(irfs, data_pars2)
    new_irfs_in_grid_near_az = check_in_delaunay_triangle(
        irfs, data_pars2, use_nearest_irf_node=True
    )
    new_irfs_no_grid = check_in_delaunay_triangle([irfs[0]], data_pars)

    az_check = Table.read(new_irfs_in_grid_near_az[0], hdu=1).meta

    assert len(new_irfs_in_grid) == 3
    assert len(new_irfs_out_grid) == 1
    assert az_check["AZ_PNT"] == az_close_t
    assert len(new_irfs_no_grid) == 1


def test_get_nearest_az_node():
    from scipy.spatial import distance

    from lstchain.high_level.interpolate import get_nearest_az_node, interp_params

    # 2 coincident nodes in IRF interpolation grid (cos zenith, sin delta)
    data_0 = {
        "ZEN_PNT": 10 * u.deg,
        "B_DELTA": 50.3607 * u.deg,
        "AZ_PNT": 102.199 * u.deg,
    }
    data_1 = {
        "ZEN_PNT": 10 * u.deg,
        "B_DELTA": 50.3607 * u.deg,
        "AZ_PNT": 248.117 * u.deg,
    }
    # Target node, with same b_delta (sin delta) value as the above nodes
    # AZ_PNT and ZEN_PNT are random close values, not physical.
    data_target = {
        "ZEN_PNT": 12 * u.deg,
        "B_DELTA": 50.3607 * u.deg,
        "AZ_PNT": 150 * u.deg,
    }
    params_list = ["ZEN_PNT", "B_DELTA"]
    params_list_full = ["ZEN_PNT", "B_DELTA", "AZ_PNT"]
    az_idx = list(data_target.keys()).index("AZ_PNT")

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
        check_params, index, check_params_full, check_params_target_full, az_idx
    )

    # Closest node
    mc_closest = check_params_full[index2]

    # Values to compare
    mc_closest_az = round(mc_closest[-1], 3)
    az_check = round(102.199 * np.pi / 180, 3)

    assert mc_closest_az == az_check


def test_interpolate_gh_cuts():
    from lstchain.high_level.interpolate import interpolate_gh_cuts

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
