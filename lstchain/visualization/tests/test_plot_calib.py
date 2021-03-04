from lstchain.tests.test_lstchain import test_calib_path
import lstchain.visualization.plot_calib as calib
import pytest


@pytest.mark.private_data
def test_plot_calib(temp_dir_observed_files):
    pdf_filename = temp_dir_observed_files / "calibration.Run2006.0000.pdf"
    calib.read_file(test_calib_path)
    calib.plot_all(
        calib.ped_data,
        calib.ff_data,
        calib.calib_data,
        run=2006,
        plot_file=pdf_filename,
    )
    assert pdf_filename.is_file()
