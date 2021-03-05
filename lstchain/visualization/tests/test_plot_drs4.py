from lstchain.tests.test_lstchain import test_drs4_pedestal_path, test_drs4_r0_path
from lstchain.visualization import plot_drs4
import pytest


@pytest.mark.private_data
def test_plot_drs4(temp_dir_observed_files):
    pdf_filename = temp_dir_observed_files / "drs4_pedestal.Run2005.0000.pdf"
    plot_drs4.plot_pedestals(
        test_drs4_r0_path,
        test_drs4_pedestal_path,
        run=2005,
        plot_file=pdf_filename,
        tel_id=1,
        offset_value=400,
        sample_size=100,
    )

    assert pdf_filename.is_file()
