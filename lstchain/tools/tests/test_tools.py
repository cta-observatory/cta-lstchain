from pathlib import Path

from ctapipe.core import run_tool
from lstchain.tools.lstchain_create_dl2_file import ReconstructionHDF5Writer


def test_create_dl2_file(temp_dir, simulated_dl1_file, rf_models):

    config = Path("./lstchain/data/dl2_tool_config.json").absolute()

    assert (
        run_tool(
            ReconstructionHDF5Writer(),
            argv=[
                f"--config={config}",
                f"--input={simulated_dl1_file}",
                f"--output={temp_dir}",
                f"--energy-model={rf_models['energy']}",
                f"--disp-model={rf_models['disp']}",
                f"--gh-model={rf_models['gh_sep']}",
            ]
        )
        == 0
    )
