"""
Convert DL2 files from lstchain to ctapipe.
"""

from importlib.resources import files, as_file
import numpy as np
import tables
from astropy import units as u
from astropy.table import Column, Table
from astropy.time import Time

from ctapipe.containers import (
    ParticleClassificationContainer,
    ReconstructedGeometryContainer,
    ReconstructedEnergyContainer,
)
from ctapipe.core import Tool
from ctapipe.core.traits import (
    Bool,
    Path,
    flag,
    Unicode,
)
from ctapipe.instrument.optics import FocalLengthKind
from ctapipe.io import read_table, write_table
from ctapipe.reco.utils import add_defaults_and_meta
from ctapipe.core import Provenance
from ctapipe.instrument import SubarrayDescription

POINTING_GROUP = "/dl0/monitoring/telescope/pointing"
DL1_TELESCOPE_GROUP = "/dl1/event/telescope"
DL2_TELESCOPE_GROUP = "/dl2/event/telescope"
DL2_SUBARRAY_GROUP = "/dl2/event/subarray"
SUBARRAY_EVENT_KEYS = ["obs_id", "event_id"]
TELESCOPE_EVENT_KEYS = ["obs_id", "event_id", "tel_id"]

__all__ = ["get_lst1_subarray_description", "DL2Converter"]

def get_lst1_subarray_description(focal_length_choice=FocalLengthKind.EFFECTIVE):
    """
    Load subarray description from bundled file
    
    Parameters
    ----------
    focal_length_choice : FocalLengthKind
        Choice of focal length to use.  Options are ``FocalLengthKind.EQUIVALENT``
        and ``FocalLengthKind.EFFECTIVE``. Default is ``FocalLengthKind.EFFECTIVE``.

    Returns
    -------
    SubarrayDescription
        Subarray description of the LST-1 telescope.
    """
    with as_file(files("lstchain") / "resources/LST-1_SubarrayDescription.h5") as path:
        Provenance().add_input_file(path, role="SubarrayDescription")
        return SubarrayDescription.from_hdf(path, focal_length_choice=focal_length_choice)


class DL2Converter(Tool):
    """
    Tool to convert DL2 files from lstchain to ctapipe format.
    The tool reads DL1 data from lstchain HDF5 files and saves the results
    in ctapipe DL2 format.
    """

    name = "lstchain_convert_dl2"
    description = __doc__

    examples = """
    To convert from DL2 lstchain data files and save the results in ctapipe DL2 format:
    > lstchain_convert_dl2 \\
        --input_url input.lstchain.dl2.h5 \\
        --output output.dl2.h5 \\
        --overwrite \\
    """

    input_url = Path(
        help="Input LST-1 DL2 HDF5 files in lstchain format",
        allow_none=True,
        exists=True,
        directory_ok=False,
        file_ok=True,
    ).tag(config=True)

    dl2_telescope = Bool(
        default_value=True,
        allow_none=False,
        help="Set whether to include dl2 telescope-event-wise data in the output file.",
    ).tag(config=True)

    dl2_subarray = Bool(
        default_value=True,
        allow_none=False,
        help="Set whether to include dl2 subarray-event-wise data in the output file.",
    ).tag(config=True)

    prefix = Unicode(
        default_value="lstchain",
        allow_none=False,
        help="Name of the reconstruction algorithm used to generate the dl2 data.",
    ).tag(config=True)

    output_path = Path(
        default_value="./ctapipe_output.dl2.h5",
        allow_none=False,
        help="Output path to save the converted DL2 file in ctapipe format.",
    ).tag(config=True)

    overwrite = Bool(help="Overwrite output file if it exists").tag(config=True)

    aliases = {
        ("i", "input_url"): "DL2Converter.input_url",
        ("o", "output"): "DL2Converter.output_path",
    }

    flags = {
        "overwrite": (
            {"DL2Converter": {"overwrite": True}},
            "Overwrite existing files",
        ),
    }

    flags = {
        **flag(
            "dl2-telescope",
            "DL2Converter.dl2_telescope",
            "Include dl2 telescope-event-wise data in the output file",
            "Exclude dl2 telescope-event-wise data in the output file",
        ),
        **flag(
            "dl2-subarray",
            "DL2Converter.dl2_subarray",
            "Include dl2 subarray-event-wise data in the output file",
            "Exclude dl2 subarray-event-wise data in the output file",
        ),
    }

    def setup(self):
        # Save dl2 tree schemas and tel id for easy access
        self.dl2_table_name = "/dl2/event/telescope/parameters/LST_LSTCam"
        self.tel_id = 1

        # Get the number of rows in the table
        with tables.open_file(self.input_url) as input_file:
            self.table_length = len(input_file.get_node(self.dl2_table_name))

        # Get the SubarrayDescription of the LST-1 telescope
        self.subarray = get_lst1_subarray_description()
        # Write the SubarrayDescription to the output file
        self.subarray.to_hdf(self.output_path, overwrite=self.overwrite)
        self.log.info("SubarrayDescription was stored in '%s'", self.output_path)

    def start(self):

        all_identifiers = read_table(self.input_url, self.dl2_table_name)
        all_identifiers.meta = {}
        
        dl2_table = all_identifiers.copy()
        tel_az = u.Quantity(dl2_table["az_tel"], unit=u.rad)
        reco_az = u.Quantity(dl2_table["reco_az"], unit=u.rad)
        tel_alt = u.Quantity(dl2_table["alt_tel"], unit=u.rad)
        reco_alt = u.Quantity(dl2_table["reco_alt"], unit=u.rad)
        reco_energy = u.Quantity(dl2_table["reco_energy"], unit=u.TeV)
        event_type = dl2_table["event_type"]
        time = Time(dl2_table["dragon_time"] * u.s, format="unix")
        # Create the pointing table
        # This table is used to store the uncalibrated
        # telescope pointing per every thousandth event
        time_ = time[::1000]
        tel_az_ = tel_az[::1000]
        tel_alt_ = tel_alt[::1000]
        # Create the dl0 telescope pointing table
        pointing_table = Table(
            {
                "time": time_,
                "azimuth": tel_az_,
                "altitude": tel_alt_,
            }
        )
        # Ensure last element is included so that pointing interpolator
        # works correctly without extrapolation.
        if time_[-1] != time[-1]:
            pointing_table.add_row(
                {
                    'time': time[-1],
                    'azimuth': tel_az[-1],
                    'altitude': tel_alt[-1],
                }
            )
        write_table(
            pointing_table,
            self.output_path,
            f"{POINTING_GROUP}/tel_{self.tel_id:03d}",
            overwrite=self.overwrite,
        )
        self.log.info(
            "DL1 telescope pointing table was stored in '%s' under '%s'",
            self.output_path,
            f"{POINTING_GROUP}/tel_{self.tel_id:03d}",
        )
        # Set the time format to MJD since in the other table we store the time in MJD
        time.format = "mjd"
        # Keep only the necessary columns for the creation of tables
        all_identifiers.keep_columns(TELESCOPE_EVENT_KEYS)
        # Create the dl1 telescope trigger table
        trigger_table = all_identifiers.copy()
        trigger_table.add_column(time, name="time")
        trigger_table.add_column(event_type, name="event_type")
        trigger_table.add_column(-1, name="n_trigger_pixels")
        write_table(
            trigger_table,
            self.output_path,
            "/dl1/event/telescope/trigger",
            overwrite=self.overwrite,
        )
        self.log.info(
            "DL1 telescope trigger table was stored in '%s' under '%s'",
            self.output_path,
            "/dl1/event/telescope/trigger",
        )
        trigger_table.keep_columns(["obs_id", "event_id", "time", "event_type"])
        trigger_table.add_column(
            np.ones((len(trigger_table), 1), dtype=bool), name="tel_with_trigger"
        )
        # Save the dl1 subrray trigger table to the output file
        write_table(
            trigger_table,
            self.output_path,
            "/dl1/event/subarray/trigger",
            overwrite=self.overwrite,
        )
        self.log.info(
            "DL1 subarray trigger table was stored in '%s' under '%s'",
            self.output_path,
            "/dl1/event/subarray/trigger",
        )
        # Rename the columns for the hillas parameters to match ctapipe naming conventions
        dl2_table.rename_column("intensity", "hillas_intensity")
        dl2_table.rename_column("x", "hillas_x")
        dl2_table.rename_column("y", "hillas_y")
        dl2_table.rename_column("phi", "hillas_phi")
        dl2_table.rename_column("psi", "hillas_psi")
        dl2_table.rename_column("length", "hillas_length")
        dl2_table.rename_column("length_uncertainty", "hillas_length_uncertainty")
        dl2_table.rename_column("width", "hillas_width")
        dl2_table.rename_column("width_uncertainty", "hillas_width_uncertainty")
        dl2_table.rename_column("skewness", "hillas_skewness")
        dl2_table.rename_column("kurtosis", "hillas_kurtosis")
        dl2_table.rename_column("time_gradient", "timing_deviation")
        dl2_table.rename_column("intercept", "timing_intercept")
        dl2_table.rename_column("n_pixels", "morphology_n_pixels")
        dl2_table.rename_column("n_islands", "morphology_n_islands")
        # Create the dl1 telescope parameters table
        parameter_table = dl2_table.copy()
        parameter_table.keep_columns(
            [
                "obs_id",
                "event_id",
                "hillas_intensity",
                "hillas_x",
                "hillas_y",
                "hillas_phi",
                "hillas_psi",
                "hillas_length",
                "hillas_length_uncertainty",
                "hillas_width",
                "hillas_width_uncertainty",
                "hillas_skewness",
                "hillas_kurtosis",
                "timing_deviation",
                "timing_intercept",
                "morphology_n_pixels",
                "morphology_n_islands",
            ]
        )
        parameter_table.add_column(self.tel_id, name="tel_id", index=2)
        # Save the dl1 parameters table to the output file
        write_table(
            parameter_table,
            self.output_path,
            f"/dl1/event/telescope/parameters/tel_{self.tel_id:03d}",
            overwrite=self.overwrite,
        )
        self.log.info(
            "DL1 parameters table was stored in '%s' under '%s'",
            self.output_path,
            f"/dl1/event/telescope/parameters/tel_{self.tel_id:03d}",
        )

        # Only keep DL2 prediction columns and rename and polish them
        dl2_table.keep_columns(TELESCOPE_EVENT_KEYS + ["gammaness"])
        # Rename gammaness to self.prefix_tel_prediction
        dl2_table.rename_column("gammaness", f"{self.prefix}_tel_prediction")
        # Add the reconstructed altitude to the prediction table
        dl2_table.add_column(
            reco_alt.to(u.deg), name=f"{self.prefix}_tel_alt"
        )
        # Add the reconstructed azimuth to the prediction table
        dl2_table.add_column(
            reco_az.to(u.deg), name=f"{self.prefix}_tel_az"
        )
        # Add the reconstructed energy to the prediction table
        dl2_table.add_column(
            reco_energy, name=f"{self.prefix}_tel_energy"
        )
        # Create is_valid column based on event_type
        is_valid_col = Column(
            event_type.data == 32,
            name=f"{self.prefix}_tel_is_valid",
            dtype=np.bool_
        )
        dl2_table.add_column(is_valid_col)
        # Create telescopes flag column for subarray tables
        telescopes_flag = Column(
            np.array(is_valid_col, dtype=np.bool_)[:, None],
            f"{self.prefix}_telescopes"
        )

        # Prepare the classification telescope table
        classification_tel_table = dl2_table.copy()
        classification_tel_table.keep_columns(
            TELESCOPE_EVENT_KEYS
            + [f"{self.prefix}_tel_prediction", f"{self.prefix}_tel_is_valid"]
        )
        # Add the default values and meta data to the table
        add_defaults_and_meta(
            classification_tel_table,
            ParticleClassificationContainer,
            prefix=self.prefix,
            add_tel_prefix=True,
        )
        # Save the prediction to the output file
        if self.dl2_telescope:
            write_table(
                classification_tel_table,
                self.output_path,
                f"{DL2_TELESCOPE_GROUP}/classification/{self.prefix}/tel_{self.tel_id:03d}",
                overwrite=self.overwrite,
            )
            self.log.info(
                "DL2 prediction data was stored in '%s' under '%s'",
                self.output_path,
                f"{DL2_TELESCOPE_GROUP}/classification/{self.prefix}/tel_{self.tel_id:03d}",
            )
        # Write the mono telescope prediction to the subarray prediction table
        if self.dl2_subarray:
            classification_subarray_table = classification_tel_table.copy()
            classification_subarray_table.remove_column("tel_id")
            for colname in classification_subarray_table.colnames:
                if "_tel_" in colname:
                    classification_subarray_table.rename_column(
                        colname, colname.replace("_tel", "")
                    )
            classification_subarray_table.add_column(telescopes_flag)
            # Save the prediction to the output file
            write_table(
                classification_subarray_table,
                self.output_path,
                f"{DL2_SUBARRAY_GROUP}/classification/{self.prefix}",
                overwrite=self.overwrite,
            )
            self.log.info(
                "DL2 prediction data was stored in '%s' under '%s'",
                self.output_path,
                f"{DL2_SUBARRAY_GROUP}/classification/{self.prefix}",
            )

        # Prepare the energy telescope table
        energy_tel_table = dl2_table.copy()
        energy_tel_table.keep_columns(
            TELESCOPE_EVENT_KEYS
            + [f"{self.prefix}_tel_energy", f"{self.prefix}_tel_is_valid"]
        )
        # Add the default values and meta data to the table
        add_defaults_and_meta(
            energy_tel_table,
            ReconstructedEnergyContainer,
            prefix=self.prefix,
            add_tel_prefix=True,
        )
        # Save the prediction to the output file
        if self.dl2_telescope:
            write_table(
                energy_tel_table,
                self.output_path,
                f"{DL2_TELESCOPE_GROUP}/energy/{self.prefix}/tel_{self.tel_id:03d}",
                overwrite=self.overwrite,
            )
            self.log.info(
                "DL2 prediction data was stored in '%s' under '%s'",
                self.output_path,
                f"{DL2_TELESCOPE_GROUP}/energy/{self.prefix}/tel_{self.tel_id:03d}",
            )
        # Write the mono telescope prediction to the subarray prediction table
        if self.dl2_subarray:
            energy_subarray_table = energy_tel_table.copy()
            energy_subarray_table.remove_column("tel_id")
            for colname in energy_subarray_table.colnames:
                if "_tel_" in colname:
                    energy_subarray_table.rename_column(
                        colname, colname.replace("_tel", "")
                    )
            energy_subarray_table.add_column(telescopes_flag)
            # Save the prediction to the output file
            write_table(
                energy_subarray_table,
                self.output_path,
                f"{DL2_SUBARRAY_GROUP}/energy/{self.prefix}",
                overwrite=self.overwrite,
            )
            self.log.info(
                "DL2 prediction data was stored in '%s' under '%s'",
                self.output_path,
                f"{DL2_SUBARRAY_GROUP}/energy/{self.prefix}",
            )

        # Prepare the direction telescope table
        direction_tel_table = dl2_table.copy()
        direction_tel_table.keep_columns(
            TELESCOPE_EVENT_KEYS
            + [
                f"{self.prefix}_tel_alt",
                f"{self.prefix}_tel_az",
                f"{self.prefix}_tel_is_valid"
            ]
        )
        # Add the default values and meta data to the table
        add_defaults_and_meta(
            direction_tel_table,
            ReconstructedGeometryContainer,
            prefix=self.prefix,
            add_tel_prefix=True,
        )
        # Save the prediction to the output file
        if self.dl2_telescope:
            write_table(
                direction_tel_table,
                self.output_path,
                f"{DL2_TELESCOPE_GROUP}/geometry/{self.prefix}/tel_{self.tel_id:03d}",
                overwrite=self.overwrite,
            )
            self.log.info(
                "DL2 prediction data was stored in '%s' under '%s'",
                self.output_path,
                f"{DL2_TELESCOPE_GROUP}/geometry/{self.prefix}/tel_{self.tel_id:03d}",
            )
        # Write the mono telescope prediction to the subarray prediction table
        if self.dl2_subarray:
            direction_subarray_table = direction_tel_table.copy()
            direction_subarray_table.remove_column("tel_id")
            for colname in direction_subarray_table.colnames:
                if "_tel_" in colname:
                    direction_subarray_table.rename_column(
                        colname, colname.replace("_tel", "")
                    )
            direction_subarray_table.add_column(telescopes_flag)
            # Save the prediction to the output file
            write_table(
                direction_subarray_table,
                self.output_path,
                f"{DL2_SUBARRAY_GROUP}/geometry/{self.prefix}",
                overwrite=self.overwrite,
            )
            self.log.info(
                "DL2 prediction data was stored in '%s' under '%s'",
                self.output_path,
                f"{DL2_SUBARRAY_GROUP}/geometry/{self.prefix}",
            )

    def finish(self):
        self.log.info("Tool is shutting down")

def main():
    # Run the tool
    tool = DL2Converter()
    tool.run()

if __name__ == "main":
    main()
