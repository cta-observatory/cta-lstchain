from ctapipe.core import Component
from ctapipe.core.traits import Float, List
import numpy as np
import astropy.units as u
from pyirf.binning import create_bins_per_decade  # , add_overflow_bins

__all__ = ["DataSelection", "DataBinning"]


class DataSelection(Component):
    """
    Collect various selection cuts to be applied for IRF production and
    DL3 data reduction

    Parameters for event filters will be combined in a dict so that the
    filter_events() can be used.
    """
    intensity = List(
        help="Range of intensity of event filter",
        default_value=[100, np.inf],
    ).tag(config=True)

    length = List(
        help="Range of length for event filter",
        default_value=[0, np.inf],
    ).tag(config=True)

    width = List(
        help="Range of width for event filter",
        default_value=[0, np.inf],
    ).tag(config=True)

    r = List(
        help="Range of r for event filter",
        default_value=[0, 1],
    ).tag(config=True)

    wl = List(
        help="Range of wl for event filter",
        default_value=[0.1, 1],
    ).tag(config=True)

    leakage_intensity_width_2 = List(
        help="Range for leakage_intensity_width_2 of event filter",
        default_value=[0, 0.2],
    ).tag(config=True)

    fixed_gh_cut = Float(
        help="Fixed selection cut for gh_score (gammaness)",
        default_value=0.6,
    ).tag(config=True)

    fixed_theta_cut = Float(
        help="Fixed selection cut for theta",
        default_value=0.2,
    ).tag(config=True)

    fixed_source_fov_offset_cut = Float(
        help="Fixed selection cut for source FoV offset",
        default_value=2.83,
    ).tag(config=True)

    src_dep_alpha = Float(
        help="Selection cut for source dependent parameter - alpha",
        default_value=8.,
    ).tag(config=True)

    lst_tel_ids = List(
        help="List of selected LST telescope ids",
        default_value=[1],
    ).tag(config=True)

    magic_tel_ids = List(
        help="List of selected MAGIC telescope ids",
        default_value=[1, 2],
    ).tag(config=True)

    aliases = {
        ("int", "intensity"): "DataSelection.intensity",
        ("len", "length"): "DataSelection.length",
        ("w", "width"): "DataSelection.width",
        "r": "DataSelection.r",
        "wl": "DataSelection.wl",
        ("leak_2", "leakage_intensity_width_2"):
            "DataSelection.leakage_intensity_width_2",
        ("gh", "fixed_gh_cut"): "DataSelection.fixed_gh_cut",
        ("theta", "fixed_theta_cut"): "DataSelection.fixed_theta_cut",
        ("src_fov", "fixed_source_fov_offset_cut"):
            "DataSelection.fixed_source_fov_offset_cut",
        ("alpha", "src_dep_alpha"): "DataSelection.src_dep_alpha",
        "lst_tel_ids": "DataSelection.lst_tel_ids",
        "magic_tel_ids": "DataSelection.magic_tel_ids",
    }

    def __init__(self, parent=None, **kwargs):
        super().__init__(parent=parent, **kwargs)

    def event_filters(self):
        """
        Creates a dict for filter_events() function
        """
        self.evt_filter = {
            "intensity": self.intensity,
            "width": self.width,
            "length": self.length,
            "r": self.r,
            "wl": self.wl,
            "leakage_intensity_width_2": self.leakage_intensity_width_2,
        }
        return self.evt_filter


class DataBinning(Component):
    """
    Collects information on generating energy and angular bins for
    generating IRFs as per pyIRF requirements.
    """
    true_energy_bins = List(
        help="Values to get true energy (TeV) binning as "
        "[e_min, e_max, bins_per_decade]",
        default_value=[0.01, 100, 5.5],
    ).tag(config=True)

    reco_energy_bins = List(
        help="Values to get reco energy (TeV) binning as "
        "[e_min, e_max, bins_per_decade]",
        default_value=[0.01, 100, 5.5],
    ).tag(config=True)

    energy_migra_bins = List(
        help="Values to get energy migration binning as [min, max, bins]",
        default_value=[0.2, 5, 31],
    ).tag(config=True)

    single_fov_offset_bins = List(
        help="List of bins for single FOV offset binning",
        default_value=[0.3, 0.5, 0.7],
    ).tag(config=True)

    multiple_fov_offset_bins = List(
        help="List of bins for multiple FOV offset binning",
        default_value=[0, 0.3, 0.5, 0.7, 0.9, 1.1],
    ).tag(config=True)

    bkg_fov_offset_bins = List(
        help="Range of values for multiple FOV offset binning "
        "for Background IRF as [o_min, o_max]",
        default_value=[0, 11],
    ).tag(config=True)

    source_offset_bins = List(
        help="Values to get source offset binning for PSF IRF "
        "as [o_min, o_max, bin_width]",
        default_value=[0, 1.0001, 0.001],
    ).tag(config=True)

    aliases = {
        ("etrue", "true_energy_bins"): "DataBinning.true_energy_bins",
        ("ereco", "reco_energy_bins"): "DataBinning.reco_energy_bins",
        ("emigra", "energy_migra_bins"): "DataBinning.energy_migra_bins",
        ("sing_fov", "single_fov_offset_bins"):
            "DataBinning.single_fov_offset_bins",
        ("mult_fov", "multiple_fov_offset_bins"):
            "DataBinning.multiple_fov_offset_bins",
        ("bkg_fov", "bkg_fov_offset_bins"): "DataBinning.bkg_fov_offset_bins",
        ("src_off", "source_offset_bins"): "DataBinning.source_offset_bins",
    }

    def __init__(self, parent=None, **kwargs):
        super().__init__(parent=parent, **kwargs)

    def true_energy(self):
        """
        Creates bins per decade for true MC energy using pyirf function.

        The overflow binning added is not needed at the current stage
        It can be used as - add_overflow_bins(***)[1:-1]
        """
        self.true_energy = create_bins_per_decade(
            self.true_energy_bins[0] * u.TeV,
            self.true_energy_bins[1] * u.TeV,
            self.true_energy_bins[2],
        )

        return self.true_energy

    def reco_energy(self):
        """
        Creates bins per decade for reconstructed MC energy using pyirf function.

        The overflow binning added is not needed at the current stage
        It can be used as - add_overflow_bins(***)[1:-1]
        """
        self.reco_energy = create_bins_per_decade(
            self.reco_energy_bins[0] * u.TeV,
            self.reco_energy_bins[1] * u.TeV,
            self.reco_energy_bins[2],
        )
        return self.reco_energy

    def energy_migration(self):
        """
        Creates bins for energy migration.
        """
        self.energy_migration = np.linspace(
            self.energy_migra_bins[0],
            self.energy_migra_bins[1],
            self.energy_migra_bins[2],
        )
        return self.energy_migration

    def source_offset(self):
        """
        Creates bins for source offset for generating PSF IRF.
        Using the same binning as in pyirf example.
        """

        self.source_offset = (
            np.arange(
                self.source_offset_bins[0],
                self.source_offset_bins[1],
                self.source_offset_bins[2],
            )
            * u.deg
        )
        return self.source_offset

    def background_offset(self):
        """
        Creates bins for FoV offset for Background IRF,
        Using the same binning as in pyirf example.
        """
        self.background_offset = (
            np.arange(
                self.bkg_fov_offset_bins[0],
                self.bkg_fov_offset_bins[1]
            )
            * u.deg
        )
        return self.background_offset
