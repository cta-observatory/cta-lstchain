from ctapipe.core import Component
from ctapipe.core.traits import Int, Float, List
from lstchain.reco.utils import filter_events

import numpy as np
import astropy.units as u
from astropy.table import QTable
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
        trait=Float(),
        default_value=[0, np.inf],
    ).tag(config=True)

    length = List(
        help="Range of length for event filter",
        trait=Float(),
        default_value=[0, np.inf],
    ).tag(config=True)

    width = List(
        help="Range of width for event filter",
        trait=Float(),
        default_value=[0, np.inf],
    ).tag(config=True)

    r = List(
        help="Range of r for event filter",
        trait=Float(),
        default_value=[0, 1],
    ).tag(config=True)

    wl = List(
        help="Range of wl for event filter",
        trait=Float(),
        default_value=[0.01, 1],
    ).tag(config=True)

    leakage_intensity_width_2 = List(
        help="Range for leakage_intensity_width_2 of event filter",
        trait=Float(),
        default_value=[0, 1],
    ).tag(config=True)

    fixed_gh_cut = Float(
        help="Fixed selection cut for gh_score (gammaness)",
        default_value=0.5,
    ).tag(config=True)

    fixed_theta_cut = Float(
        help="Fixed selection cut for theta",
        default_value=0.5,
    ).tag(config=True)

    fixed_source_fov_offset_cut = Float(
        help="Fixed selection cut for source FoV offset",
        default_value=3,
    ).tag(config=True)

    lst_tel_ids = List(
        help="List of selected LST telescope ids",
        trait=Int(),
        default_value=[1],
    ).tag(config=True)

    def __init__(self, parent=None, **kwargs):
        super().__init__(parent=parent, **kwargs)

    def __call__(self, data):
        """
        Check if the table passed is QTable or not
        """
        if type(data).__name__ != 'QTable':
            self.log.debug("Data table is not of QTable class")
            pass

    def filter_cut(self, data):
        return filter_events(data, self.event_filters())

    def gh_cut(self, data):
        return data[data["gh_score"] > self.fixed_gh_cut]

    def theta_cut(self, data):
        return data[data["theta"] < u.Quantity(self.fixed_theta_cut) * u.deg]

    def true_src_fov_offset_cut(self, data):
        return data[
                data["true_source_fov_offset"] < u.Quantity(
                    self.fixed_source_fov_offset_cut) * u.deg
            ]

    def reco_src_fov_offset_cut(self, data):
        return data[
                data["reco_source_fov_offset"] < u.Quantity(
                    self.fixed_source_fov_offset_cut) * u.deg
            ]

    def tel_ids_filter(self, data):
        for i in self.lst_tel_ids:
            data["sel_tel"] = data["tel_id"] == i
        return data[data["sel_tel"]]

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
        trait=Float(),
        default_value=[0.01, 100, 5.5],
    ).tag(config=True)

    reco_energy_bins = List(
        help="Values to get reco energy (TeV) binning as "
        "[e_min, e_max, bins_per_decade]",
        trait=Float(),
        default_value=[0.01, 100, 5.5],
    ).tag(config=True)

    energy_migra_bins = List(
        help="Values to get energy migration binning as [min, max, bins]",
        default_value=[0.2, 5, 31],
    ).tag(config=True)

    single_fov_offset_bins = List(
        help="List of bins for single FOV offset binning",
        trait=Float(),
        default_value=[0.3, 0.5, 0.7],
    ).tag(config=True)

    multiple_fov_offset_bins = List(
        help="List of bins for multiple FOV offset binning",
        trait=Float(),
        default_value=[0, 0.3, 0.5, 0.7, 0.9, 1.1],
    ).tag(config=True)

    bkg_fov_offset_bins = List(
        help="Range of values for multiple FOV offset binning "
        "for Background IRF as [o_min, o_max]",
        trait=Float(),
        default_value=[0, 11],
    ).tag(config=True)

    source_offset_bins = List(
        help="Values to get source offset binning for PSF IRF "
        "as [o_min, o_max, bin_width]",
        trait=Float(),
        default_value=[0, 1.0001, 0.001],
    ).tag(config=True)

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
            np.arange(self.bkg_fov_offset_bins[0], self.bkg_fov_offset_bins[1]) * u.deg
        )
        return self.background_offset
