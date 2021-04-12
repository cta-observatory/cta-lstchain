from ctapipe.core import Component
from ctapipe.core.traits import Dict, List, Float, Int
from lstchain.reco.utils import filter_events

import numpy as np
import astropy.units as u
from pyirf.binning import create_bins_per_decade  # , add_overflow_bins


__all__ = ["EventSelector", "DL3FixedCuts", "DataBinning"]


class EventSelector(Component):
    """
    Filter values used for event filters and list of finite parameters are
    taken as inputs and filter_events() is used on a table of events
    called in with the Component.
    """

    filters = Dict(
        help="Dict of event filter parameters",
        default_value={
            "r": [0, 1],
            "wl": [0.01, 1],
            "leakage_intensity_width_2": [0, 1],
        },
    ).tag(config=True)

    finite_params = List(
        help="List of parameters to ensure finite values",
        default_value=["intensity", "length", "width"],
    ).tag(config=True)

    def filter_cut(self, events):
        return filter_events(events, self.filters, self.finite_params)


class DL3FixedCuts(Component):
    """
    Temporary fixed selection cuts for DL2 to DL3 conversion
    """

    fixed_gh_cut = Float(
        help="Fixed selection cut for gh_score (gammaness)",
        default_value=0.6,
    ).tag(config=True)

    fixed_theta_cut = Float(
        help="Fixed selection cut for theta",
        default_value=0.2,
    ).tag(config=True)

    allowed_tels = List(
        help="List of allowed LST telescope ids",
        trait=Int(),
        default_value=[1],
    ).tag(config=True)

    def gh_cut(self, data):
        return data[data["gh_score"] > self.fixed_gh_cut]

    def theta_cut(self, data):
        return data[data["theta"].to_value(u.deg) < self.fixed_theta_cut]

    def allowed_tels_filter(self, data):
        mask = np.zeros(len(data), dtype=bool)
        for tel_id in self.allowed_tels:
            mask |= data["tel_id"] == tel_id
        return data[mask]


class DataBinning(Component):
    """
    Collects information on generating energy and angular bins for
    generating IRFs as per pyIRF requirements.
    """

    true_energy_min = Float(
        help="Minimum value for True Energy bins in TeV units",
        default_value=0.01,
    ).tag(config=True)

    true_energy_max = Float(
        help="Maximum value for True Energy bins in TeV units",
        default_value=100,
    ).tag(config=True)

    true_energy_n_bins_per_decade = Float(
        help="Number of edges per decade for True Energy bins",
        default_value=5.5,
    ).tag(config=True)

    reco_energy_min = Float(
        help="Minimum value for Reco Energy bins in TeV units",
        default_value=0.01,
    ).tag(config=True)

    reco_energy_max = Float(
        help="Maximum value for Reco Energy bins in TeV units",
        default_value=100,
    ).tag(config=True)

    reco_energy_n_bins_per_decade = Float(
        help="Number of edges per decade for Reco Energy bins",
        default_value=5.5,
    ).tag(config=True)

    energy_migration_min = Float(
        help="Minimum value of Energy Migration matrix",
        default_value=0.2,
    ).tag(config=True)

    energy_migration_max = Float(
        help="Maximum value of Energy Migration matrix",
        default_value=5,
    ).tag(config=True)

    energy_migration_n_bins = Int(
        help="Number of bins in log scale for Energy Migration matrix",
        default_value=31,
    ).tag(config=True)

    fov_offset_min = Float(
        help="Minimum value for FoV Offset bins",
        default_value=0.1,
    ).tag(config=True)

    fov_offset_max = Float(
        help="Maximum value for FoV offset bins",
        default_value=1.1,
    ).tag(config=True)

    fov_offset_n_edges = Int(
        help="Number of edges for FoV offset bins",
        default_value=9,
    ).tag(config=True)

    bkg_fov_offset_min = Float(
        help="Minimum value for FoV offset bins for Background IRF",
        default_value=0,
    ).tag(config=True)

    bkg_fov_offset_max = Float(
        help="Maximum value for FoV offset bins for Background IRF",
        default_value=10,
    ).tag(config=True)

    bkg_fov_offset_n_edges = Int(
        help="Number of edges for FoV offset bins for Background IRF",
        default_value=21,
    ).tag(config=True)

    source_offset_min = Float(
        help="Minimum value for Source offset for PSF IRF",
        default_value=0.0001,
    ).tag(config=True)

    source_offset_max = Float(
        help="Maximum value for Source offset for PSF IRF",
        default_value=1.0001,
    ).tag(config=True)

    source_offset_n_edges = Int(
        help="Number of edges for Source offset for PSF IRF",
        default_value=1000,
    ).tag(config=True)

    def true_energy_bins(self):
        """
        Creates bins per decade for true MC energy using pyirf function.

        The overflow binning added is not needed at the current stage
        It can be used as - add_overflow_bins(***)[1:-1]
        """
        true_energy = create_bins_per_decade(
            self.true_energy_min * u.TeV,
            self.true_energy_max * u.TeV,
            self.true_energy_n_bins_per_decade,
        )
        return true_energy

    def reco_energy_bins(self):
        """
        Creates bins per decade for reconstructed MC energy using pyirf function.

        The overflow binning added is not needed at the current stage
        It can be used as - add_overflow_bins(***)[1:-1]
        """
        reco_energy = create_bins_per_decade(
            self.reco_energy_min * u.TeV,
            self.reco_energy_max * u.TeV,
            self.reco_energy_n_bins_per_decade,
        )
        return reco_energy

    def energy_migration_bins(self):
        """
        Creates bins for energy migration.
        """
        energy_migration = np.geomspace(
            self.energy_migration_min,
            self.energy_migration_max,
            self.energy_migration_n_bins,
        )
        return energy_migration

    def fov_offset_bins(self):
        """
        Creates bins for single/multiple FoV offset
        """
        fov_offset = (
            np.linspace(
                self.fov_offset_min,
                self.fov_offset_max,
                self.fov_offset_n_edges,
            )
            * u.deg
        )
        return fov_offset

    def bkg_fov_offset_bins(self):
        """
        Creates bins for FoV offset for Background IRF,
        Using the same binning as in pyirf example.
        """
        background_offset = (
            np.linspace(
                self.bkg_fov_offset_min,
                self.bkg_fov_offset_max,
                self.bkg_fov_offset_n_edges,
            )
            * u.deg
        )
        return background_offset

    def source_offset_bins(self):
        """
        Creates bins for source offset for generating PSF IRF.
        Using the same binning as in pyirf example.
        """

        source_offset = (
            np.linspace(
                self.source_offset_min,
                self.source_offset_max,
                self.source_offset_n_edges,
            )
            * u.deg
        )
        return source_offset
