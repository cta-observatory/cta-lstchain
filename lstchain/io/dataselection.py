from ctapipe.core import Component
from ctapipe.core.traits import Int, Float, List, Dict
from lstchain.reco.utils import filter_events

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

    event_filters = Dict(
        help="Dict of event filter parameters",
        default_value={
            "intensity": [0, np.inf],
            "length": [0, np.inf],
            "width": [0, np.inf],
            "r": [0, 1],
            "wl": [0.01, 1],
            "leakage_intensity_width_2": [0, 1],
        },
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

    lst_tel_ids = List(
        help="List of selected LST telescope ids",
        trait=Int(),
        default_value=[1],
    ).tag(config=True)

    def filter_cut(self, data):
        return filter_events(data, self.event_filters)

    def gh_cut(self, data):
        return data[data["gh_score"] > self.fixed_gh_cut]

    def theta_cut(self, data):
        return data[data["theta"] < u.Quantity(
            self.fixed_theta_cut
            ) * u.deg
        ]

    def true_src_fov_offset_cut(self, data):
        return data[
                data["true_source_fov_offset"] < u.Quantity(
                    self.fixed_source_fov_offset_cut
                    ) * u.deg
            ]

    def reco_src_fov_offset_cut(self, data):
        return data[
                data["reco_source_fov_offset"] < u.Quantity(
                    self.fixed_source_fov_offset_cut
                    ) * u.deg
            ]

    def tel_ids_filter(self, data):
        for i in self.lst_tel_ids:
            data["sel_tel"] = data["tel_id"] == i
        return data[data["sel_tel"]]


class DataBinning(Component):
    """
    Collects information on generating energy and angular bins for
    generating IRFs as per pyIRF requirements.
    """

    energy_bins = Dict(
        help="Binning Dict information for the 3 Energy bins",
        default_value={
            "true_energy_min": 0.01,
            "true_energy_max": 100,
            "true_energy_n_bins_per_decade": 5.5,
            "reco_energy_min": 0.01,
            "reco_energy_max": 100,
            "reco_energy_n_bins_per_decade": 5.5,
            "energy_migration_min": 0.2,
            "energy_migration_max": 5,
            "energy_migration_n_bins": 31
        },
    ).tag(config=True)

    angular_bins = Dict(
        help="Binning Dict information for 3 angular bins",
        default_value={
            "fov_offset_min": 0.3,
            "fov_offset_max": 0.7,
            "fov_offset_n_edges": 3,
            "bkg_fov_offset_min": 0,
            "bkg_fov_offset_max": 10,
            "bkg_fov_offset_n_edges": 21,
            "source_offset_min": 0.0001,
            "source_offset_max": 1.0001,
            "source_offset_n_edges": 1000
        },
    ).tag(config=True)

    def true_energy_bins(self):
        """
        Creates bins per decade for true MC energy using pyirf function.

        The overflow binning added is not needed at the current stage
        It can be used as - add_overflow_bins(***)[1:-1]
        """
        true_energy = create_bins_per_decade(
            self.energy_bins["true_energy_min"] * u.TeV,
            self.energy_bins["true_energy_max"] * u.TeV,
            self.energy_bins["true_energy_n_bins_per_decade"],
        )
        return true_energy

    def reco_energy_bins(self):
        """
        Creates bins per decade for reconstructed MC energy using pyirf function.

        The overflow binning added is not needed at the current stage
        It can be used as - add_overflow_bins(***)[1:-1]
        """
        reco_energy = create_bins_per_decade(
            self.energy_bins["reco_energy_min"] * u.TeV,
            self.energy_bins["reco_energy_max"] * u.TeV,
            self.energy_bins["reco_energy_n_bins_per_decade"],
        )
        return reco_energy

    def energy_migration_bins(self):
        """
        Creates bins for energy migration.
        """
        energy_migration = np.geomspace(
            self.energy_bins["energy_migration_min"],
            self.energy_bins["energy_migration_max"],
            self.energy_bins["energy_migration_n_bins"],
        )
        return energy_migration

    def fov_offset_bins(self):
        """
        Creates bins for single/multiple FoV offset
        """
        fov_offset = (
            np.linspace(
                self.angular_bins["fov_offset_min"],
                self.angular_bins["fov_offset_max"],
                self.angular_bins["fov_offset_n_edges"],
            ) * u.deg
        )
        return fov_offset

    def bkg_fov_offset_bins(self):
        """
        Creates bins for FoV offset for Background IRF,
        Using the same binning as in pyirf example.
        """
        background_offset = (
            np.linspace(
                self.angular_bins["bkg_fov_offset_min"],
                self.angular_bins["bkg_fov_offset_max"],
                self.angular_bins["bkg_fov_offset_n_edges"],
            ) * u.deg
        )
        return background_offset

    def source_offset_bins(self):
        """
        Creates bins for source offset for generating PSF IRF.
        Using the same binning as in pyirf example.
        """

        source_offset = (
            np.linspace(
                self.angular_bins["source_offset_max"],
                self.angular_bins["source_offset_max"],
                self.angular_bins["source_offset_n_edges"],
            ) * u.deg
        )
        return source_offset
