import numpy as np
import operator
import astropy.units as u

from ctapipe.containers import EventType
from ctapipe.core.traits import Dict, List, Float, Int
from lstchain.reco.utils import filter_events

from lstchain.ctapipe_compat import Component

from pyirf.binning import create_bins_per_decade  # , add_overflow_bins
from pyirf.cuts import calculate_percentile_cut, evaluate_binned_cut


__all__ = ["EventSelector", "DL3Cuts", "DataBinning"]


class EventSelector(Component):
    """
    Filter values used for event filters and list of finite parameters are
    taken as inputs and filter_events() is used on a table of events
    called in with the Component.

    For event_type, we choose the sub-array trigger, EventType.SUBARRAY.value,
    32, which is for shower event candidate, as per the latest CTA R1 Event
    Data Model.
    """

    filters = Dict(
        help="Dict of event filter parameters",
        default_value={
            "r": [0, 1],
            "wl": [0.01, 1],
            "leakage_intensity_width_2": [0, 1],
            "event_type": [EventType.SUBARRAY.value, EventType.SUBARRAY.value],
        },
    ).tag(config=True)

    finite_params = List(
        help="List of parameters to ensure finite values",
        default_value=["intensity", "length", "width"],
    ).tag(config=True)

    def filter_cut(self, events):
        """
        Apply the event filters
        """
        return filter_events(events, self.filters, self.finite_params)


class DL3Cuts(Component):
    """
    Selection cuts for DL2 to DL3 conversion
    """

    min_event_p_en_bin = Float(
        help="Minimum events per energy bin, to evaluate percentile cuts",
        default_value=100,
    ).tag(config=True)

    global_gh_cut = Float(
        help="Global selection cut for gh_score (gammaness)",
        default_value=0.6,
    ).tag(config=True)

    gh_efficiency = Float(
        help="Gamma efficiency for optimized g/h cuts in %",
        default_value=0.95,
    ).tag(config=True)

    min_gh_cut = Float(
        help="Minimum gh_score (gammaness) cut in an energy bin",
        default_value=0.1,
    ).tag(config=True)

    max_gh_cut = Float(
        help="Maximum gh_score (gammaness) cut in an energy bin",
        default_value=0.95,
    ).tag(config=True)

    min_theta_cut = Float(
        help="Minimum theta cut (deg) in an energy bin",
        default_value=0.05,
    ).tag(config=True)

    max_theta_cut = Float(
        help="Maximum theta cut (deg) in an energy bin",
        default_value=0.32,
    ).tag(config=True)

    fill_theta_cut = Float(
        help="Fill value of theta cut (deg) in an energy bin with fewer " +
            "than minimum number of events present",
        default_value=0.32,
    ).tag(config=True)

    theta_containment = Float(
        help="Percentage containment region for theta cuts",
        default_value=0.68,
    ).tag(config=True)

    global_theta_cut = Float(
        help="Global selection cut (deg) for theta",
        default_value=0.2,
    ).tag(config=True)

    min_alpha_cut = Float(
        help="Minimum alpha cut (deg) in an energy bin",
        default_value=1,
    ).tag(config=True)

    max_alpha_cut = Float(
        help="Maximum alpha cut (deg) in an energy bin",
        default_value=45,
    ).tag(config=True)

    fill_alpha_cut = Float(
        help="Fill value of alpha cut (deg) in an energy bin with fewer " +
            "than minimum number of events present",
        default_value=45,
    ).tag(config=True)

    alpha_containment = Float(
        help="Percentage containment region for alpha cuts",
        default_value=0.68,
    ).tag(config=True)

    global_alpha_cut = Float(
        help="Global selection cut (deg) for alpha",
        default_value=20,
    ).tag(config=True)

    allowed_tels = List(
        help="List of allowed LST telescope ids",
        trait=Int(),
        default_value=[1],
    ).tag(config=True)

    def apply_global_gh_cut(self, data):
        """
        Applying a global gammaness cut on a given data
        """
        return data[data["gh_score"] > self.global_gh_cut]

    def energy_dependent_gh_cuts(
        self, data, energy_bins, smoothing=None
    ):
        """
        Evaluating energy-dependent gammaness cuts, in a given
        data, with provided reco energy bins, and other parameters to
        pass to the pyirf.cuts.calculate_percentile_cut function
        """

        gh_cuts = calculate_percentile_cut(
            data["gh_score"],
            data["reco_energy"],
            bins=energy_bins,
            min_value=self.min_gh_cut,
            max_value=self.max_gh_cut,
            fill_value=data["gh_score"].max(),
            percentile=100 * (1 - self.gh_efficiency),
            smoothing=smoothing,
            min_events=self.min_event_p_en_bin,
        )
        return gh_cuts


    def apply_energy_dependent_gh_cuts(self, data, gh_cuts):
        """
        Applying a given energy-dependent gh cuts to a data file, along the reco
        energy bins provided.
        """

        data["selected_gh"] = evaluate_binned_cut(
            data["gh_score"],
            data["reco_energy"],
            gh_cuts,
            operator.ge,
        )
        return data[data["selected_gh"]]

    def apply_global_theta_cut(self, data):
        """
        Applying a global theta cut on a given data
        """
        return data[data["theta"].to_value(u.deg) < self.global_theta_cut]

    def energy_dependent_theta_cuts(
        self, data, energy_bins, smoothing=None,
    ):
        """
        Evaluating an optimized energy-dependent theta cuts, in a given
        data, with provided reco energy bins, and other parameters to
        pass to the pyirf.cuts.calculate_percentile_cut function.

        Note: Using too fine binning will result in too un-smooth cuts.
        """

        theta_cuts = calculate_percentile_cut(
            data["theta"],
            data["reco_energy"],
            bins=energy_bins,
            min_value=self.min_theta_cut * u.deg,
            max_value=self.max_theta_cut * u.deg,
            fill_value=self.fill_theta_cut * u.deg,
            percentile=100 * self.theta_containment,
            smoothing=smoothing,
            min_events=self.min_event_p_en_bin,
        )
        return theta_cuts

    def apply_energy_dependent_theta_cuts(self, data, theta_cuts):
        """
        Applying a given energy-dependent theta cuts to a data file, along the
        reco energy bins provided.
        """

        data["selected_theta"] = evaluate_binned_cut(
            data["theta"],
            data["reco_energy"],
            theta_cuts,
            operator.le,
        )
        return data[data["selected_theta"]]

    def apply_global_alpha_cut(self, data):
        """
        Applying a global alpha cut on a given data
        """
        return data[data["alpha"].to_value(u.deg) < self.global_alpha_cut]

    def energy_dependent_alpha_cuts(
            self, data, energy_bins, smoothing=None
    ):
        """
        Evaluating an optimized energy-dependent alpha cuts, in a given
        data, with provided reco energy bins, and other parameters to
        pass to the pyirf.cuts.calculate_percentile_cut function.
        Note: Using too fine binning will result in too un-smooth cuts.
        """
        
        alpha_cuts = calculate_percentile_cut(
            data["alpha"],
            data["reco_energy"],
            bins=energy_bins,
            min_value=self.min_alpha_cut * u.deg,
            max_value=self.max_alpha_cut * u.deg,
            fill_value=self.fill_alpha_cut * u.deg,
            percentile=100 * self.alpha_containment,
            smoothing=smoothing,
            min_events=self.min_event_p_en_bin,
        )
        return alpha_cuts
        
    def apply_energy_dependent_alpha_cuts(self, data, alpha_cuts):
        """
        Applying a given energy-dependent alpha cuts to a data file, along the
        reco energy bins provided.
        """
        
        data["selected_alpha"] = evaluate_binned_cut(
            data["alpha"],
            data["reco_energy"],
            alpha_cuts,
            operator.le,
        )
        return data[data["selected_alpha"]]
            
    def allowed_tels_filter(self, data):
        """
        Applying a filter on telescopes used for observation.
        """
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
        default_value=0.005,
    ).tag(config=True)

    true_energy_max = Float(
        help="Maximum value for True Energy bins in TeV units",
        default_value=200,
    ).tag(config=True)

    true_energy_n_bins_per_decade = Float(
        help="Number of edges per decade for True Energy bins",
        default_value=5,
    ).tag(config=True)

    reco_energy_min = Float(
        help="Minimum value for Reco Energy bins in TeV units",
        default_value=0.005,
    ).tag(config=True)

    reco_energy_max = Float(
        help="Maximum value for Reco Energy bins in TeV units",
        default_value=200,
    ).tag(config=True)

    reco_energy_n_bins_per_decade = Float(
        help="Number of edges per decade for Reco Energy bins",
        default_value=5,
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
        default_value=0,
    ).tag(config=True)

    source_offset_max = Float(
        help="Maximum value for Source offset for PSF IRF",
        default_value=1,
    ).tag(config=True)

    source_offset_n_edges = Int(
        help="Number of edges for Source offset for PSF IRF",
        default_value=101,
    ).tag(config=True)

    def true_energy_bins(self):
        """
        Creates bins per decade for true MC energy using pyirf function.
        The overflow binning added is not needed at the current stage.

        Examples
        --------
        It can be used as:

        >>> add_overflow_bins(***)[1:-1]
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
        The overflow binning added is not needed at the current stage.

        Examples
        --------
        It can be used as:

        >>> add_overflow_bins(***)[1:-1]
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
        Creates bins for single/multiple FoV offset.
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
