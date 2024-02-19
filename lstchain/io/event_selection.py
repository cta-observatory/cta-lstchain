import numpy as np
import operator
import astropy.units as u

from ctapipe.core import Component
from ctapipe.containers import EventType
from ctapipe.core.traits import Dict, List, Float, Int
from lstchain.reco.utils import filter_events


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
        Apply the standard event filters.
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
        default_value=0.1,
    ).tag(config=True)

    max_theta_cut = Float(
        help="Maximum theta cut (deg) in an energy bin",
        default_value=0.32,
    ).tag(config=True)

    fill_theta_cut = Float(
        help=(
            "Fill value of theta cut (deg) in an energy bin with fewer "
            + "than minimum number of events present"
        ),
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
        default_value=20,
    ).tag(config=True)

    fill_alpha_cut = Float(
        help=(
            "Fill value of alpha cut (deg) in an energy bin with fewer "
            + "than minimum number of events present"
        ),
        default_value=20,
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

    def update_fill_cuts(self, cut_table):
        """
        For an energy-dependent cut table, update the cuts for bins with number
        of events fewer than the minimum number, for which pyirf uses a
        constant fill_value, usually at the energy threshold limits, with cut
        evaluated at the nearest bin with number of events more than the given
        minimum.

        In the case, where the low-events bin is in between high-events bins,
        the cut value for that low-events bin is taken as the mean of the
        neighbouring cut values.
        """
        cut_table_new = cut_table.copy()

        low_event_bins = np.nonzero(
            cut_table["n_events"] < self.min_event_p_en_bin
        )[0]
        high_event_bins = np.nonzero(
            cut_table["n_events"] >= self.min_event_p_en_bin
        )[0]

        for low_ in low_event_bins:
            if low_ < high_event_bins[0]:
                cut_table_new["cut"].value[low_] = cut_table["cut"].value[
                    high_event_bins[0]
                ]
            elif low_ > high_event_bins[-1]:
                cut_table_new["cut"].value[low_] = cut_table["cut"].value[
                    high_event_bins[-1]
                ]
            else:
                cut_table_new["cut"].value[low_] = np.mean([
                    cut_table["cut"].value[low_-1],
                    cut_table["cut"].value[low_+1]
                ])

        return cut_table_new

    def energy_dependent_gh_cuts(self, data, energy_bins, smoothing=None):
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
        if (gh_cuts["n_events"] < self.min_event_p_en_bin).any():
            gh_cuts = self.update_fill_cuts(gh_cuts)

        return gh_cuts

    def apply_energy_dependent_gh_cuts(self, data, gh_cuts):
        """
        Applying a given energy-dependent gh cuts on the given data, along the
        reco energy bins.
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
        self,
        data,
        energy_bins,
        use_same_disp_sign=True,
        smoothing=None,
    ):
        """
        Evaluating energy-dependent theta cuts, in a given MC data,
        with provided reco energy bins, and other parameters to pass to the
        pyirf.cuts.calculate_percentile_cut function.

        For MC events, the disp_sign may be reconstructed incorrectly with
        respect to the true value, and thus resulting in a bi-modal PSF.
        For evaluating the energy-dependent theta cuts, we want to consider,
        only the central region of PSF. To fix this issue, by default, we apply
        a mask on the data, so as to only use events with the same disp_sign
        after reconstruction, for evaluating the percentile cut.

        Note: In this case, at low energies, where disp_sign determination is
        pretty uncertain, an efficiency of 40% or larger will result in a cut
        which keeps the whole central region of the PSF.

        If the user wishes to not use this method, they can make the boolean
        use_same_disp_sign as False.

        Note: Using too fine binning will result in too un-smooth cuts.
        """
        if use_same_disp_sign:
            disp_mask = data["reco_disp_sign"] == data["disp_sign"]
            data = data[disp_mask]

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

        if (theta_cuts["n_events"] < self.min_event_p_en_bin).any():
            theta_cuts = self.update_fill_cuts(theta_cuts)

        return theta_cuts

    def apply_energy_dependent_theta_cuts(self, data, theta_cuts):
        """
        Applying a given energy-dependent theta cuts on a given data, along the
        reco energy bins.
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

    def energy_dependent_alpha_cuts(self, data, energy_bins, smoothing=None):
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

        if (alpha_cuts["n_events"] < self.min_event_p_en_bin).any():
            alpha_cuts = self.update_fill_cuts(alpha_cuts)

        return alpha_cuts

    def apply_energy_dependent_alpha_cuts(self, data, alpha_cuts):
        """
        Applying a given energy-dependent alpha cuts on a given data, along the
        reco energy bins.
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
        default_value=500,
    ).tag(config=True)

    true_energy_n_bins = Int(
        help="Number of bins in log scale for True Energy",
        default_value=25,
    ).tag(config=True)

    reco_energy_min = Float(
        help="Minimum value for Reco Energy bins in TeV units",
        default_value=0.005,
    ).tag(config=True)

    reco_energy_max = Float(
        help="Maximum value for Reco Energy bins in TeV units",
        default_value=500,
    ).tag(config=True)

    reco_energy_n_bins = Int(
        help="Number of bins in log scale for Reco Energy",
        default_value=25,
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
        default_value=30,
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
        Creates bins for true energy in log scale using numpy.geomspace with
        true_energy_n_bins + 1 edges.
        """
        true_energy = np.geomspace(
            self.true_energy_min * u.TeV,
            self.true_energy_max * u.TeV,
            self.true_energy_n_bins + 1,
        )
        return true_energy

    def reco_energy_bins(self):
        """
        Creates bins for reco energy in log scale using numpy.geomspace with
        reco_energy_n_bins + 1 edges.
        """
        reco_energy = np.geomspace(
            self.reco_energy_min * u.TeV,
            self.reco_energy_max * u.TeV,
            self.reco_energy_n_bins + 1,
        )
        return reco_energy

    def energy_migration_bins(self):
        """
        Creates bins for energy migration in log scale using numpy.geomspace
        with energy_migration_n_bins + 1 edges.
        """
        energy_migration = np.geomspace(
            self.energy_migration_min,
            self.energy_migration_max,
            self.energy_migration_n_bins + 1,
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
