from ctapipe.core import Component
from ctapipe.core.traits import Dict, List
from lstchain.reco.utils import filter_events

import numpy as np


__all__ = ["EventSelector"]


class EventSelector(Component):
    """
    Filter values used for event filters and list of finite parameters are taken as inputs and
    filter_events() is used on a table of events called in with the Component.
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

