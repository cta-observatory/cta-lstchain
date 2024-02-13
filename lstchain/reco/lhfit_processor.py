from ctapipe.containers import ArrayEventContainer
from ctapipe.core import TelescopeComponent
from ctapipe.core.traits import Bool


class LHFitProcessor(TelescopeComponent):
    """ """

    compute_lhfit_parameters = Bool(
        help="Compute and store LHFit Parameters",
        default_value=False,
    ).tag(config=True)

    def __init__(self, subarray, config=None, parent=None, **kwargs):
        super().__init__(config, parent, **kwargs)

    def __call__(self, event: ArrayEventContainer):
        pass
