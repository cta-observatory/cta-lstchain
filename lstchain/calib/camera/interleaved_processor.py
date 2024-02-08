from ctapipe.containers import ArrayEventContainer
from ctapipe.core import TelescopeComponent


class LSTInterleavedProcessor(TelescopeComponent):
    def __init__(self, subarray, config=None, parent=None, **kwargs):
        super().__init__(subarray, config, parent, **kwargs)

    def __call__(self, event: ArrayEventContainer) -> None:
        pass
