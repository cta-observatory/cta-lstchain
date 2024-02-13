from ctapipe.image import ImageProcessor
from ctapipe.containers import ArrayEventContainer, ImageParametersContainer
from ctapipe.instrument import SubarrayDescription

from cleaning import LSTImageCleaner

DEFAULT_TRUE_IMAGE_PARAMETERS = ImageParametersContainer()

__all__ = ["LSTImageProcessor"]


class LSTImageProcessor(ImageProcessor):
    """
    LST specific image processor which takes DL1/Image data and cleans and parametrizes
    the images into DL1/parameters. It inherits from ctapipes `ImageProcessor` and adapts
    the cleaner to the LSTImageCleaner with a different API (`ArrayEventContainer` as input).
    """

    def __init__(
        self, subarray: SubarrayDescription, config=None, parent=None, **kwargs
    ):
        """
        Parameters
        ----------
        subarray: SubarrayDescription
            Description of the subarray. Provides information about the
            camera which are useful in calibration. Also required for
            configuring the TelescopeParameter traitlets.
        config: traitlets.loader.Config
            Configuration specified by config file or cmdline arguments.
            Used to set traitlet values.
            This is mutually exclusive with passing a ``parent``.
        parent: ctapipe.core.Component or ctapipe.core.Tool
            Parent of this component in the configuration hierarchy,
            this is mutually exclusive with passing ``config``
        """

        super().__init__(subarray=subarray, config=config, parent=parent, **kwargs)
        self.clean = LSTImageCleaner(subarray=subarray, parent=self)

    def _process_telescope_event(self, event: ArrayEventContainer):
        """
        Loop over telescopes and process the calibrated images into parameters
        """
        for tel_id, dl1_camera in event.dl1.tel.items():
            if self.apply_image_modifier.tel[tel_id]:
                dl1_camera.image = self.modify(tel_id=tel_id, image=dl1_camera.image)

            dl1_camera.image_mask = self.clean(event, tel_id)

            dl1_camera.parameters = self._parameterize_image(
                tel_id=tel_id,
                image=dl1_camera.image,
                signal_pixels=dl1_camera.image_mask,
                peak_time=dl1_camera.peak_time,
                default=self.default_image_container,
            )

            self.log.debug("params: %s", dl1_camera.parameters.as_dict(recursive=True))

            if (
                event.simulation is not None
                and tel_id in event.simulation.tel
                and event.simulation.tel[tel_id].true_image is not None
            ):
                sim_camera = event.simulation.tel[tel_id]
                sim_camera.true_parameters = self._parameterize_image(
                    tel_id,
                    image=sim_camera.true_image,
                    signal_pixels=sim_camera.true_image > 0,
                    peak_time=None,  # true image from simulation has no peak time
                    default=DEFAULT_TRUE_IMAGE_PARAMETERS,
                )
                for container in sim_camera.true_parameters.values():
                    if not container.prefix.startswith("true_"):
                        container.prefix = f"true_{container.prefix}"

                self.log.debug(
                    "sim params: %s",
                    event.simulation.tel[tel_id].true_parameters.as_dict(
                        recursive=True
                    ),
                )
