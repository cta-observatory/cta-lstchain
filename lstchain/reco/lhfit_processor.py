from ctapipe.containers import ArrayEventContainer
from ctapipe.core import TelescopeComponent
from ctapipe.core.traits import Bool, Float, FloatTelescopeParameter, Int, Path


class LHFitProcessor(TelescopeComponent):
    """
    Class used to perform event reconstruction by fitting of a model on waveforms.
    """

    compute_lhfit_parameters = Bool(
        help="Compute and store LHFit Parameters",
        default_value=False,
    ).tag(config=True)
    sigma_s = FloatTelescopeParameter(
        default_value=1,
        help="Width of the single photo-electron peak distribution.",
        allow_none=False,
    ).tag(config=True)
    crosstalk = FloatTelescopeParameter(
        default_value=0, help="Average pixel crosstalk.", allow_none=False
    ).tag(config=True)
    sigma_space = Float(
        4,
        help="Size of the region on which the fit is performed relative to the image extension.",
        allow_none=False,
    ).tag(config=True)
    sigma_time = Float(
        3,
        help="Time window on which the fit is performed relative to the image temporal extension.",
        allow_none=False,
    ).tag(config=True)
    time_before_shower = FloatTelescopeParameter(
        default_value=10,
        help="Additional time at the start of the fit temporal window.",
        allow_none=False,
    ).tag(config=True)
    time_after_shower = FloatTelescopeParameter(
        default_value=20,
        help="Additional time at the end of the fit temporal window.",
        allow_none=False,
    ).tag(config=True)
    use_weight = Bool(
        False,
        help="If True, the brightest sample is twice as important as the dimmest pixel in the "
        "likelihood. If false all samples are equivalent.",
        allow_none=False,
    ).tag(config=True)
    no_asymmetry = Bool(
        False,
        help="If true, the asymmetry of the spatial model is fixed to 0.",
        allow_none=False,
    ).tag(config=True)
    use_interleaved = Path(
        None,
        help="Location of the dl1 file used to estimate the pedestal exploiting interleaved"
        " events.",
        allow_none=True,
    ).tag(config=True)
    n_peaks = Int(
        0,
        help="Maximum brightness (p.e.) for which the full likelihood computation is used. "
        "If the Poisson term for Np.e.>n_peak is more than 1e-6 a Gaussian approximation is used.",
        allow_none=False,
    ).tag(config=True)
    bound_charge_factor = FloatTelescopeParameter(
        default_value=4,
        help="Maximum relative change to the fitted charge parameter.",
        allow_none=False,
    ).tag(config=True)
    bound_t_cm_value = FloatTelescopeParameter(
        default_value=10, help="Maximum change to the t_cm parameter.", allow_none=False
    ).tag(config=True)
    bound_centroid_control_parameter = FloatTelescopeParameter(
        default_value=1,
        help="Maximum change of the centroid coordinated in " "number of seed length",
        allow_none=False,
    ).tag(config=True)
    bound_max_length_factor = FloatTelescopeParameter(
        default_value=2,
        help="Maximum relative increase to the fitted length parameter.",
        allow_none=False,
    ).tag(config=True)
    bound_length_asymmetry = FloatTelescopeParameter(
        default_value=9, help="Bounds for the fitted rl parameter.", allow_none=False
    ).tag(config=True)
    bound_max_v_cm_factor = FloatTelescopeParameter(
        default_value=2,
        help="Maximum relative increase to the fitted v_cm parameter.",
        allow_none=False,
    ).tag(config=True)
    default_seed_t_cm = FloatTelescopeParameter(
        default_value=0,
        help="Default starting value of t_cm when the seed extraction failed.",
        allow_none=False,
    ).tag(config=True)
    default_seed_v_cm = FloatTelescopeParameter(
        default_value=40,
        help="Default starting value of v_cm when the seed extraction failed.",
        allow_none=False,
    ).tag(config=True)
    verbose = Int(
        0,
        help="4 - used for tests: create debug plots\n"
        "3 - create debug plots, wait for input after each event, increase minuit verbose level\n"
        "2 - create debug plots, increase minuit verbose level\n"
        "1 - increase minuit verbose level\n"
        "0 - silent",
        allow_none=False,
    ).tag(config=True)

    def __init__(self, writer, subarray, config=None, parent=None, **kwargs):
        super().__init__(subarray=subarray, config=config, parent=parent, **kwargs)

        self.data_writer = writer

    def __call__(self, event: ArrayEventContainer):
        # self.data_writer.writer.write(table_name, lhfit_parameters_container)
        pass
