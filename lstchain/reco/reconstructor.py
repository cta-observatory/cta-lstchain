import logging

from iminuit import Minuit
import numpy as np
import astropy.units as u

from ctapipe.containers import EventType
from ctapipe.core import TelescopeComponent
from ctapipe.core.traits import Bool, Float, FloatTelescopeParameter, Int, Unicode

from lstchain.data.normalised_pulse_template import NormalizedPulseTemplate
from lstchain.io.lstcontainers import DL1LikelihoodParametersContainer
from lstchain.reco.reconstructorCC import log_pdf as log_pdf

logger = logging.getLogger(__name__)


class TimeWaveformFitter(TelescopeComponent):
    """
    Class used to perform event reconstruction by fitting of a model on waveforms.
    """
    sigma_s = FloatTelescopeParameter(default_value=1, help='Width of the single photo-electron peak distribution.',
                                      allow_none=False).tag(config=True)
    crosstalk = FloatTelescopeParameter(default_value=0, help='Average pixel crosstalk.',
                                        allow_none=False).tag(config=True)
    spatial_selection = Unicode(default_value='dvr',
                                help='Method to select pixels to perform the likelihood fit. Can be:\n'
                                     'hillas : use the hillas length and width times sigma_space as an ellipsis.\n'
                                     'dvr : use data volume reduction logic with dvr_pic_threshold and'
                                     ' dvr_pix_for_full_image',
                                allow_none=False).tag(config=True)
    dvr_pic_threshold = Int(default_value=8, help='Pixel charge threshold for dvr like pixel selection.',
                            allow_none=False).tag(config=True)
    dvr_pix_for_full_image = Int(default_value=500, help='Number of selected pixels above which all are kept.',
                                 allow_none=False).tag(config=True)
    sigma_space = Float(default_value=4,
                        help='Size of the region on which the fit is performed relative to the image extension.',
                        allow_none=False).tag(config=True)
    sigma_time = Float(default_value=3,
                       help='Time window on which the fit is performed relative to the image temporal extension.',
                       allow_none=False).tag(config=True)
    time_before_shower = FloatTelescopeParameter(default_value=10,
                                                 help='Additional time at the start of the fit temporal window.',
                                                 allow_none=False).tag(config=True)
    time_after_shower = FloatTelescopeParameter(default_value=20,
                                                help='Additional time at the end of the fit temporal window.',
                                                allow_none=False).tag(config=True)
    no_asymmetry = Bool(default_value=False, help='If true, the asymmetry of the spatial model is fixed to 0.',
                        allow_none=False).tag(config=True)
    use_interleaved = Bool(default_value=None, help='If true, the std deviation of pedestals and dimmed pixels are '
                                                    'estimated on interleaved events', allow_none=True).tag(config=True)
    n_peaks = Int(default_value=0,
                  help='Maximum brightness (p.e.) for which the full likelihood computation is used. '
                       'If the Poisson term for Np.e.>n_peak is more than 1e-6 a Gaussian approximation is used.',
                  allow_none=False).tag(config=True)
    bound_charge_factor = FloatTelescopeParameter(default_value=4,
                                                  help='Maximum relative change to the fitted charge parameter.',
                                                  allow_none=False).tag(config=True)
    bound_t_cm_value = FloatTelescopeParameter(default_value=10,
                                               help='Maximum change to the t_cm parameter.',
                                               allow_none=False).tag(config=True)
    bound_centroid_control_parameter = FloatTelescopeParameter(default_value=1,
                                                               help='Maximum change of the centroid coordinated in '
                                                                    'number of seed length',
                                                               allow_none=False).tag(config=True)
    bound_max_length_factor = FloatTelescopeParameter(default_value=2,
                                                      help='Maximum relative increase to the fitted length parameter.',
                                                      allow_none=False).tag(config=True)
    bound_length_asymmetry = FloatTelescopeParameter(default_value=9,
                                                     help='Bounds for the fitted rl parameter.',
                                                     allow_none=False).tag(config=True)
    bound_max_v_cm_factor = FloatTelescopeParameter(default_value=2,
                                                    help='Maximum relative increase to the fitted v_cm parameter.',
                                                    allow_none=False).tag(config=True)
    default_seed_t_cm = FloatTelescopeParameter(default_value=0,
                                                help='Default starting value of t_cm when the seed extraction failed.',
                                                allow_none=False).tag(config=True)
    default_seed_v_cm = FloatTelescopeParameter(default_value=40,
                                                help='Default starting value of v_cm when the seed extraction failed.',
                                                allow_none=False).tag(config=True)
    verbose = Int(default_value=0,
                  help='4 - used for tests: create debug plots\n'
                       '3 - create debug plots, wait for input after each event, increase minuit verbose level\n'
                       '2 - create debug plots, increase minuit verbose level\n'
                       '1 - increase minuit verbose level\n'
                       '0 - silent', allow_none=False).tag(config=True)

    def __init__(self, subarray, config=None, parent=None, **kwargs):
        super().__init__(subarray=subarray, config=config, parent=parent, **kwargs)
        self.subarray = subarray
        self.template_dict = {}
        self.template_time_of_max_dict = {}
        for tel_id in subarray.tel:
            self.template_dict[tel_id] = NormalizedPulseTemplate.load_from_eventsource(
                subarray.tel[tel_id].camera.readout)
            self.template_time_of_max_dict[tel_id] = self.template_dict[tel_id].compute_time_of_max()
        poisson_peaks = np.arange(self.n_peaks + 1, dtype=int)
        poisson_peaks[0] = 1
        self.factorial = np.cumprod(poisson_peaks, dtype='u8')
        # Find the transition charge between full likelihood computation and Gaussian approximation
        # The maximum charge is selected such that each Poisson terms in the full likelihood computation
        # above the n_peaks limit account for less than (1/n_peaks)%
        transition_charges = {}
        for config_crosstalk in self.crosstalk:
            # if n_peaks is set to 0, only the Gaussian approximation is used
            transition_charges[config_crosstalk[2]] = 0.0 if self.n_peaks == 0 \
                else self.find_transition_charge(config_crosstalk[2], 1e-2 / self.n_peaks)
        self.transition_charges = {}
        for tel_id in subarray.tel:
            self.transition_charges[tel_id] = transition_charges[self.crosstalk.tel[tel_id]]
        self.error = None
        self.allowed_pixels = True

        self.start_parameters = None
        self.names_parameters = None
        self.end_parameters = None
        self.error_parameters = None
        self.bound_parameters = None
        self.fcn = None

    def get_ped_from_interleaved(self, source):
        """
        Use interleaved events to extract the pixel-wise pedestal variance from data.
        """
        self.error = {}
        waveforms = {}
        for tel_id in self.subarray.tel:
            waveforms[tel_id] = []
        for i, event in enumerate(source):
            if event.trigger.event_type == EventType.SKY_PEDESTAL:
                source.r0_r1_calibrator.calibrate(event)
                for tel_id in event.r1.tel.keys():
                    waveforms[tel_id].append(event.r1.tel[tel_id].waveform.squeeze())
        for tel_id, tel_waveforms in waveforms.items():
            x = np.concatenate(np.asarray(tel_waveforms), axis=1)
            std = []
            for elt in x:
                std.append(np.nanstd(elt))
            self.error[tel_id] = std
            self.allowed_pixels = (std > 0.5 * np.median(std))

    def get_ped_from_true_signal_less(self, source, nsb_tunner):
        """
        Use pixels with no true signal in MC to extract the pedestal variance, assumed uniform in the camera.
        """
        self.error = {}
        waveforms = {}
        n_pix = {}
        for tel_id in self.subarray.tel:
            waveforms[tel_id] = []
        for i, event in enumerate(source):
            if i > 50 and len(waveforms) > 1000:
                break
            if event.trigger.event_type == EventType.SUBARRAY:
                for tel_id in event.r1.tel.keys():
                    if i == 0:
                        n_pix[tel_id] = event.r1.tel[tel_id].waveform.shape[1]
                    mask = event.simulation.tel[tel_id].true_image == 0
                    wave = event.r1.tel[tel_id].waveform
                    if nsb_tunner is not None:
                        selected_gains = event.r1.tel[tel_id].selected_gain_channel
                        mask_high = (selected_gains == 0)
                        nsb_tunner.tune_nsb_on_waveform(wave, tel_id, mask_high, self.subarray)
                    waveforms[tel_id].append(wave[0, mask].flatten())
        for tel_id, tel_waveforms in waveforms.items():
            self.error[tel_id] = np.full(n_pix[tel_id], np.nanstd(np.concatenate(tel_waveforms)))
            self.allowed_pixels = np.ones(n_pix[tel_id], dtype=bool)

    def call_setup(self, event, telescope_id, dl1_container):
        """
        Extract all event dependent quantities used for the fit.

        Parameters
        ----------
        event: ctapipe event container
            Current event container.
        telescope_id: int
            Id of the telescope
        dl1_container: DL1ParametersContainer
            Contains the Hillas parameters used as seed for the fit

        Returns
        -------
        focal_length: astropy.units.Quantity
            Focal length of the telescope
        fit_params: array
            Array containing all the variable needed to compute the likelihood
            during the fir excluding the model parameters
        """

        geometry = self.subarray.tel[telescope_id].camera.geometry
        unit = geometry.pix_x.unit
        pix_x = geometry.pix_x.to_value(unit)
        pix_y = geometry.pix_y.to_value(unit)
        r_max = geometry.guess_radius().to_value(unit)
        pix_radius = np.sqrt(geometry.pix_area[0].to_value(unit ** 2) / np.pi)  # find linear size of a pixel
        readout = self.subarray.tel[telescope_id].camera.readout
        sampling_rate = readout.sampling_rate.to_value(u.GHz)
        dt = (1.0 / sampling_rate)
        template = self.template_dict[telescope_id]
        image = event.dl1.tel[telescope_id].image
        hillas_signal_pixels = event.dl1.tel[telescope_id].image_mask
        start_x_cm, start_y_cm = init_centroid(dl1_container,
                                               geometry[hillas_signal_pixels],
                                               unit,
                                               image[hillas_signal_pixels],
                                               self.no_asymmetry
                                               )

        waveform = event.r1.tel[telescope_id].waveform

        dl1_calib = event.calibration.tel[telescope_id].dl1
        # TODO check if this is correct here or if it is applied to r1 waveform earlier
        if dl1_calib.pedestal_offset is not None:
            waveform = waveform - dl1_calib.pedestal_offset[:, np.newaxis]

        n_samples = waveform.shape[2]
        times = np.arange(0, n_samples) * dt
        selected_gains = event.r1.tel[telescope_id].selected_gain_channel
        time_shift = dl1_calib.time_shift
        
        is_high_gain = (selected_gains == 0)

        # We assume that the time gradient is given in unit of 'geometry spatial unit'/ns
        v = dl1_container.time_gradient
        psi = dl1_container.psi.to_value(u.rad)
        # We use only positive time gradients and psi is projected in [-pi,pi] from [-pi/2,pi/2]
        if v < 0:
            if psi >= 0:
                psi = psi - np.pi
            else:
                psi = psi + np.pi

        start_length = max(dl1_container.length.to_value(unit), pix_radius)
        # With current likelihood computation, order and type of the parameters are important
        start_parameters = {'charge': dl1_container.intensity,
                            't_cm': dl1_container.intercept
                                    - self.template_time_of_max_dict[telescope_id],
                            'x_cm': start_x_cm.to_value(unit),
                            'y_cm': start_y_cm.to_value(unit),
                            'length': start_length,
                            'wl': max(dl1_container.wl, 0.01),
                            'psi': psi,
                            'v': np.abs(v),
                            'rl': 0.0
                            }

        # Temporal parameters extraction fails when cleaning select only 2 pixels, we use defaults values in this case
        if np.isnan(start_parameters['t_cm']):
            start_parameters['t_cm'] = self.default_seed_t_cm.tel[telescope_id]
        if np.isnan(start_parameters['v']):
            start_parameters['v'] = self.default_seed_v_cm.tel[telescope_id]

        t_max = n_samples * dt
        v_min, v_max = 0, max(self.bound_max_v_cm_factor.tel[telescope_id] * start_parameters['v'], 50)
        rl_min, rl_max = -self.bound_length_asymmetry.tel[telescope_id], self.bound_length_asymmetry.tel[telescope_id]
        if self.no_asymmetry:
            rl_min, rl_max = 0.0, 0.0
        bound_centroid = self.bound_centroid_control_parameter.tel[telescope_id] * start_length

        bound_parameters = {'charge': (dl1_container.intensity / self.bound_charge_factor.tel[telescope_id],
                                       dl1_container.intensity * self.bound_charge_factor.tel[telescope_id]),
                            't_cm': (-self.bound_t_cm_value.tel[telescope_id],
                                     t_max + self.bound_t_cm_value.tel[telescope_id]),
                            'x_cm': (start_x_cm.to_value(unit) - bound_centroid,
                                     start_x_cm.to_value(unit) + bound_centroid),
                            'y_cm': (start_y_cm.to_value(unit) - bound_centroid,
                                     start_y_cm.to_value(unit) + bound_centroid),
                            'length': (pix_radius,
                                       min(self.bound_max_length_factor.tel[telescope_id] * start_length, r_max)),
                            'wl': (0.001, 1.0),
                            'psi': (-np.pi * 2.0, np.pi * 2.0),
                            'v': (v_min, v_max),
                            'rl': (rl_min, rl_max)
                            }

        mask_pixel, mask_time = self.clean_data(image, geometry, pix_x, pix_y, pix_radius, times, start_parameters,
                                                telescope_id)
        mask_pixel = mask_pixel & self.allowed_pixels
        spatial_ones = np.ones(np.sum(mask_pixel))

        is_high_gain = is_high_gain[mask_pixel]
        sig_s = spatial_ones * self.sigma_s.tel[telescope_id]
        crosstalks = spatial_ones * self.crosstalk.tel[telescope_id]

        times = (np.arange(0, n_samples) * dt)[mask_time]

        time_shift = np.choose(selected_gains, time_shift)
        time_shift = time_shift[mask_pixel]

        p_x = pix_x[mask_pixel]
        p_y = pix_y[mask_pixel]
        pix_area = geometry.pix_area[mask_pixel].to_value(unit ** 2)

        data = waveform[0]
        error = self.error

        filter_pixels = np.nonzero(~mask_pixel)
        filter_times = np.nonzero(~mask_time)

        if error is None:
            std = np.std(data[~mask_pixel])
            error = np.full(data.shape[0], std)
        else:
            error = self.error[telescope_id]

        data = np.delete(data, filter_pixels, axis=0)
        data = np.delete(data, filter_times, axis=1)
        error = np.delete(error, filter_pixels, axis=0)

        # Fill the set of non-fitted parameters needed to compute the likelihood. Order and type sensitive.
        fit_params = [data, error, is_high_gain,
                      sig_s, crosstalks, times,
                      np.float32(time_shift), p_x, p_y,
                      np.float64(pix_area), template.dt,
                      template.t0, template.amplitude_LG,
                      template.amplitude_HG, self.n_peaks,
                      self.transition_charges[telescope_id],
                      self.factorial]

        self.start_parameters = start_parameters
        self.names_parameters = start_parameters.keys()
        self.bound_parameters = bound_parameters

        return unit, fit_params

    def __call__(self, event, telescope_id, dl1_container):
        # setup angle to distance conversion on the camera plane for the current telescope
        focal_length = self.subarray.tel[telescope_id].optics.equivalent_focal_length
        angle_dist_eq = [(u.rad, u.m, lambda x: np.tan(x) * focal_length.to_value(u.m),
                          lambda x: np.arctan(x / focal_length.to_value(u.m))),
                         (u.rad ** 2, u.m ** 2, lambda x: (np.tan(np.sqrt(x)) * focal_length.to_value(u.m)) ** 2,
                          lambda x: (np.arctan(np.sqrt(x) / focal_length.to_value(u.m))) ** 2)]
        with u.set_enabled_equivalencies(angle_dist_eq):
            self.start_parameters = None
            self.names_parameters = None
            unit_cam, fit_params = self.call_setup(event, telescope_id, dl1_container)
            self.end_parameters = None
            self.error_parameters = None
            self.fcn = None

            return self.predict(unit_cam, fit_params)

    def clean_data(self, image, geometry, pix_x, pix_y, pix_radius, times, start_parameters, telescope_id):
        """
        Method used to select pixels and time samples used in the fitting procedure.
        The spatial selection can be done in one of two methods:
        - takes pixels in an ellipsis obtained from the seed Hillas parameters extended by one pixel size and multiplied
        by a factor sigma_space.
        - use all pixels kept by the data volume reduction algorythm
        The temporal selection takes a time window centered on the seed time of center of mass and of duration equal to
        the time of propagation of the signal along the length of the ellipsis times a factor sigma_time.
        An additional fixed duration is also added before and after this time window through the time_before_shower and
        time_after_shower arguments.

        Parameters
        ----------
        image : array_like
            Charge in each pixel
        geometry: `ctapipe.instrument.CameraGeometry`
            Camera geometry
        pix_x, pix_y: array-like
            Pixels positions
        pix_radius: float
        times: array-like
            Sampling times before timeshift corrections
        start_parameters: dict
            Seed parameters derived from the Hillas parameters
        telescope_id: int

        Returns
        ----------
        mask_pixel, mask_time: array-like
            Mask used to select pixels and times for the fit

        """
        length = start_parameters['length']
        if self.spatial_selection == 'hillas':
            x_cm = start_parameters['x_cm']
            y_cm = start_parameters['y_cm']
            width = start_parameters['wl'] * length
            psi = start_parameters['psi']

            dx = pix_x - x_cm
            dy = pix_y - y_cm

            lon = dx * np.cos(psi) + dy * np.sin(psi)
            lat = dx * np.sin(psi) - dy * np.cos(psi)

            mask_pixel = ((lon / (length + pix_radius)) ** 2 + (
                    lat / (width + pix_radius)) ** 2) < self.sigma_space ** 2
        elif self.spatial_selection == 'dvr':
            mask_pixel = (image > self.dvr_pic_threshold)
            # we add-up (sum) the selected-pixel-wise map of neighbors, to find
            # those who appear at least once (>0). Those should be added:
            additional_pixels = (np.sum(geometry.neighbor_matrix[mask_pixel], axis=0) > 0)
            mask_pixel |= additional_pixels
            # if more than min_npixels_for_full_event were selected, keep whole camera:
            if mask_pixel.sum() > self.dvr_pix_for_full_image:
                mask_pixel = np.array(geometry.n_pixels * [True])

        v = start_parameters['v']
        t_start = (start_parameters['t_cm']
                   - (np.abs(v) * length / 2 * self.sigma_time)
                   - self.time_before_shower.tel[telescope_id])
        t_end = (start_parameters['t_cm']
                 + (np.abs(v) * length / 2 * self.sigma_time)
                 + self.time_after_shower.tel[telescope_id])

        mask_time = (times < t_end) * (times > t_start)

        return mask_pixel, mask_time

    def find_transition_charge(self, crosstalk, poisson_proba_min=1e-2):
        """
        Find the charge below which the full likelihood computation is performed and above which a Gaussian
        approximation is used. For a given pixel crosstalk it finds the maximum charge with a Generalised Poisson term
        below poisson_proba_min for n_peaks photo-electrons. n_peaks here is the configured maximum number of
        photo-electron considered in the full likelihood computation.

        Parameters
        ----------
        crosstalk : float
            Pixels crosstalk
        poisson_proba_min: float

        Returns
        -------
        transition_charge: float32
            Model charge of transition between full and approximated likelihood

        """
        transition_charge = self.n_peaks / (1 + crosstalk)
        step = transition_charge / 100

        def poisson(mu, cross_talk):
            return (mu * pow(mu + self.n_peaks * cross_talk, (self.n_peaks - 1)) / self.factorial[self.n_peaks] *
                    np.exp(-mu - self.n_peaks * cross_talk))

        while poisson(transition_charge, crosstalk) > poisson_proba_min:
            transition_charge -= step
        logger.info(f'Transition charge between full and approximated likelihood for camera '
                    f'with crosstalk = {crosstalk:.4f} is,  {transition_charge:.4f}, p.e.')
        return np.float32(transition_charge)

    def fit(self, fit_params):
        """
        Performs the fitting procedure.

        Parameters
        ----------
        fit_params: array
            Parameters used to compute the likelihood but not fitted

        """

        def f(*args):
            return -2 * self.log_likelihood(*args, fit_params=fit_params)

        print_level = 2 if self.verbose in [1, 2, 3] else 0
        m = Minuit(f,
                   name=self.names_parameters,
                   *self.start_parameters.values())
        for key, val in self.bound_parameters.items():
            m.limits[key] = val
        m.print_level = print_level
        m.errordef = 0.5
        m.simplex().migrad()
        self.end_parameters = m.values.to_dict()
        self.fcn = m.fval
        self.error_parameters = m.errors.to_dict()

    def predict(self, unit_cam, fit_params):
        """
            Call the fitting procedure and fill the results.

        Parameters
        ----------
        unit_cam: astropy.units.unit
            Unit used for the camera geometry and for spatial variable in the fit
        fit_params: array
            Parameters used to compute the likelihood but not fitted

        Returns
        ----------
        container: DL1LikelihoodParametersContainer
            Filled parameter container

        """
        container = DL1LikelihoodParametersContainer(lhfit_call_status=1)
        try:
            self.fit(fit_params)
            container.lhfit_TS = self.fcn

            container.lhfit_x = (self.end_parameters['x_cm'] * unit_cam).to(u.m)
            container.lhfit_x_uncertainty = (self.error_parameters['x_cm'] * unit_cam).to(u.m)
            container.lhfit_y = (self.end_parameters['y_cm'] * unit_cam).to(u.m)
            container.lhfit_y_uncertainty = (self.error_parameters['y_cm'] * unit_cam).to(u.m)
            container.lhfit_r = np.sqrt(container.lhfit_x ** 2 + container.lhfit_y ** 2)
            container.lhfit_phi = np.arctan2(container.lhfit_y, container.lhfit_x)
            if self.end_parameters['psi'] > np.pi:
                self.end_parameters['psi'] -= 2 * np.pi
            if self.end_parameters['psi'] < -np.pi:
                self.end_parameters['psi'] += 2 * np.pi
            container.lhfit_psi = self.end_parameters['psi'] * u.rad
            container.lhfit_psi_uncertainty = self.error_parameters['psi'] * u.rad
            length_asy = 1 + self.end_parameters['rl'] if self.end_parameters['rl'] >= 0 else 1 / (
                    1 - self.end_parameters['rl'])
            lhfit_length = (((1.0 + length_asy)
                             * self.end_parameters['length'] / 2.0) * unit_cam).to(u.deg)
            container.lhfit_length = lhfit_length
            lhfit_length_rel_err = self.error_parameters['length'] / self.end_parameters['length']
            # We assume that the relative error is the same in the fitted and saved unit
            container.lhfit_length_uncertainty = lhfit_length_rel_err * container.lhfit_length
            container.lhfit_width = self.end_parameters['wl'] * container.lhfit_length

            container.lhfit_time_gradient = self.end_parameters['v']
            container.lhfit_time_gradient_uncertainty = self.error_parameters['v']
            container.lhfit_ref_time = self.end_parameters['t_cm']
            container.lhfit_ref_time_uncertainty = self.error_parameters['t_cm']

            container.lhfit_wl = u.Quantity(self.end_parameters['wl'])
            container.lhfit_wl_uncertainty = u.Quantity(self.error_parameters['wl'])
            container.lhfit_intensity = self.end_parameters['charge']
            container.lhfit_intensity_uncertainty = self.error_parameters['charge']
            container.lhfit_log_intensity = np.log10(container.lhfit_intensity)
            container.lhfit_t_68 = container.lhfit_length.value * container.lhfit_time_gradient
            container.lhfit_area = container.lhfit_length * container.lhfit_width
            container.lhfit_length_asymmetry = self.end_parameters['rl']
            container.lhfit_length_asymmetry_uncertainty = self.error_parameters['rl']
        except ZeroDivisionError:
            # TODO Check occurrence rate and solve
            container = DL1LikelihoodParametersContainer(lhfit_call_status=-1)
            logger.error('ZeroDivisionError encounter during the fitting procedure, skipping event.')

        return container

    def __str__(self):
        """
            Define the print format of TimeWaveformFitter objects.

            Returns
            -------
            str: string
                Contains the starting and bound parameters used for the fit,
                and the end results with errors and associated log-likelihood
                in readable format.

        """
        s = 'Event processed\n'
        s += 'Start parameters :\n\t{}\n'.format(self.start_parameters)
        s += 'Bound parameters :\n\t{}\n'.format(self.bound_parameters)
        s += 'End parameters :\n\t{}\n'.format(self.end_parameters)
        s += 'Error parameters :\n\t{}\n'.format(self.error_parameters)
        s += '-2Log-Likelihood :\t{}'.format(self.fcn)

        return s

    @staticmethod
    def log_likelihood(*args, fit_params, **kwargs):
        """Compute the log-likelihood used in the fitting procedure."""
        llh = log_pdf(*args, *fit_params, **kwargs)
        return np.sum(llh)


def init_centroid(dl1_container, geom, unit, image, no_asymmetry):
    """
    Initialise the centroid position for the likelihood fit to the Hillas
    parameter centroid or using a centroid weighting as the signal squared
    for asymmetric models

    Parameters
    ----------
    dl1_container: DL1ParametersContainer
        Hillas parameter container
    geom: `ctapipe.instrument.CameraGeometry`
        Camera geometry
    unit: astropy.unit
        Unit used to describe spatial quantities
    image : array_like
        Charge in each pixel
    no_asymmetry: bool
        Information on the used spatial model

    """
    if no_asymmetry:
        return dl1_container.x.to(unit), dl1_container.y.to(unit)
    else:
        return asy_centroid(geom, unit, image)


def asy_centroid(geom, unit, image):
    """
    Compute a centroid weighting as the signal squared for asymmetric models

    Parameters
    ----------
    geom: `ctapipe.instrument.CameraGeometry`
        Camera geometry
    unit: astropy.unit
        Unit used to describe spatial quantities
    image : array_like
        Charge in each pixel

    Returns
    -------
    (x_cm, y_cm): `astropy.units.Quantity`

    """
    pix_x = geom.pix_x.to_value(unit)
    pix_y = geom.pix_y.to_value(unit)
    image = np.asanyarray(image, dtype=np.float64)
    image = np.ma.filled(image, 0)
    cog_x = np.average(pix_x, weights=image ** 2)
    cog_y = np.average(pix_y, weights=image ** 2)
    return [u.Quantity(cog_x, unit), u.Quantity(cog_y, unit)]
