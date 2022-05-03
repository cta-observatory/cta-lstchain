import logging

from iminuit import Minuit
import numpy as np
import astropy.units as u

from lstchain.data.normalised_pulse_template import NormalizedPulseTemplate

from ctapipe.core import TelescopeComponent
from ctapipe.core.traits import Bool, Float, FloatTelescopeParameter, Int, Path
from lstchain.io.lstcontainers import DL1LikelihoodParametersContainer

logger = logging.getLogger(__name__)

try:
    from lstchain.reco.log_pdf_CC import log_pdf as log_pdf
except ImportError:
    pass


class TimeWaveformFitter(TelescopeComponent):
    sigma_s = FloatTelescopeParameter(default_value=1, help='Width of the single photo-electron peak distribution.',
                                      allow_none=False).tag(config=True)
    crosstalk = FloatTelescopeParameter(default_value=0, help='Average pixel crosstalk.',
                                        allow_none=False).tag(config=True)
    sigma_space = Float(4, help='Size of the region on which the fit is performed relative to the image extension.',
                        allow_none=False).tag(config=True)
    sigma_time = Float(3, help='Time window on which the fit is performed relative to the image temporal extension.',
                       allow_none=False).tag(config=True)
    time_before_shower = FloatTelescopeParameter(default_value=10,
                                                 help='Additional time at the start of the fit temporal window.',
                                                 allow_none=False).tag(config=True)
    time_after_shower = FloatTelescopeParameter(default_value=20,
                                                help='Additional time at the end of the fit temporal window.',
                                                allow_none=False).tag(config=True)
    use_weight = Bool(False, help='If True, the brightest sample is twice as important as the dimmest pixel in the '
                                  'likelihood. If false all samples are equivalent.', allow_none=False).tag(config=True)
    no_asymmetry = Bool(False, help='If true, the asymmetry of the spatial model is fixed to 0.',
                        allow_none=False).tag(config=True)
    use_interleaved = Path(None, help='Location of the dl1 file used to estimate the pedestal exploiting interleaved'
                                      ' events.', allow_none=True).tag(config=True)
    n_peaks = Int(50, help='Maximum brightness (p.e.) for which the full likelihood computation is used. '
                           'If the Poisson term for Np.e.>n_peak is more than 1e-6 a Gaussian approximation is used.',
                  allow_none=False).tag(config=True)
    verbose = Int(0, help='4 - used for tests: create debug plots\n'
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
        poisson_peaks = np.arange(self.n_peaks, dtype=int)
        poisson_peaks[0] = 1
        self.factorial = np.cumprod(poisson_peaks, dtype='u8')
        self.start_parameters = None
        self.names_parameters = None
        self.end_parameters = None
        self.error_parameters = None
        self.bound_parameters = None
        self.fcn = None

    def call_setup(self, event, telescope_id, dl1_container):

        geometry = self.subarray.tel[telescope_id].camera.geometry
        pix_x = geometry.pix_x.to_value(u.m)
        pix_y = geometry.pix_y.to_value(u.m)
        r_max = geometry.guess_radius().to_value(u.m)
        pix_radius = np.sqrt(geometry.pix_area[0].to_value(u.m ** 2)/np.pi)  # find linear size of a pixel
        focal_length = self.subarray.tel[telescope_id].optics.equivalent_focal_length
        readout = self.subarray.tel[telescope_id].camera.readout
        sampling_rate = readout.sampling_rate.to_value(u.GHz)
        dt = (1.0 / sampling_rate)
        template = self.template_dict[telescope_id]
        image = event.dl1.tel[telescope_id].image
        hillas_signal_pixels = event.dl1.tel[telescope_id].image_mask
        start_x_cm, start_y_cm = init_centroid(dl1_container,
                                               geometry[hillas_signal_pixels],
                                               image[hillas_signal_pixels],
                                               self.no_asymmetry
                                               )

        waveform = event.r1.tel[telescope_id].waveform

        dl1_calib = event.calibration.tel[telescope_id].dl1
        time_shift = dl1_calib.time_shift
        # TODO check if this is correct here or if it is applied to r1 waveform earlier
        if dl1_calib.pedestal_offset is not None:
            waveform = waveform - dl1_calib.pedestal_offset[:, np.newaxis]

        n_pixels, n_samples = waveform.shape
        times = np.arange(0, n_samples) * dt
        selected_gains = event.r1.tel[telescope_id].selected_gain_channel
        is_high_gain = (selected_gains == 0)

        v = dl1_container.time_gradient
        psi = dl1_container.psi.to_value(u.rad)
        # We use only positive time gradients and psi is projected in [-pi,pi] from [-pi/2,pi/2]
        if v < 0:
            if psi >= 0:
                psi = psi - np.pi
            else:
                psi = psi + np.pi

        start_length = max(np.tan(dl1_container.length.to_value(u.rad)) * focal_length.to_value(u.m), 0.02)
        # With current likelihood computation, order and type of the parameters are important
        start_parameters = {'charge': dl1_container.intensity,
                            't_cm': dl1_container.intercept
                            - self.template_time_of_max_dict[telescope_id],
                            'x_cm': start_x_cm.to_value(u.m),
                            'y_cm': start_y_cm.to_value(u.m),
                            'length': start_length,
                            'wl': max(dl1_container.wl, 0.01),
                            'psi': psi,
                            'v': np.abs(v),
                            'rl': 0.0
                            }

        if np.isnan(start_parameters['t_cm']):
            start_parameters['t_cm'] = 0.
        if np.isnan(start_parameters['v']):
            start_parameters['v'] = 40

        t_max = n_samples * dt
        v_min, v_max = 0, max(2 * start_parameters['v'], 50)
        rl_min, rl_max = -9, 9
        if self.no_asymmetry:
            rl_min, rl_max = 0.0, 0.0

        bound_parameters = {'charge': (dl1_container.intensity * 0.25,
                                       dl1_container.intensity * 4.0),
                            't_cm': (-10, t_max + 10),
                            'x_cm': (start_x_cm.to_value(u.m)
                                     - start_length,
                                     start_x_cm.to_value(u.m)
                                     + start_length),
                            'y_cm': (start_y_cm.to_value(u.m)
                                     - start_length,
                                     start_y_cm.to_value(u.m)
                                     + start_length),
                            'length': (pix_radius,
                                       min(2 * start_length, r_max)),
                            'wl': (0.001, 1.0),
                            'psi': (-np.pi * 2.0, np.pi * 2.0),
                            'v': (v_min, v_max),
                            'rl': (rl_min, rl_max)
                            }

        mask_pixel, mask_time = self.clean_data(pix_x, pix_y, pix_radius, times, start_parameters, telescope_id)
        spatial_ones = np.ones(np.sum(mask_pixel))

        is_high_gain = is_high_gain[mask_pixel]
        sig_s = spatial_ones * self.sigma_s.tel[telescope_id]
        crosstalks = spatial_ones * self.crosstalk.tel[telescope_id]

        times = (np.arange(0, n_samples) * dt)[mask_time]
        time_shift = time_shift[mask_pixel]

        p_x = pix_x[mask_pixel]
        p_y = pix_y[mask_pixel]
        pix_area = geometry.pix_area[mask_pixel].to_value(u.m ** 2)

        data = waveform
        error = None  # TODO include option to use calibration data

        filter_pixels = np.nonzero(~mask_pixel)
        filter_times = np.nonzero(~mask_time)

        if error is None:
            std = np.std(data[~mask_pixel])
            error = np.full(data.shape[0], std)

        data = np.delete(data, filter_pixels, axis=0)
        data = np.delete(data, filter_times, axis=1)
        error = np.delete(error, filter_pixels, axis=0)

        # Fill the set of non-fitted parameters needed to compute the likelihood. Order and type sensitive.
        fit_params = [data, error, is_high_gain,
                      sig_s, crosstalks, times,
                      time_shift, p_x, p_y,
                      pix_area, template.dt,
                      template.t0, template.amplitude_LG,
                      template.amplitude_HG, self.n_peaks,
                      self.use_weight, self.factorial]

        self.start_parameters = start_parameters
        self.names_parameters = start_parameters.keys()
        self.bound_parameters = bound_parameters

        return focal_length, fit_params

    def __call__(self, event, telescope_id, dl1_container):
        self.start_parameters = None
        self.names_parameters = None
        focal_length, fit_params = self.call_setup(event, telescope_id, dl1_container)
        self.end_parameters = None
        self.error_parameters = None
        self.fcn = None

        return self.predict(focal_length, fit_params)

    def clean_data(self, pix_x, pix_y, pix_radius, times, start_parameters, telescope_id):
        """
            Method used to select pixels and time samples used in the
            fitting procedure. The spatial selection takes pixels in an
            ellipsis obtained from the seed Hillas parameters extraction.
            An additional factor sigma_space is added to the length and width
            to extend the selected region. The temporal selection takes a time
            window centered on the seed time of center of mass and of duration
            equal to the time of propagation of the signal along the length
            of the ellipsis times a factor sigma_time. An additional fixed
            duration is also added before and after this time window through
            the time_before_shower and time_after_shower arguments.

        """
        x_cm = start_parameters['x_cm']
        y_cm = start_parameters['y_cm']
        length = start_parameters['length']
        width = start_parameters['wl'] * length
        psi = start_parameters['psi']

        dx = pix_x - x_cm
        dy = pix_y - y_cm

        lon = dx * np.cos(psi) + dy * np.sin(psi)
        lat = dx * np.sin(psi) - dy * np.cos(psi)

        mask_pixel = ((lon / (length + pix_radius)) ** 2 + (
                lat / (width + pix_radius)) ** 2) < self.sigma_space ** 2

        v = start_parameters['v']
        t_start = (start_parameters['t_cm']
                   - (np.abs(v) * length / 2 * self.sigma_time)
                   - self.time_before_shower.tel[telescope_id])
        t_end = (start_parameters['t_cm']
                 + (np.abs(v) * length / 2 * self.sigma_time)
                 + self.time_after_shower.tel[telescope_id])

        mask_time = (times < t_end) * (times > t_start)

        return mask_pixel, mask_time

    def fit(self, fit_params):
        """
            Performs the fitting procedure.

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
        try:
            self.error_parameters = m.errors.to_dict()
        except (KeyError, AttributeError, RuntimeError):
            self.error_parameters = {key: np.nan for key in self.names_parameters}

    def predict(self, focal_length, fit_params):
        """
            Call the fitting procedure and fill the results.

        """
        container = DL1LikelihoodParametersContainer(lhfit_call_status=1)
        self.fit(fit_params)
        container.lhfit_TS = self.fcn

        container.lhfit_x = self.end_parameters['x_cm'] * u.m
        container.lhfit_y = self.end_parameters['y_cm'] * u.m
        container.lhfit_r = np.sqrt(container.lhfit_x ** 2 + container.lhfit_y ** 2)
        container.lhfit_phi = np.arctan2(container.lhfit_y, container.lhfit_x)
        if self.end_parameters['psi'] > np.pi:
            self.end_parameters['psi'] -= 2 * np.pi
        if self.end_parameters['psi'] < -np.pi:
            self.end_parameters['psi'] += 2 * np.pi
        container.lhfit_psi = self.end_parameters['psi'] * u.rad
        length_asy = 1 + self.end_parameters['rl'] if self.end_parameters['rl'] >= 0 else 1 / (
                1 - self.end_parameters['rl'])
        lhfit_length_m = ((1.0 + length_asy)
                          * self.end_parameters['length'] / 2.0) * u.m
        container.lhfit_length = np.rad2deg(np.arctan(lhfit_length_m/focal_length))
        container.lhfit_width = self.end_parameters['wl'] * container.lhfit_length

        container.lhfit_time_gradient = self.end_parameters['v']
        container.lhfit_ref_time = self.end_parameters['t_cm']

        container.lhfit_wl = u.Quantity(self.end_parameters['wl'])
        container.lhfit_intensity = self.end_parameters['charge']
        container.lhfit_log_intensity = np.log10(container.lhfit_intensity)
        container.lhfit_t_68 = container.lhfit_length.value * container.lhfit_time_gradient
        container.lhfit_area = container.lhfit_length * container.lhfit_width
        container.lhfit_length_asymmetry = self.end_parameters['rl']

        return container

    def __str__(self):
        """
            Define the print format of DL0Fitter objects.

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


def init_centroid(dl1_container, geom, image, no_asymmetry):
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
    image : array_like
        Charge in each pixel
    no_asymmetry: bool
        Information on the used spatial model

    """
    if no_asymmetry:
        return dl1_container.x, dl1_container.y
    else:
        return asy_centroid(geom, image)


def asy_centroid(geom, image):
    """
    Compute a centroid weighting as the signal squared for asymmetric models

    Parameters
    ----------
    geom: `ctapipe.instrument.CameraGeometry`
        Camera geometry
    image : array_like
        Charge in each pixel

    Returns
    -------
    (x_cm, y_cm): `astropy.units.Quantity`

    """
    unit = geom.pix_x.unit
    pix_x = geom.pix_x.to_value(unit)
    pix_y = geom.pix_y.to_value(unit)
    image = np.asanyarray(image, dtype=np.float64)
    image = np.ma.filled(image, 0)
    cog_x = np.average(pix_x, weights=image ** 2)
    cog_y = np.average(pix_y, weights=image ** 2)
    return [u.Quantity(cog_x, unit), u.Quantity(cog_y, unit)]
