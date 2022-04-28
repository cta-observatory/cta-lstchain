import inspect
import logging

from iminuit import Minuit
import numpy as np
from copy import copy
import astropy.units as u

from lstchain.data.normalised_pulse_template import NormalizedPulseTemplate

from ctapipe.core import Component
from ctapipe.core.traits import Bool, Float, Int, Path
from lstchain.calib.camera.pixel_threshold_estimation import get_bias_and_std
from lstchain.io.lstcontainers import DL1LikelihoodParametersContainer

logger = logging.getLogger(__name__)

try:
    from lstchain.reco.log_pdf_CC import log_pdf_ll as log_pdf_ll
    from lstchain.reco.log_pdf_CC import log_pdf_hl as log_pdf_hl
    from lstchain.reco.log_pdf_CC import asygaussian2d as asygaussian2d
except ImportError:
    pass


class TimeWaveformFitter(Component):
    sigma_s = Float(1, help='Width of the single photo-electron peak distribution.', allow_none=False).tag(config=True)
    crosstalk = Float(0, help='Average pixel crosstalk.', allow_none=False).tag(config=True)
    sigma_space = Float(4, help='Size of the region on which the fit is performed relative to the image extension.',
                        allow_none=False).tag(config=True)
    sigma_time = Float(3, help='Time window on which the fit is performed relative to the image temporal extension.',
                       allow_none=False).tag(config=True)
    time_before_shower = Float(10, help='Additional time at the start of the fit temporal window.',
                               allow_none=False).tag(config=True)
    time_after_shower = Float(20, help='Additional time at the end of the fit temporal window.',
                              allow_none=False).tag(config=True)
    use_weight = Bool(False, help='If True, the brightest sample is twice as important as the dimmest pixel in the '
                                  'likelihood. If false all samples are equivalent.', allow_none=False).tag(config=True)
    no_asymmetry = Bool(False, help='If true, the asymmetry of the spatial model is fixed to 0.',
                        allow_none=False).tag(config=True)
    use_interleaved = Path(None, help='Location of the dl1 file used to estimate the pedestal exploiting interleaved'
                                      ' events.', allow_none=True).tag(config=True)
    telescope_id = Int(1, help='Id of the telescope in use.', allow_none=False).tag(config=True)
    # Allows only one telescope for now, need to use telescope-wise information later
    n_peaks = Int(50, help='Maximum brightness (p.e.) for which the full likelihood computation is used. '
                           'If the Poisson term for Np.e.>n_peak is more than 1e-6 a Gaussian approximation is used.',
                  allow_none=False).tag(config=True)
    verbose = Int(0, help='4 - used for tests: create debug plots\n'
                          '3 - create debug plots, wait for input after each event, increase minuit verbose level\n'
                          '2 - create debug plots, increase minuit verbose level\n'
                          '1 - increase minuit verbose level\n'
                          '0 - silent', allow_none=False).tag(config=True)

    def __init__(self, subarray, config=None, parent=None, **kwargs):
        super().__init__(config=config, parent=parent, **kwargs)
        self.subarray = subarray
        self.geometry = self.subarray.tel[self.telescope_id].camera.geometry
        self.pix_x = self.geometry.pix_x.to_value(u.m)
        self.pix_y = self.geometry.pix_y.to_value(u.m)
        self.pix_area = self.geometry.pix_area.to_value(u.m ** 2)
        self.r_max = self.geometry.guess_radius().to_value(u.m)
        self.focal_length = subarray.tel[self.telescope_id].optics.equivalent_focal_length
        readout = self.subarray.tel[self.telescope_id].camera.readout
        sampling_rate = readout.sampling_rate.to_value(u.GHz)
        self.dt = (1.0 / sampling_rate)
        self.names_parameters = list(inspect.signature(self.log_pdf).parameters)
        self.template = NormalizedPulseTemplate.load_from_eventsource(subarray.tel[self.telescope_id].camera.readout)
        self.template_time_of_max = self.template.compute_time_of_max()
        if self.use_interleaved is None:  # test to include interleaved correction on pedestal, NOT FUNCTIONAL
            self.pedestal_std = None
        else:
            _, self.pedestal_std = get_bias_and_std(config['lh_fit_config']['use_interleaved'])
        self.end_parameters = None
        self.error_parameters = None
        self.fcn = None

        poisson_peaks = np.arange(self.n_peaks, dtype=int)
        poisson_peaks[0] = 1
        self.factorial = np.cumprod(poisson_peaks)

    def __call__(self, event, dl1_container):
        self.image = event.dl1.tel[self.telescope_id].image
        hillas_signal_pixels = event.dl1.tel[self.telescope_id].image_mask
        start_x_cm, start_y_cm = init_centroid(dl1_container,
                                               self.geometry[hillas_signal_pixels],
                                               self.image[hillas_signal_pixels],
                                               self.no_asymmetry
                                               )

        waveform = event.r1.tel[self.telescope_id].waveform

        dl1_calib = event.calibration.tel[self.telescope_id].dl1
        time_shift = dl1_calib.time_shift
        if dl1_calib.pedestal_offset is not None:
            waveform = waveform - dl1_calib.pedestal_offset[:, np.newaxis]

        self.n_pixels, self.n_samples = waveform.shape
        self.times = np.arange(0, self.n_samples) * self.dt
        selected_gains = event.r1.tel[self.telescope_id].selected_gain_channel
        self.is_high_gain = (selected_gains == 0)

        v = dl1_container.time_gradient
        psi = dl1_container.psi.to_value(u.rad)
        # We use only positive time gradients and psi is projected in [-pi,pi] from [-pi/2,pi/2]
        if v < 0:
            if psi >= 0:
                psi = psi - np.pi
            else:
                psi = psi + np.pi

        start_length = max(np.tan(dl1_container.length.to_value(u.rad)) * self.focal_length.to_value(u.m), 0.02)
        self.start_parameters = {'x_cm': start_x_cm.to_value(u.m),
                                 'y_cm': start_y_cm.to_value(u.m),
                                 'charge': dl1_container.intensity,
                                 't_cm': dl1_container.intercept - self.template_time_of_max,
                                 'v': np.abs(v),
                                 'psi': psi,
                                 'length': start_length,
                                 'wl': max(dl1_container.wl, 0.01),
                                 'rl': 0.0
                                 }

        if np.isnan(self.start_parameters['t_cm']):
            self.start_parameters['t_cm'] = 0.
        if np.isnan(self.start_parameters['v']):
            self.start_parameters['v'] = 40

        t_max = self.n_samples * self.dt
        v_min, v_max = 0, max(2 * self.start_parameters['v'], 50)
        rl_min, rl_max = -9, 9
        if self.no_asymmetry:
            rl_min, rl_max = 0.0, 0.0

        self.bound_parameters = {'x_cm': (start_x_cm.to_value(u.m)
                                          - 1.0 * start_length,
                                          start_x_cm.to_value(u.m)
                                          + 1.0 * start_length),
                                 'y_cm': (start_y_cm.to_value(u.m)
                                          - 1.0 * start_length,
                                          start_y_cm.to_value(u.m)
                                          + 1.0 * start_length),
                                 'charge': (dl1_container.intensity * 0.25,
                                            dl1_container.intensity * 4.0),
                                 't_cm': (-10, t_max + 10),
                                 'v': (v_min, v_max),
                                 'psi': (-np.pi * 2.0, np.pi * 2.0),
                                 'length': (0.001,
                                            min(2 * start_length,
                                                self.r_max)),
                                 'wl': (0.001, 1.0),
                                 'rl': (rl_min, rl_max)
                                 }
        self.end_parameters = None
        self.error_parameters = None
        self.fcn = None

        self.mask_pixel, self.mask_time = self.clean_data()
        spatial_ones = np.ones(np.sum(self.mask_pixel))

        self.is_high_gain = self.is_high_gain[self.mask_pixel]
        self.sig_s = spatial_ones * self.sigma_s
        self.crosstalks = spatial_ones * self.crosstalk

        self.times = (np.arange(0, self.n_samples) * self.dt)[self.mask_time]
        self.time_shift = time_shift[self.mask_pixel]

        self.p_x = self.pix_x[self.mask_pixel]
        self.p_y = self.pix_y[self.mask_pixel]
        self.pix_area = self.geometry.pix_area[self.mask_pixel].to_value(u.m ** 2)

        self.data = waveform
        self.error = copy(self.pedestal_std)

        filter_pixels = np.nonzero(~self.mask_pixel)
        filter_times = np.nonzero(~self.mask_time)

        if self.error is None:
            std = np.std(self.data[~self.mask_pixel])
            self.error = np.full(self.data.shape[0], std)
        else:
            std = np.std(self.data[~self.mask_pixel])
            self.error[self.error <= std / 2] = std

        self.data = np.delete(self.data, filter_pixels, axis=0)
        self.data = np.delete(self.data, filter_times, axis=1)
        self.error = np.delete(self.error, filter_pixels, axis=0)

        return self.predict()

    def clean_data(self):
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
        x_cm = self.start_parameters['x_cm']
        y_cm = self.start_parameters['y_cm']
        length = self.start_parameters['length']
        width = self.start_parameters['wl'] * length
        psi = self.start_parameters['psi']

        dx = self.pix_x - x_cm
        dy = self.pix_y - y_cm

        lon = dx * np.cos(psi) + dy * np.sin(psi)
        lat = dx * np.sin(psi) - dy * np.cos(psi)

        mask_pixel = ((lon / (length + 0.0228)) ** 2 + (
                lat / (width + 0.0228)) ** 2) < self.sigma_space ** 2

        v = self.start_parameters['v']
        t_start = (self.start_parameters['t_cm']
                   - (np.abs(v) * length / 2 * self.sigma_time)
                   - self.time_before_shower)
        t_end = (self.start_parameters['t_cm']
                 + (np.abs(v) * length / 2 * self.sigma_time)
                 + self.time_after_shower)

        mask_time = (self.times < t_end) * (self.times > t_start)

        return mask_pixel, mask_time

    def fit(self):
        """
            Performs the fitting procedure.

        """

        def f(*args):
            return -2 * self.log_likelihood(*args)

        print_level = 2 if self.verbose in [1, 2, 3] else 0
        m = Minuit(f,
                   name=self.names_parameters,
                   **self.start_parameters)
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

    def log_pdf(self, charge, t_cm, x_cm, y_cm,
                length, wl, psi, v, rl):
        """
            Compute the log likelihood of the model used for a set of input
            parameters.

        Parameters
        ----------
        self: Contains all the information about the data and calibration
        charge: float
            Charge of the peak of the spatial model
        t_cm: float
            Time of the middle of the energy deposit in the camera
            for the temporal model
        x_cm, y_cm: float
            Position of the center of the spatial model
        length, wl: float
            Spatial dispersion of the model along the main and
            fraction of this dispersion along the minor axis
        psi: float
            Orientation of the main axis of the spatial model and of the
            propagation of the temporal model
        v: float
            Velocity of the evolution of the signal over the camera
        rl: float
            Asymmetry of the spatial model along the main axis
        """
        dx = (self.p_x - x_cm)
        dy = (self.p_y - y_cm)
        long = dx * np.cos(psi) + dy * np.sin(psi)
        p = [v, t_cm]
        t = np.polyval(p, long)
        t = self.times[..., None] - t
        t = t.T - self.time_shift[..., None]
        templates = (self.template(t, 'HG').T * self.is_high_gain
                     + self.template(t, 'LG').T * (~self.is_high_gain)).T

        rl = 1 + rl if rl >= 0 else 1 / (1 - rl)

        mu = asygaussian2d(charge * self.pix_area,
                           self.p_x,
                           self.p_y,
                           x_cm,
                           y_cm,
                           wl * length,
                           length,
                           psi,
                           rl)

        # We reduce the sum by limiting to the poisson term contributing for
        # more than 10^-6. The limits are approximated by 2 broken linear
        # function obtained for 0 crosstalk.
        # The choice of kmin and kmax is currently not done on a pixel basis
        mask_LL = (mu <= self.n_peaks / 1.096 - 47.8) & (mu > 0)
        mask_HL = ~mask_LL

        if len(mu[mask_LL]) == 0:
            kmin, kmax = 0, self.n_peaks
        else:
            min_mu = min(mu[mask_LL])
            max_mu = max(mu[mask_LL])
            if min_mu < 120:
                kmin = int(0.66 * (min_mu - 20))
            else:
                kmin = int(0.904 * min_mu - 42.8)
            if max_mu < 120:
                kmax = int(np.ceil(1.34 * (max_mu - 20) + 45))
            else:
                kmax = int(np.ceil(1.096 * max_mu + 47.8))
        if kmin < 0:
            kmin = 0
        if kmax > self.n_peaks:
            kmax = int(self.n_peaks)
            logger.warning("kmax forced to %s", kmax)

        if self.use_weight:
            weight = 1.0 + (self.data / np.max(self.data))
        else:
            weight = np.ones(self.data.shape)

        mask_k = (np.arange(self.n_peaks) >= kmin) & (np.arange(self.n_peaks) < kmax)

        log_pdf_faint = log_pdf_ll(mu[mask_LL],
                                   self.data[mask_LL],
                                   self.error[mask_LL],
                                   self.crosstalks[mask_LL],
                                   self.sig_s[mask_LL],
                                   templates[mask_LL],
                                   self.factorial[mask_k],
                                   kmin, kmax,
                                   weight[mask_LL])

        log_pdf_bright = log_pdf_hl(mu[mask_HL],
                                   self.data[mask_HL],
                                   self.error[mask_HL],
                                   self.crosstalks[mask_HL],
                                   templates[mask_HL],
                                   weight[mask_HL])

        log_pdf = (log_pdf_faint + log_pdf_bright) / np.sum(weight)

        return log_pdf

    def predict(self):
        """
            Call the fitting procedure and fill the results.

        """
        container = DL1LikelihoodParametersContainer(lhfit_call_status=1)
        self.fit()
        GoF = 0.0  # removing the goodness of fit as it is not accurate for now
        container.lhfit_goodness_of_fit = GoF
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
        container.lhfit_length = np.rad2deg(np.arctan(lhfit_length_m/self.focal_length))
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
        s =  'Event processed\n'
        s += 'Start parameters :\n\t{}\n'.format(self.start_parameters)
        s += 'Bound parameters :\n\t{}\n'.format(self.bound_parameters)
        s += 'End parameters :\n\t{}\n'.format(self.end_parameters)
        s += 'Error parameters :\n\t{}\n'.format(self.error_parameters)
        s += 'Log-Likelihood :\t{}'.format(self.log_likelihood(**self.end_parameters))

        return s

    def log_likelihood(self, *args, **kwargs):
        """Compute the log-likelihood used in the fitting procedure."""
        llh = self.log_pdf(*args, **kwargs)
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
