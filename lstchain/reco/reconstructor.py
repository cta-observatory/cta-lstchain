import inspect
import logging

from iminuit import Minuit
import numpy as np
import numexpr as ne
import matplotlib.pyplot as plt
from copy import copy
import astropy.units as u

from lstchain.data.normalised_pulse_template import NormalizedPulseTemplate

from ctapipe.core import Component
from ctapipe.core.traits import Bool, Float, Int, Path
from lstchain.calib.camera.pixel_threshold_estimation import get_bias_and_std
from lstchain.io.lstcontainers import DL1LikelihoodParametersContainer
from lstchain.image.pdf import log_gaussian, log_asygaussian2d
from lstchain.visualization.camera import display_array_camera

logger = logging.getLogger(__name__)


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
        self.pix_area = self.geometry.pix_area.to(u.m ** 2).value
        self.r_max = np.sqrt(self.pix_x ** 2 + self.pix_y ** 2).max()
        self.focal_length = subarray.tel[self.telescope_id].optics.equivalent_focal_length
        readout = self.subarray.tel[self.telescope_id].camera.readout
        sampling_rate = readout.sampling_rate.to_value(u.GHz)
        self.dt = (1.0 / sampling_rate)
        self.labels = {'charge': 'Charge [p.e.]',
                       't_cm': '$t_{CM}$ [ns]',
                       'x_cm': '$x_{CM}$ [m]',
                       'y_cm': '$y_{CM}$ [m]',
                       'length': r'$\sigma_l$ [m]',
                       'wl': r'$\sigma_w$ / $\sigma_l$',
                       'psi': r'$\psi$ [rad]',
                       'v': '$v$ [m/ns]',
                       'rl': 'length asymmetry'
                       }
        self.names_parameters = list(inspect.signature(self.log_pdf).parameters)
        self.template = NormalizedPulseTemplate.load_from_eventsource(subarray.tel[1].camera.readout)
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
        log_factorial = np.log(poisson_peaks)
        self.log_factorial = np.cumsum(log_factorial)

    def __call__(self, event, dl1_container):
        self.event_id = event.index.event_id
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
        self.pix_area = self.geometry.pix_area[self.mask_pixel].to_value(u.m **2)

        self.data = waveform
        self.error = copy(self.pedestal_std)

        filter_pixels = np.arange(self.n_pixels)[~self.mask_pixel]
        filter_times = np.arange(self.n_samples)[~self.mask_time]

        if self.error is None:
            std = np.std(self.data[~self.mask_pixel])
            self.error = np.ones(self.data.shape) * std
        else:
            std = np.std(self.data[~self.mask_pixel])
            self.error[self.error <= std / 2] = std
            self.error = self.error[..., None] * np.ones(self.data.shape)

        self.data = np.delete(self.data, filter_pixels, axis=0)
        self.data = np.delete(self.data, filter_times, axis=1)
        self.error = np.delete(self.error, filter_pixels, axis=0)
        self.error = np.delete(self.error, filter_times, axis=1)

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
            pass

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

        log_mu = log_asygaussian2d(size=charge * self.pix_area,
                                   x=self.p_x,
                                   y=self.p_y,
                                   x_cm=x_cm,
                                   y_cm=y_cm,
                                   width=wl * length,
                                   length=length,
                                   psi=psi,
                                   rl=rl)
        mu = ne.evaluate("exp(log_mu)", local_dict=dict(log_mu=log_mu))
        mu[mu <= 0] = 1e-320

        # We reduce the sum by limiting to the poisson term contributing for
        # more than 10^-6. The limits are approximated by 2 broken linear
        # function obtained for 0 crosstalk.
        # The choice of kmin and kmax is currently not done on a pixel basis
        mask_LL = (mu <= self.n_peaks / 1.096 - 47.8)

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

        photo_peaks = np.arange(kmin, kmax, dtype=int)
        crosstalk_factor = photo_peaks[..., None] * self.crosstalks[mask_LL]

        # Compute the Poisson term in the pixel likelihood for
        # low luminosity pixels
        mu_plus_crosstalk = mu[mask_LL] + crosstalk_factor
        log_mu_plus_crosstalk = ne.evaluate("log(mu_plus_crosstalk)",
                                            local_dict=dict(mu_plus_crosstalk=mu_plus_crosstalk))
        log_mu_plus_crosstalk = ((photo_peaks - 1)
                                 * log_mu_plus_crosstalk.T).T
        log_poisson = (log_mu[mask_LL]
                       - self.log_factorial[kmin:kmax][..., None]
                       - mu_plus_crosstalk
                       + log_mu_plus_crosstalk)

        # Compute the Gaussian term in the pixel likelihood for
        # low luminosity pixels
        signal = self.data

        if self.use_weight:
            weight = 1.0 + (signal / np.max(signal))
        else:
            weight = np.ones(signal.shape)

        mean = (photo_peaks
                * templates[mask_LL][..., None])
        sigma_n = (photo_peaks
                   * ((self.sig_s[mask_LL][..., None] * templates[mask_LL]) ** 2)[..., None])
        sigma_n = (self.error[mask_LL] ** 2)[..., None] + sigma_n
        sigma_n = ne.evaluate("sqrt(sigma_n)", local_dict=dict(sigma_n=sigma_n))
        log_gauss = log_gaussian(signal[mask_LL][..., None], mean, sigma_n)

        # Compute the pixel likelihood using a Gaussian approximation for
        # high luminosity pixels
        if np.any(~mask_LL):
            mu_hat = ((mu[~mask_LL] / (1 - self.crosstalks[~mask_LL]))[..., None]
                      * templates[~mask_LL])
            sigma_hat = (((mu[~mask_LL]
                           / np.power(1 - self.crosstalks[~mask_LL], 3))[..., None]
                          * templates[~mask_LL] ** 2))
            sigma_hat = np.sqrt((self.error[~mask_LL] ** 2) + sigma_hat)

            if self.use_weight:
                log_pixel_pdf_HL = weight[~mask_LL] * log_gaussian(signal[~mask_LL], mu_hat, sigma_hat)
            else:
                log_pixel_pdf_HL = log_gaussian(signal[~mask_LL], mu_hat, sigma_hat)
            n_points_HL = np.sum(weight[~mask_LL])
        else:
            log_pixel_pdf_HL, n_points_HL = np.asarray([0]), 0

        log_poisson = np.expand_dims(log_poisson.T, axis=1)
        log_pixel_pdf_LL = ne.evaluate("log_poisson + log_gauss",
                                       local_dict=dict(log_poisson=log_poisson, log_gauss=log_gauss))
        if self.use_weight:
            pixel_pdf_LL = np.sum(np.exp(log_pixel_pdf_LL), axis=2)
        else:
            pixel_pdf_LL = ne.evaluate("sum(exp(log_pixel_pdf_LL), axis=2)",
                                       local_dict=dict(log_pixel_pdf_LL=log_pixel_pdf_LL))

        mask = (pixel_pdf_LL <= 0)
        pixel_pdf_LL[mask] = 1e-320
        n_points_LL = np.sum(weight[mask_LL])
        if self.use_weight:
            log_pixel_pdf_LL = weight[mask_LL] * np.log(pixel_pdf_LL)
        else:
            log_pixel_pdf_LL = np.log(pixel_pdf_LL)
        log_pdf = ((log_pixel_pdf_LL.sum() + log_pixel_pdf_HL.sum())
                   / (n_points_LL + n_points_HL))

        return log_pdf

    def predict(self, **kwargs):
        """
            Call the fitting procedure and fill the results.

        """
        container = DL1LikelihoodParametersContainer(lhfit_call_status=1)
        self.fit(**kwargs)
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

        container.lhfit_wl = self.end_parameters['wl']
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
        s =  'Event {}\n'.format(self.event_id)
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

    def plot_debug(self):
        """
        Create a set of plots for one event

        """
        image = self.image
        self.plot_event(image, init=True, save=True, ids=str(self.event_id))
        self.plot_waveforms(image, save=True, ids=str(self.event_id))
        self.plot_event(image, save=True, ids=str(self.event_id))
        self.plot_residual(image, save=True, ids=str(self.event_id))
        self.plot_model(save=True, ids=str(self.event_id))

        for params in self.start_parameters.keys():
            self.plot_likelihood(params, save=True, ids=str(self.event_id))
        self.plot_likelihood('x_cm', 'y_cm', save=True, ids=str(self.event_id))

        if self.verbose == 3:
            print("event plot produced, press Enter to continue or Ctrl+C and Enter to stop")
            input()

    def plot_1dlikelihood(self, parameter_name, axes=None, size=1000,
                          x_label=None, invert=False, loc='best'):
        """
            Plot the 1D evolution of the log-likelihood for a parameter
            when fixing the other parameters to their end value.

            Parameters
            ----------
            parameter_name: string
                Parameter over which the log-likelihood needs to be plotted
            axes: matplotlib.pyplot.axis
                Axis used to store the figure
                If None, a new one is created
            size: int
                Number of points of the likelihood curve
            x_label: string
                Label of the x axis
            invert: bool
                If True, invert the x and y axis
            loc: string
                Legend position

            Returns
            -------
            axes: matplotlib.pyplot.axis
                Axis object filled with the 1D log-likelihood figure

        """
        key = parameter_name

        if key not in self.names_parameters:
            raise NameError('Parameter : {} not in existing parameters :'
                            '{}'.format(key, self.names_parameters))

        x = np.linspace(self.bound_parameters[key][0],
                        self.bound_parameters[key][1], num=size)
        if axes is None:
            fig = plt.figure()
            axes = fig.add_subplot(111)

        params = copy(self.end_parameters)
        llh = np.zeros(x.shape)

        for i, xx in enumerate(x):
            params[key] = xx
            llh[i] = self.log_likelihood(**params)

        x_label = self.labels[key] if x_label is None else x_label

        if not invert:
            axes.plot(x, -llh, color='r')
            axes.axvline(self.end_parameters[key], linestyle='--', color='k',
                         label='Fitted value {:.2f}'.format(
                             self.end_parameters[key]))
            axes.axvline(self.start_parameters[key], linestyle='--',
                         color='b', label='Starting value {:.2f}'.format(
                    self.start_parameters[key]
                ))
            axes.set_ylabel(r'-$\ln \mathcal{L}$')
            axes.set_xlabel(x_label)

        else:

            axes.plot(-llh, x, color='r')
            axes.axhline(self.end_parameters[key], linestyle='--',
                         color='k',
                         label='Fitted value {:.2f}'.format(
                             self.end_parameters[key]))
            axes.axhline(self.start_parameters[key], linestyle='--',
                         color='b', label='Starting value {:.2f}'.format(
                    self.start_parameters[key]
                ))
            axes.axhspan(self.bound_parameters[key][0],
                         self.bound_parameters[key][1], label='bounds',
                         alpha=0.5, facecolor='k')
            axes.set_xlabel(r'-$\ln \mathcal{L}$')
            axes.set_ylabel(x_label)
            axes.xaxis.set_label_position('top')

        axes.legend(loc=loc)
        return axes

    def plot_2dlikelihood(self, parameter_1, parameter_2=None, size=100,
                          x_label=None, y_label=None):
        """
            Plot the 2D evolution of the log-likelihood for a pair of
            parameters when fixing the other parameters to their end value.

            Parameters
            ----------
            parameter_1: string
                First parameter over which the function needs to be plotted
            parameter_2: string
                Second parameter over which the function needs to be plotted
            size: int or (int, int)
                Number of points of the likelihood per dimension
            x_label: string
                Label of the x axis
            y_label: string
                Label of the y axis

            Returns
            -------
            axes: matplotlib.pyplot.axis
                Axis object filled with the 2D log-likelihood figures

        """

        if isinstance(size, int):
            size = (size, size)

        key_x = parameter_1
        key_y = parameter_2
        x = np.linspace(self.bound_parameters[key_x][0],
                        self.bound_parameters[key_x][1], num=size[0])
        y = np.linspace(self.bound_parameters[key_y][0],
                        self.bound_parameters[key_y][1], num=size[1])
        dx = x[1] - x[0]
        dy = y[1] - y[0]

        params = copy(self.end_parameters)
        llh = np.zeros(size)

        for i, xx in enumerate(x):
            params[key_x] = xx
            for j, yy in enumerate(y):
                params[key_y] = yy
                llh[i, j] = self.log_likelihood(**params)

        fig = plt.figure()
        left, width = 0.1, 0.6
        bottom, height = 0.1, 0.6
        spacing = 0.005
        rect_center = [left, bottom, width, height]
        rect_x = [left, bottom + height + spacing, width, 0.2]
        rect_y = [left + width + spacing, bottom, 0.2, height]
        axes = fig.add_axes(rect_center)
        axes_x = fig.add_axes(rect_x)
        axes_y = fig.add_axes(rect_y)
        axes.tick_params(direction='in', top=True, right=True)
        self.plot_1dlikelihood(parameter_name=parameter_1, axes=axes_x,
                               loc='upper left')
        self.plot_1dlikelihood(parameter_name=parameter_2, axes=axes_y,
                               invert=True, loc='lower right')
        axes_x.tick_params(direction='in', labelbottom=False)
        axes_y.tick_params(direction='in', labelleft=False)

        axes_x.set_xlabel('')
        axes_y.set_ylabel('')
        axes_x.tick_params(
            axis='x',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            bottom=False,  # ticks along the bottom edge are off
            top=False,  # ticks along the top edge are off
            labelbottom=False)
        axes_y.tick_params(
            axis='y',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            bottom=False,  # ticks along the bottom edge are off
            top=False,  # ticks along the top edge are off
            labelbottom=False)

        x_label = self.labels[key_x] if x_label is None else x_label
        y_label = self.labels[key_y] if y_label is None else y_label

        im = axes.imshow(-llh.T, origin='lower', extent=[x.min() - dx / 2.,
                                                         x.max() - dx / 2.,
                                                         y.min() - dy / 2.,
                                                         y.max() - dy / 2.],
                         aspect='auto')

        axes.scatter(self.end_parameters[key_x], self.end_parameters[key_y],
                     marker='x', color='w', label='- log Likelihood')
        axes.set_xlabel(x_label)
        axes.set_ylabel(y_label)
        axes.legend(loc='upper left')
        plt.colorbar(mappable=im, ax=axes_y, label=r'-$\ln \mathcal{L}$')
        axes_x.set_xlim(x.min(), x.max())
        axes_y.set_ylim(y.min(), y.max())

        return axes

    def plot_likelihood(self, parameter_1, parameter_2=None,
                        axes=None, size=100,
                        x_label=None, y_label=None,
                        save=False, ids=''):
        """
            Plot the 1D or 2D likelihood.

            Parameters
            ----------
            parameter_1: string
                Parameter over which the function needs to be plotted
            parameter_2: string
                Second parameter over which the function needs to be plotted
            axes: `matplotlib.pyplot.axis`
                Axis used to store the figure
                If None, a new one is created
            size: int
                Number of points of the likelihood curve for each dimension
            x_label: string
                Label of the x axis
            y_label: string
                Label of the y axis
            save: bool
                Save and close the figure if True, return it otherwise
            ids: string
                Can be used to modify the save location

            Returns
            -------
            None or `matplotlib.pyplot.axis` object filled with the log-likelihood figure

        """
        if parameter_2 is None:
            axes = self.plot_1dlikelihood(parameter_name=parameter_1,
                                          axes=axes, x_label=x_label,
                                          size=size)
        else:
            axes = self.plot_2dlikelihood(parameter_1,
                                          parameter_2=parameter_2,
                                          size=size,
                                          x_label=x_label,
                                          y_label=y_label)
        if save:
            axes.get_figure().savefig('event/' + ids + '_'
                                      + parameter_1 + '_' + str(parameter_2) + '.png')
            plt.close()
        return None if save else axes

    def plot_event(self, image, n_sigma=3, init=False, show_ellipsis=True, save=False, ids=''):
        """
            Plot the image of the event in the camera along with the extracted
            ellipsis before or after the fitting procedure.

        Parameters
        ----------
        image:
            Distribution of signal for the event in number of p.e.
        n_sigma: float
            Multiplicative factor on the extracted width and length
            used for the displayed ellipsis
        init: boolean
            If True, use the starting parameters for the ellipsis
            If False, use the ending parameters for the ellipsis
        show_ellipsis: boolean
            If True, display the ellipsis
        save: bool
            Save and close the figure if True, return it otherwise
        ids: string
            Can be used to modify the save location
        Returns
        -------
        cam_display: `ctapipe.visualization.CameraDisplay`
            Camera image using matplotlib

        """
        cam_display = display_array_camera(image,
                                           camera_geometry=self.geometry)
        if init:
            params = self.start_parameters
        else:
            params = self.end_parameters

        length = n_sigma * params['length']
        psi = params['psi']
        if show_ellipsis:
            cam_display.add_ellipse(centroid=(params['x_cm'],
                                              params['y_cm']),
                                    width=n_sigma * params['wl'] * params['length'],
                                    length=length,
                                    angle=psi,
                                    linewidth=6, color='r', linestyle='--',
                                    label=r'{} $\sigma$ contour'.format(n_sigma))
            cam_display.axes.legend(loc='best')
        if init:
            cam_display.highlight_pixels(self.mask_pixel, color='r')

        if save:
            cam_display.axes.get_figure().savefig('event/' + ids +
                                                  '_init' + str(init) + '.png')
            plt.close()
        return None if save else cam_display

    def plot_residual(self, image, save=False, ids=''):
        """
            Plot the residuals image- spatial_model in the camera after fitting

        Parameters
        ----------
        image:
            Distribution of signal for the event in number of p.e.
        save: bool
            Save and close the figure if True, return it otherwise
        ids: string
            Can be used to modify the save location

        Returns
        -------
        cam_display: `ctapipe.visualization.CameraDisplay`
            Camera image using matplotlib

        """

        params = self.end_parameters

        rl = 1 + params['rl'] if params['rl'] >= 0 else 1 / (1 - params['rl'])
        log_mu = log_asygaussian2d(size=params['charge']
                                        * self.geometry.pix_area.to(u.m ** 2).value,
                                   x=self.geometry.pix_x.value,
                                   y=self.geometry.pix_y.value,
                                   x_cm=params['x_cm'],
                                   y_cm=params['y_cm'],
                                   width=params['wl'] * params['length'],
                                   length=params['length'],
                                   psi=params['psi'],
                                   rl=rl)
        mu = np.exp(log_mu)
        residual = image - mu

        cam_display = display_array_camera(residual,
                                           camera_geometry=self.geometry)
        if save:
            cam_display.axes.get_figure().savefig('event/' + ids +
                                                  '_residuals.png')
            plt.close()
        return None if save else cam_display

    def plot_model(self, save=False, ids=''):
        """
        Create a CameraDisplay object showing the spatial model fitted to
        the current event

        Parameters
        -------
        save: bool
            Save and close the figure if True, return it otherwise
        ids: string
            Can be used to modify the save location

        Returns
        -------
        cam_display: `ctapipe.visualization.CameraDisplay`
            Camera image using matplotlib

        """

        params = self.end_parameters
        rl = 1 + params['rl'] if params['rl'] >= 0 else 1 / (1 - params['rl'])
        log_mu = log_asygaussian2d(size=params['charge']
                                        * self.geometry.pix_area.to(u.m ** 2).value,
                                   x=self.geometry.pix_x.value,
                                   y=self.geometry.pix_y.value,
                                   x_cm=params['x_cm'],
                                   y_cm=params['y_cm'],
                                   width=params['wl'] * params['length'],
                                   length=params['length'],
                                   psi=params['psi'],
                                   rl=rl)
        mu = np.exp(log_mu)

        cam_display = display_array_camera(mu,
                                           camera_geometry=self.geometry)
        if save:
            cam_display.axes.get_figure().savefig('event/' + ids +
                                                  '_model.png')
            plt.close()
        return None if save else cam_display

    def plot_waveforms(self, image, axes=None, save=False, ids=''):
        """
            Plot the intensity of the signal in the camera as a function of
            time and of the position projected on the main axis of the fitted
            ellipsis.

        Parameters
        ----------
        image:
            Distribution of signal for the event in number of p.e.
        axes: `matplotlib.pyplot.axis`
            Axis used to store the figure
            If None, a new one is created
        save: bool
            Save and close the figure if True, return it otherwise
        ids: string
            Can be used to modify the save location

        Returns
        -------
        axes: `matplotlib.pyplot.axis`
            Object filled with the figure
        """
        image = image[self.mask_pixel]
        n_pixels = min(20, len(image))
        pixels = np.argsort(image)[-n_pixels:]
        dx = (self.pix_x[pixels] - self.end_parameters['x_cm'])
        dy = (self.pix_y[pixels] - self.end_parameters['y_cm'])
        long_pix = dx * np.cos(self.end_parameters['psi']) + dy * np.sin(
            self.end_parameters['psi'])
        fitted_times = np.polyval(
            [self.end_parameters['v'], self.end_parameters['t_cm']], long_pix)
        times_index = np.argsort(fitted_times)

        waveforms = self.data[pixels]
        waveforms = waveforms[times_index]
        long_pix = long_pix[times_index]
        fitted_times = fitted_times[times_index]

        X, Y = np.meshgrid(self.times, long_pix)

        if axes is None:
            fig = plt.figure()
            axes = fig.add_subplot(111)
        M = axes.pcolormesh(X, Y, waveforms)
        axes.set_xlabel('time [ns]')
        axes.set_ylabel('Longitude [m]')
        label = (self.labels['t_cm']
                 + ' : {:.2f} [ns]'.format(self.end_parameters['t_cm']))
        label += ('\n' + self.labels['v']
                  + ' : {:.2f} [m/ns]'.format(self.end_parameters['v']))
        axes.plot(fitted_times, long_pix, color='r',
                  label=label)
        axes.legend(loc='best')
        axes.get_figure().colorbar(label='[p.e.]', ax=axes, mappable=M)

        if save:
            axes.get_figure().savefig('event/' + ids +
                                      '_waveform.png')
            plt.close()
        return None if save else axes


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
