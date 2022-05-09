import numpy as np
from numba.pycc import CC
from numba import njit


def compile_reconstructor_cc():
    """
    Ahead of time compilation with numba.
    This will creates c shared library with with optimized functions.
    Functions pool:
    -log_pdf_ll
    -log_pdf_hl
    -asygaussian2d
    -linval
    -template_interpolation
    -log_pdf

    Decorator @cc.export take the name to be used for the compiled function and the function signature.
    Meaning of the symbols is defined here https://numba.pydata.org/numba-doc/dev/reference/types.html#numba-types
    @njit decorator makes function available to other function compiled here.

    """

    cc = CC('log_pdf_CC')
    cc.verbose = True

    @njit()
    @cc.export('log_pdf_ll', 'f8(f8[:],f4[:,:],f4[:],f8[:],f8[:],f8[:,:],u8[:],i8,i8,f8[:,:])')
    def log_pdf_ll(mu, waveform, error, crosstalk, sig_s, templates, factorial, kmin, kmax, weight):
        """
        Performs the sum log likelihood for low luminosity pixels in TimeWaveformFitter.
        The log likelihood is sum(pixels) sum(times) of the log single sample likelihood.
        The single sample likelihood is the sum(possible number of pe) of a generalised
        Poisson term times a Gaussian term.

        Parameters:
        ----------
        mu: float64 1D array
            Expected charge per pixel
        waveform: float32 2D array
            Measured signal in p.e. per ns
        error: float32 1D array
            Pedestal standard deviation per pixel
        crosstalk: float64 1D array
            Crosstalk factor for each pixel
        sig_s: float64 1D array
            Single p.e. intensity distribution standard deviation for each pixel
        templates: float64 2D array
            Value of the pulse template evaluated in each pixel at each observed time
        factorial: unsigned int64
            Pre-computed table of factorials
        kmin, kmax: int64
            Range of possible number of photo-electron in a pixel with relevant Poisson probability
        weight : float 64 2D array
            Weight to use in the likelihood for each sample

        Returns
        ----------
        sumlh : float64
            Sum log likelihood

        """
        n_pixels, n_samples = waveform.shape
        n_k = kmax - kmin
        sumlh = 0.0
        for i in range(n_pixels):
            for j in range(n_samples):
                sumlh_k = 0.0
                for k in range(n_k):
                    # Generalised Poisson term
                    poisson = (mu[i] * pow(mu[i] + (kmin + k) * crosstalk[i], (kmin + k - 1)) / factorial[k]
                               * np.exp(-mu[i] - (kmin + k) * crosstalk[i]))
                    # Gaussian term
                    mean = (kmin + k) * templates[i, j]
                    sigma = (kmin + k) * ((sig_s[i] * templates[i, j]) ** 2)
                    sigma = np.sqrt(error[i] * error[i] + sigma)
                    gauss = 1 / (np.sqrt(2 * np.pi) * sigma) * np.exp(
                        -(waveform[i, j] - mean) * (waveform[i, j] - mean) / 2.0 / sigma / sigma)
                    # Add the contribution for the k+kmin number of p.e. to the single sample likelihood
                    sumlh_k += poisson * gauss
                # Security to deal with negatively rounded values
                if sumlh_k <= 0:
                    return -np.inf
                # Add the log single sample likelihood to the full log likelihood
                # An optional weight increasing high signal ample importance is supported
                sumlh += weight[i, j] * np.log(sumlh_k)
        return sumlh

    @njit()
    @cc.export('log_pdf_hl', 'f8(f8[:],f4[:,:],f4[:],f8[:],f8[:,:],f8[:,:])')
    def log_pdf_hl(mu, waveform, error, crosstalk, templates, weight):
        """
        Performs the sum log likelihood for high luminosity pixels in TimeWaveformFitter.log_pdf
        The log likelihood is sum(pixels) sum(times) of the log single sample likelihood.
        The single sample likelihood is a Gaussian term.

        Parameters:
        ----------
        mu: float64 1D array
            Expected charge per pixel
        waveform: float32 2D array
            Measured signal in p.e. per ns
        error: float32 1D array
            Pedestal standard deviation per pixel
        crosstalk: float64 1D array
            Crosstalk factor for each pixel
        templates: float64 2D array
            Value of the pulse template evaluated in each pixel at each observed time
        weight : float 64 2D array
            Weight to use in the likelihood for each sample

        Returns
        ----------
        sumlh : float64
            Sum log likelihood

        """
        n_pixels, n_samples = waveform.shape
        sumlh = 0
        for i in range(n_pixels):
            for j in range(n_samples):
                # Log Gaussian term
                mean = mu[i] / (1 - crosstalk[i]) * templates[i, j]
                sigma = (mu[i] / (1 - crosstalk[i]) / (1 - crosstalk[i]) / (1 - crosstalk[i])
                         * templates[i, j] * templates[i, j])
                sigma = np.sqrt((error[i] ** 2) + sigma)
                log_gauss = (-(waveform[i, j] - mean) * (waveform[i, j] - mean) / 2.0 / sigma / sigma
                             - np.log(np.sqrt(2 * np.pi) * sigma))
                # Add the log single sample likelihood to the full log likelihood
                # An optional weight increasing high signal ample importance is supported
                sumlh += weight[i, j] * log_gauss
        return sumlh

    @njit()
    @cc.export('asygaussian2d', 'f8[:](f8[:],f8[:],f8[:],f8,f8,f8,f8,f8,f8)')
    def asygaussian2d(size, x, y, x_cm, y_cm, width, length, psi, rl):
        """
        Evaluate the bi-dimensional gaussian law with asymmetry along the
        main axis.

        Parameters
        ----------
        size: array-like
            Integral of the 2D Gaussian
        x, y: array-like
            Position at which the log gaussian is evaluated
        x_cm, y_cm: float
            Center of the 2D Gaussian
        width, length: float
            Standard deviations of the 2 dimensions of the 2D Gaussian law
        psi: float
            Orientation of the 2D Gaussian
        rl: float
            Asymmetry factor between the two lengths

        Returns
        -------
        gauss2d: array-like
            Evaluation of the 2D gaussian law at (x,y)

        """
        gauss2d = np.empty(len(x), dtype=np.float64)
        norm = 1 / ((rl + 1.0) * np.pi * width * length)
        for i in range(len(x)):
            # Compute the x and y coordinates projection in the 2D gaussian length and width coordinates
            le = (x[i] - x_cm) * np.cos(psi) + (y[i] - y_cm) * np.sin(psi)
            wi = -(x[i] - x_cm) * np.sin(psi) + (y[i] - y_cm) * np.cos(psi)
            # Check which side of the maximum the current point is for asymetry purpose
            rl_pos = rl if (le < 0.0) else 1.0
            a = 2 * (rl_pos * length) ** 2
            b = 2 * width ** 2
            # Evaluate the 2D gaussian term
            gauss2d[i] = norm * size[i] * np.exp(-(le ** 2 / a + wi ** 2 / b))
        return gauss2d

    @njit()
    @cc.export('linval', 'f8[:](f8,f8,f8[:])')
    def linval(a, b, x):
        """
        Linear law function

        Parameters
        ----------
        a: float64
            Slope
        b: float64
            Intercept
        x: float64 1D array
            Values at which the function is evaluated

        Returns
        -------
        y: float64 1D array
            Linear law evaluated at x

        """
        y = np.empty(x.shape)
        for i in range(len(x)):
            y[i] = b + a * x[i]
        return y

    @njit()
    @cc.export('template_interpolation', 'f8[:,:](b1[:],f8[:,:],f8,f8,f8[:],f8[:],i8)')
    def template_interpolation(gain, times, t0, dt, a_hg, a_lg, size):
        """
        Fast template interpolator using uniformly sampled base with known origin and step.
        The algorithm finds the indexes between which the template is needed and performs a linear interpolation.

        Parameters
        ----------
        gain: boolean 1D array
            Gain channel used per pixel
        times: float64 1D array
            Times of each waveform samples
        t0: float64
            Time of the first value of the pulse templates
        dt: float 64
            Time step between templates values
        a_hg: float64 1D array
            Template values for the high gain channel
        a_lg: float64 1D array
            Template values for the low gain channel
        size: int64
            Number of element in a_hg and a_lg

        Returns
        -------
        out: float64 2D array
            Pulse template gain selected and interpolated at each sample times

        """
        n, m = times.shape
        out = np.empty((n, m))
        for i in range(n):
            for j in range(m):
                # Find the index before the requested time
                a = (times[i, j]-t0)/dt
                t = int(a)
                if a < size:
                    # Select the gain and interpolate the pulse template at the requested time
                    out[i, j] = a_hg[t] * (1. - a + t) + a_hg[t+1] * (a-t) if gain[i] else \
                        a_lg[t] * (1. - a + t) + a_lg[t+1] * (a-t)
                else:
                    # Assume 0 if outside of the recorded range
                    out[i, j] = 0.0
        return out

    @cc.export('log_pdf', 'f8(f8,f8,f8,f8,f8,f8,f8,f8,f8,'
                          'f4[:,:],f4[:],b1[:],f8[:],f8[:],f8[:],f4[:],'
                          'f8[:],f8[:],f8[:],f8,f8,f8[:],'
                          'f8[:],i8,b1,u8[:])')
    def log_pdf(charge, t_cm, x_cm, y_cm, length, wl, psi, v, rl,
                data, error, is_high_gain, sig_s, crosstalks, times, time_shift,
                p_x, p_y, pix_area,  template_dt, template_t0, template_lg,
                template_hg, n_peaks, use_weight, factorial):
        """
        Compute the log likelihood of the model used for a set of input parameters.

        Fitted Parameters
        ----------
        charge: float64
            Charge of the peak of the spatial model
        t_cm: float64
            Time of the middle of the energy deposit in the camera
            for the temporal model
        x_cm, y_cm: float64
            Position of the center of the spatial model
        length, wl: float64
            Spatial dispersion of the model along the main and
            fraction of this dispersion along the minor axis
        psi: float64
            Orientation of the main axis of the spatial model and of the
            propagation of the temporal model
        v: float64
            Velocity of the evolution of the signal over the camera
        rl: float64
            Asymmetry of the spatial model along the main axis

        Other Parameters:
        ----------
        data : float32 2D array
            Waveform
        error : float32 1D array
            Pedestal standard deviation per pixel
        is_high_gain : boolean 1D array
        sig_s :  float64 1D array
            Single p.e. intensity distribution standard deviation for each pixel
        crosstalks: float64 1D array
            Crosstalk factor for each pixel
        times : float64 1D array
            Relative time of successive waveform samples
        time_shift : float64 1D array
            Time shift correction to be applied per pixel
        p_x, p_y, pix_area: float64 1D array
            Pixels position and surface area
        template_dt, template_t0, template_lg, template_hg : float64, float64, float64 1D array, float64 1D array
            Pulse template properties used in the interpolation of the model
        n_peaks: int64
            Maximum number of p.e. term used in the low luminosity likelihood
        use_weight: bool
            If True, the brightest sample are made more important in the likelihood computation
        factorial unsigned int64
            Pre-computed table of factorials

        Returns
        ----------
        log_lh: float64
            Reduced log likelihood of the model
        """
        n_pixels, n_samples = data.shape
        dx = (p_x - x_cm)
        dy = (p_y - y_cm)
        long = dx * np.cos(psi) + dy * np.sin(psi)
        t_model = linval(v, t_cm, long)
        t = np.empty(data.shape, dtype=np.float64)
        for i in range(n_pixels):
            for j in range(n_samples):
                t[i, j] = times[j] - t_model[i] - time_shift[i]
        size_template = template_hg.shape[0]
        templates = template_interpolation(is_high_gain, t, template_t0, template_dt,
                                           template_hg, template_lg, size_template)
        rl = 1 + rl if rl >= 0 else 1 / (1 - rl)
        mu = asygaussian2d(charge * pix_area,
                           p_x,
                           p_y,
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
        # TODO correct for crosstalk>0 or remove since kmax limited to 21 by int precision with factorials
        mask_LL = (mu <= n_peaks / 1.096 - 47.8) & (mu > 0)
        mask_HL = ~mask_LL

        if len(mu[mask_LL]) == 0:
            kmin, kmax = 0, n_peaks
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
        if kmax > n_peaks:
            kmax = n_peaks

        if use_weight:
            weight = 1.0 + (data / np.max(data))
        else:
            weight = np.ones(data.shape)

        mask_k = (np.arange(n_peaks) >= kmin) & (np.arange(n_peaks) < kmax)

        log_pdf_faint = log_pdf_ll(mu[mask_LL],
                                   data[mask_LL],
                                   error[mask_LL],
                                   crosstalks[mask_LL],
                                   sig_s[mask_LL],
                                   templates[mask_LL],
                                   factorial[mask_k],
                                   kmin, kmax,
                                   weight[mask_LL])

        log_pdf_bright = log_pdf_hl(mu[mask_HL],
                                    data[mask_HL],
                                    error[mask_HL],
                                    crosstalks[mask_HL],
                                    templates[mask_HL],
                                    weight[mask_HL])

        log_lh = (log_pdf_faint + log_pdf_bright) / np.sum(weight)

        return log_lh

    cc.compile()
