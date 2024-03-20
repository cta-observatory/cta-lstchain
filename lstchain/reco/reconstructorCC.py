"""
Numba jitted functions for the waveform likelihood reconstructor
"""

import numpy as np
from numba import njit


@njit(cache=True)
def log_pdf_ll(mu, waveform, error, crosstalk, sig_s, templates, factorial, n_peaks):
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
    factorial: unsigned int64 1D array
        Pre-computed table of factorials
    n_peaks: int64
        Size of the factorial array and possible number of photo-electron
        in a pixel with relevant Poisson probability

    Returns
    ----------
    sumlh : float64
        Sum log likelihood

    """
    n_pixels, n_samples = waveform.shape
    sumlh = 0.0
    for i in range(n_pixels):
        poisson = np.empty((n_pixels, n_peaks))
        for k in range(n_peaks):
            # Generalised Poisson term
            poisson[i, k] = (mu[i] * pow(mu[i] + k * crosstalk[i], (k - 1)) / factorial[k]
                             * np.exp(-mu[i] - k * crosstalk[i]))
        for j in range(n_samples):
            sumlh_k = 0.0
            for k in range(n_peaks):
                # Gaussian term
                mean = k * templates[i, j]
                sigma = k * ((sig_s[i] * templates[i, j]) ** 2)
                sigma = np.sqrt(error[i] * error[i] + sigma)
                gauss = 1 / (np.sqrt(2 * np.pi) * sigma) * np.exp(
                    -(waveform[i, j] - mean) * (waveform[i, j] - mean) / 2.0 / sigma / sigma)
                # Add the contribution for the k number of p.e. to the single sample likelihood
                sumlh_k += poisson[i, k] * gauss
            # Security to deal with negatively rounded values
            if sumlh_k <= 0:
                return -np.inf
            # Add the log single sample likelihood to the full log likelihood
            sumlh += np.log(sumlh_k)
    return sumlh

@njit(cache=True)
def log_pdf_hl(mu, waveform, error, crosstalk, templates):
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
            sumlh += log_gauss
    return sumlh


@njit(cache=True)
def asygaussian2d(size, x, y, x_cm, y_cm, width, length, psi, rl):
    """
    Evaluate the bi-dimensional gaussian law with asymmetry along the
    main axis.

    Parameters
    ----------
    size: float64 1D array
        Integral of the 2D Gaussian
    x, y: float64 1D array
        Position at which the log gaussian is evaluated
    x_cm, y_cm: float64
        Center of the 2D Gaussian
    width, length: float64
        Standard deviations of the 2 dimensions of the 2D Gaussian law
    psi: float64
        Orientation of the 2D Gaussian
    rl: float64
        Asymmetry factor between the two lengths

    Returns
    -------
    gauss2d: float64 1D array
        Evaluation of the 2D gaussian law at (x,y)

    """
    gauss2d = np.empty(len(x), dtype=np.float64)
    norm = 1 / ((rl + 1.0) * np.pi * width * length)
    for i in range(len(x)):
        # Compute the x and y coordinates projection in the 2D gaussian length and width coordinates
        le = (x[i] - x_cm) * np.cos(psi) + (y[i] - y_cm) * np.sin(psi)
        wi = -(x[i] - x_cm) * np.sin(psi) + (y[i] - y_cm) * np.cos(psi)
        # Check which side of the maximum the current point is for asymmetry purpose
        rl_pos = rl if (le < 0.0) else 1.0
        a = 2 * (rl_pos * length) ** 2
        b = 2 * width ** 2
        # Evaluate the 2D gaussian term
        gauss2d[i] = norm * size[i] * np.exp(-(le ** 2 / a + wi ** 2 / b))
    return gauss2d


@njit(cache=True)
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


@njit(cache=True)
def template_interpolation(gain, times, t0, dt, a_hg, a_lg):
    """
    Fast template interpolator using uniformly sampled base with known origin and step.
    The algorithm finds the indexes between which the template is needed and performs a linear interpolation.

    Parameters
    ----------
    gain: boolean 1D array
        Gain channel used per pixel
    times: float64 2D array
        Times of each waveform samples
    t0: float64
        Time of the first value of the pulse templates
    dt: float 64
        Time step between templates values
    a_hg: float64 1D array
        Template values for the high gain channel
    a_lg: float64 1D array
        Template values for the low gain channel

    Returns
    -------
    out: float64 2D array
        Pulse template gain selected and interpolated at each sample times

    """
    n, m = times.shape
    size = a_hg.shape[0]
    out = np.empty((n, m))
    for i in range(n):
        for j in range(m):
            # Find the index before the requested time
            a = (times[i, j]-t0)/dt
            t = int(a)
            if 0 < t+1 < size:
                # Select the gain and interpolate the pulse template at the requested time
                out[i, j] = a_hg[t] * (1. - a + t) + a_hg[t+1] * (a-t) if gain[i] else \
                    a_lg[t] * (1. - a + t) + a_lg[t+1] * (a-t)
            else:
                # Assume 0 if outside the recorded range
                out[i, j] = 0.0
    return out


@njit(cache=True)
def nsb_only_waveforms(time, is_high_gain, additional_nsb, amplitude, t_0,
                        t0_template, dt_template, a_hg_template, a_lg_template):
    """
    Generate waveforms of pure NSB. NSB photons injected through a fast interpolator using a
    uniformly sampled normalised template with known time of first value and time step.
    The method requires as input the number of NSB events to inject per pixel,
    the times of injection and the events normalisations.

    Interpolation code duplicated from function template_interpolation.

    Parameters
    ----------
    time: float64 1D array
        Times of each waveform samples
    is_high_gain: boolean 1D array
        Gain channel used per pixel: True=hg, False=lg
    additional_nsb: float64 1D array
        Number of NSB photons to inject per pixel
    amplitude: float 2D array
        Normalisation factor to apply to the template per photon in each pixel
    t_0: float 2D array
        Shift in the origin of time per photon in each pixel
    t0_template: float64
        Time of the first value of the pulse template
    dt_template: float 64
        Time step between template values
    a_hg_template: float64 1D array
        Template values for the high gain channel
    a_lg_template: float64 1D array
        Template values for the low gain channel

    Returns
    -------
    nsb_waveform: float64 2D array
        Charge (p.e. / ns) in each pixel and sampled time of the injected NSB photons
    """
    n_pixels = additional_nsb.shape[0]
    m = time.shape[0]
    nsb_waveform = np.zeros((n_pixels, m), dtype=np.float64)
    size = a_hg_template.shape[0]
    single_spe_waveform = np.empty(m)

    times = time / dt_template
    t0 = (t_0 + t0_template) / dt_template
    for i in range(n_pixels):
        for j in range(additional_nsb[i]):
            # Find the index before the requested time
            a = times - t0[i, j]

            for k in range(m):
                t = int(a[k])

                if 0 < t + 1 < size:
                    # Select the gain and interpolate the pulse template at the requested time
                    single_spe_waveform[k] = a_hg_template[t] * (1. - a[k] + t) + a_hg_template[t + 1] * (a[k] - t) if is_high_gain[i] else\
                        a_lg_template[t] * (1. - a[k] + t) + a_lg_template[t + 1] * (a[k] - t)
                else:
                    # Assume 0 if outside the recorded range
                    single_spe_waveform[k] = 0.0

            nsb_waveform[i] += amplitude[i, j] * single_spe_waveform

    return nsb_waveform


@njit(cache=True)
def log_pdf(charge, t_cm, x_cm, y_cm, length, wl, psi, v, rl,
            data, error, is_high_gain, sig_s, crosstalks, times, time_shift,
            p_x, p_y, pix_area,  template_dt, template_t0, template_lg,
            template_hg, n_peaks, transition_charge, factorial):
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
    transition_charge: float32
        Model charge above which the Gaussian approximation (log_pdf_hl) is used
    factorial: unsigned int64
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
    templates = template_interpolation(is_high_gain, t, template_t0, template_dt,
                                       template_hg, template_lg)
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

    # We split pixels between high and low luminosity pixels.
    # The transition is dependent on the camera crosstalk and maximum
    # number of p.e. allowed in the config (n_peak) for the sum in the
    # faint pixels likelihood computation

    mask_LL = (mu <= transition_charge) & (mu > 0)
    mask_HL = ~mask_LL

    log_pdf_faint = log_pdf_ll(mu[mask_LL],
                               data[mask_LL],
                               error[mask_LL],
                               crosstalks[mask_LL],
                               sig_s[mask_LL],
                               templates[mask_LL],
                               factorial,
                               n_peaks)

    log_pdf_bright = log_pdf_hl(mu[mask_HL],
                                data[mask_HL],
                                error[mask_HL],
                                crosstalks[mask_HL],
                                templates[mask_HL])

    log_lh = (log_pdf_faint + log_pdf_bright) / (data.shape[0] * data.shape[1])

    return log_lh
