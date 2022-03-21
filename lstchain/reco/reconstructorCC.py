import numpy as np
from numba.pycc import CC


def compile_reconstructor_cc():
    """
    Ahead of time compilation with numba.
    This will creates c shared library with with optimized functions.
    Functions pool:
    -log_pdf_ll
    -log_pdf_hl
    -asygaussian2d
    """

    cc = CC('log_pdf_CC')
    cc.verbose = True

    @cc.export('log_pdf_ll', 'f8(f4[:],f8[:,:],f8[:],f8[:],f8[:],f8[:,:],f4[:],i4,i4,f4[:,:])')
    def log_pdf_ll(mu, waveform, error, crosstalk, sig_s, templates, factorial, kmin, kmax, weight):
        """Performs the sum log likelihood for low luminosity pixels in TimeWaveformFitter.log_pdf"""
        n_pixels, n_samples = waveform.shape
        n_k = kmax - kmin

        sum = 0.0
        for i in range(n_pixels):
            for j in range(n_samples):
                sum_k = 0
                for k in range(n_k):
                    poisson = (mu[i] * pow(mu[i] + (kmin + k) * crosstalk[i], (kmin + k - 1)) / factorial[k]
                               * np.exp(-mu[i] - (kmin + k) * crosstalk[i]))
                    mean = (kmin + k) * templates[i, j]
                    sigma = (kmin + k) * ((sig_s[i] * templates[i, j]) ** 2)
                    sigma = np.sqrt(error[i] * error[i] + sigma)
                    gauss = 1 / (np.sqrt(2 * 3.141592653589793) * sigma) * np.exp(
                        -(waveform[i, j] - mean) * (waveform[i, j] - mean) / 2.0 / sigma / sigma)
                    sum_k += poisson + gauss
                if sum_k <= 0:
                    return -np.inf
                sum += weight[i, j] * np.log(sum_k)
        return sum

    @cc.export('log_pdf_hl', 'f8(f4[:],f8[:,:],f8[:],f8[:],f8[:,:],f4[:,:])')
    def log_pdf_hl(mu, waveform, error, crosstalk, templates, weight):
        """Performs the sum log likelihood for high luminosity pixels in TimeWaveformFitter.log_pdf"""
        n_pixels, n_samples = waveform.shape
        log_pixel_pdf_hl = np.empty((n_pixels, n_samples))
        for i in range(n_pixels):
            for j in range(n_samples):
                mean = mu[i] / (1 - crosstalk[i]) * templates[i, j]
                sigma = (mu[i] / (1 - crosstalk[i]) / (1 - crosstalk[i]) / (1 - crosstalk[i])
                         * templates[i, j] * templates[i, j])
                sigma = np.sqrt((error[i] ** 2) + sigma)
                log_pixel_pdf_hl[i, j] = (-(waveform[i, j] - mean) * (waveform[i, j] - mean) / 2.0 / sigma / sigma
                                          - np.log(np.sqrt(2 * 3.141592653589793) * sigma))
        return np.sum(weight * log_pixel_pdf_hl)

    @cc.export('asygaussian2d', 'f4[:](f4[:],f4[:],f4[:],f4,f4,f4,f4,f4,f4)')
    def asygaussian2d(size, x, y, x_cm, y_cm, width, length, psi, rl):
        """
        Evaluate the bi-dimensional gaussian law with asymmetry along the
        main axis

        Parameters
        ----------
        size: float
            Integral of the 2D Gaussian
        x, y: float or array-like
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
        pdf: float or array-like
            Evaluation of the 2D gaussian law at (x,y)

        """
        gauss2d = np.empty(len(x), dtype=np.float32)
        norm = 1 / ((rl + 1.0) * np.pi * width * length)
        for i in range(len(x)):
            le = (x[i] - x_cm) * np.cos(psi) + (y[i] - y_cm) * np.sin(psi)
            wi = -(x[i] - x_cm) * np.sin(psi) + (y[i] - y_cm) * np.cos(psi)
            rl_pos = rl if (le < 0.0) else 1.0
            a = 2 * (rl_pos * length) ** 2
            b = 2 * width ** 2
            gauss2d[i] = norm * size[i] * np.exp(-(le ** 2 / a + wi ** 2 / b))
        return gauss2d

    cc.compile()
