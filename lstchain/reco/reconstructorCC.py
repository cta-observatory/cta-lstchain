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

    Decorator @cc.export take the name to be used for the compiled function and the function signature.
    Meaning of the symbols is defined here https://numba.pydata.org/numba-doc/dev/reference/types.html#numba-types

    """

    cc = CC('log_pdf_CC')
    cc.verbose = True

    @cc.export('log_pdf_ll', 'f8(f8[:],f4[:,:],f4[:],f8[:],f8[:],f8[:,:],i8[:],i8,i8,f8[:,:])')
    def log_pdf_ll(mu, waveform, error, crosstalk, sig_s, templates, factorial, kmin, kmax, weight):
        """
        Performs the sum log likelihood for low luminosity pixels in TimeWaveformFitter.log_pdf
        The log likelihood is sum(pixels) sum(times) of the log single sample likelihood.
        The single sample likelihood is the sum(possible number of pe) of a generalised
        Poisson term times a Gaussian term.

        """
        n_pixels, n_samples = waveform.shape
        n_k = kmax - kmin
        sum = 0.0
        for i in range(n_pixels):
            for j in range(n_samples):
                sum_k = 0.0
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
                    sum_k += poisson * gauss
                # Security to deal with negatively rounded values
                if sum_k <= 0:
                    return -np.inf
                # Add the log single sample likelihood to the full log likelihood
                # An optional weight increasing high signal ample importance is supported
                sum += weight[i, j] * np.log(sum_k)
        return sum

    @cc.export('log_pdf_hl', 'f8(f8[:],f4[:,:],f4[:],f8[:],f8[:,:],f8[:,:])')
    def log_pdf_hl(mu, waveform, error, crosstalk, templates, weight):
        """
        Performs the sum log likelihood for high luminosity pixels in TimeWaveformFitter.log_pdf
        The log likelihood is sum(pixels) sum(times) of the log single sample likelihood.
        The single sample likelihood is a Gaussian term.

        """
        n_pixels, n_samples = waveform.shape
        sum = 0
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
                sum += weight[i, j] * log_gauss
        return sum

    @cc.export('asygaussian2d', 'f8[:](f8[:],f8[:],f8[:],f8,f8,f8,f8,f8,f8)')
    def asygaussian2d(size, x, y, x_cm, y_cm, width, length, psi, rl):
        """
        Evaluate the bi-dimensional gaussian law with asymmetry along the
        main axis.

        Parameters
        ----------
        size: float
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
        pdf: float or array-like
            Evaluation of the 2D gaussian law at (x,y)

        """
        gauss2d = np.empty(len(x), dtype=np.float64)
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
