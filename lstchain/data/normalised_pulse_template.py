import numpy as np
from scipy.interpolate import interp1d


class NormalizedPulseTemplate:
    """
    Class for handling the template for the pulsed response of the pixels
    of the camera to a single photo-electron in high and low gain.

    """

    def __init__(self, amplitude_HG, amplitude_LG, time, amplitude_HG_err=None,
                 amplitude_LG_err=None):
        """
        Save the pulse template and optional error
        and create an interpolation.

        Parameters
        ----------
        amplitude_HG/LG: array
            Amplitude of the signal produced in a pixel by a photo-electron
            in high gain (HG) and low gain (LG) for successive time samples
        time: array
            Times of the samples
        amplitude_HG/LG_err: array
            Error on the pulse template amplitude

        """

        self.time = np.array(time)
        self.amplitude_HG = np.array(amplitude_HG)
        self.amplitude_LG = np.array(amplitude_LG)
        if amplitude_HG_err is not None:
            assert np.array(amplitude_HG_err).shape == self.amplitude_HG.shape
            self.amplitude_HG_err = np.array(amplitude_HG_err)
        else:
            self.amplitude_HG_err = np.zeros(self.amplitude_HG.shape)
        if amplitude_LG_err is not None:
            assert np.array(amplitude_LG_err).shape == self.amplitude_LG.shape
            self.amplitude_LG_err = np.array(amplitude_LG_err)
        else:
            self.amplitude_LG_err = self.amplitude_LG * 0
        self._template = self._interpolate()
        self._template_err = self._interpolate_err()

    def __call__(self, time, gain, amplitude=1, t_0=0, baseline=0):
        """
        Use the interpolated template to access the value of the pulse at
        time = time in gain regime = gain. Additionally, an alternative
        normalisation, origin of time and baseline can be used.

        Parameters
        ----------
        time: float array
            Time after the origin to estimate the value of the pulse
        gain: string array
            Identifier of the gain channel used for each pixel
            Either "HG" or "LG"
        amplitude: float
            Normalisation factor to apply to the template
        t_0: float
            Shift in the origin of time
        baseline: float array
            Baseline to be subtracted for each pixel

        Return
        ----------
        y: array
            Value of the template in each pixel at the requested times

        """

        y = amplitude * self._template[gain](time - t_0) + baseline
        return np.array(y)

    def get_error(self, time, gain, amplitude=1, t_0=0):
        """
        Use the interpolated error on the template to access the value
        of the pulse at time = time in gain regime = gain.
        Additionally, an alternative normalisation and origin of time
        can be used.

        Parameters
        ----------
        time: float array
            Time after the origin to estimate the value of the error
        gain: string array
            Identifier of the gain channel used for each pixel
            Either "HG" or "LG"
        amplitude: float
            Normalisation factor to apply to the error
        t_0: float
            Shift in the origin of time

        Return
        ----------
        y: array
            Value of the template in each pixel at the requested times

        """

        y = amplitude * self._template_err[gain](time - t_0)
        return np.array(y)

    def save(self, filename):
        """
        Save a loaded template to a text file.

        Parameters
        ----------
        filename: string
            Location of the output text file

        """

        data = np.vstack([self.time, self.amplitude_HG, self.amplitude_HG_err,
                          self.amplitude_LG, self.amplitude_LG_err])
        np.savetxt(filename, data.T)

    @classmethod
    def load_from_file(cls, filename):
        """
        Load a pulse template from a text file.
        Allows for only one gain channel and no errors,
        two gain channels and no errors or two gain channels with errors.

        Parameters
        ----------
        cls: This class
        filename: string
            Location of the template file

        Return
        ----------
        cls(): Instance of NormalizedPulseTemplate receiving the information
               from the input file

        """

        data = np.loadtxt(filename).T
        assert len(data) in [2, 3, 5]
        if len(data) == 2:  # one shape in file
            t, x = data
            return cls(amplitude_HG=x, amplitude_LG=x, time=t)
        if len(data) == 3:  # no error in file
            t, hg, lg = data
            return cls(amplitude_HG=hg, amplitude_LG=lg, time=t)
        elif len(data) == 5:  # two gains and errors
            t, hg, lg, dhg, dlg = data
            return cls(amplitude_HG=hg, amplitude_LG=lg, time=t,
                       amplitude_HG_err=dhg, amplitude_LG_err=dlg)

    @classmethod
    def load_from_eventsource(cls, eventsource_camera_readout):
        """
        Load a pulse template from an event source camera readout.
        Read the sampling rate to create a time variable reaching
        9 ns at the HG maximum

        Parameters
        ----------
        cls: This class
        eventsource_camera_readout: `CameraReadout`
            CameraReadout object obtained from the LST event source

        Return
        ----------
        cls(): Instance of NormalizedPulseTemplate receiving the information
               from the input file

        """

        t = eventsource_camera_readout.reference_pulse_sample_time.to_value('ns')
        hg, lg = eventsource_camera_readout.reference_pulse_shape
        i = np.argmax(hg)
        t = t - t[i] + 9.0
        return cls(amplitude_HG=hg, amplitude_LG=lg, time=t)

    @staticmethod
    def _normalize(time, amplitude, error):
        """
        Normalize the pulse template in p.e/ns.

        """

        normalization = np.sum(amplitude) * (np.max(time) - np.min(time)) / (len(time)-1)
        return amplitude / normalization, error / normalization

    def _interpolate(self):
        """
        Creates a normalised interpolation of the pulse template from a
        discrete and non-normalised input. Also normalises the error.

        Return
        ----------
        A dictionary containing a 1d cubic interpolation of the normalised
        amplitude of the template versus time,
        for the high and low gain channels.

        """

        self.amplitude_HG, self.amplitude_HG_err = self._normalize(self.time,
                                                                   self.amplitude_HG,
                                                                   self.amplitude_HG_err)
        self.amplitude_LG, self.amplitude_LG_err = self._normalize(self.time,
                                                                   self.amplitude_LG,
                                                                   self.amplitude_LG_err)
        return {"HG": interp1d(self.time, self.amplitude_HG, kind='cubic',
                               bounds_error=False, fill_value=0.,
                               assume_sorted=True),
                "LG": interp1d(self.time, self.amplitude_LG, kind='cubic',
                               bounds_error=False, fill_value=0.,
                               assume_sorted=True)}

    def _interpolate_err(self):
        """
        Creates an interpolation of the error on the pulse template
        from a discrete and normalised input.

        Return
        ----------
        A dictionary containing a 1d cubic interpolation of the error on the
        normalised amplitude of the template versus time,
        for the high and low gain channels.

        """

        return {"HG": interp1d(self.time, self.amplitude_HG_err, kind='cubic',
                               bounds_error=False, fill_value=np.inf,
                               assume_sorted=True),
                "LG": interp1d(self.time, self.amplitude_LG_err, kind='cubic',
                               bounds_error=False, fill_value=np.inf,
                               assume_sorted=True)}

    def compute_time_of_max(self):
        """
        Find the average of the times of maximum
        of the high and low gain pulse shapes.

        Returns
        -------
        t_max: float
            Time of maximum of the pulse shapes (averaged)

        """

        t_max = (self.time[np.argmax(self.amplitude_HG)] +
                 self.time[np.argmax(self.amplitude_LG)]) / 2
        return t_max
