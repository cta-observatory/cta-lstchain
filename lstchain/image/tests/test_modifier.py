import pytest
import numpy as np
from scipy.sparse import csr_matrix


@pytest.mark.parametrize('fraction', [0.2, 0.5])
def test_psf_smearer(fraction):
    from lstchain.image.modifier import random_psf_smearer, set_numba_seed

    set_numba_seed(0)

    # simple toy example with a single 7 pixel hexgroup
    image = np.zeros(7)

    # really large number so we can make tight checks on the distributed photon counts
    # 0 is the central, then clockwise
    image[0] = 1e4

    neighbor_matrix = csr_matrix(np.array([
       #0  1  2  3  4  5  6
       [0, 1, 1, 1, 1, 1, 1],  # 0
       [1, 0, 1, 0, 0, 0, 1],  # 1
       [1, 1, 0, 1, 0, 0, 0],  # 2
       [1, 0, 1, 0, 1, 0, 0],  # 3
       [1, 0, 0, 1, 0, 1, 0],  # 4
       [1, 0, 0, 0, 1, 0, 1],  # 5
       [1, 1, 0, 0, 0, 1, 0],  # 6
    ]))

    smeared = random_psf_smearer(image, fraction, neighbor_matrix.indices, neighbor_matrix.indptr)
    # no charge lost in this case
    assert image.sum() == smeared.sum()

    # test charge is distributed into neighbors
    assert np.isclose(smeared[0], (1 - fraction) * image[0], rtol=0.01)
    assert np.allclose(smeared[1:], image[0] * fraction / 6, rtol=0.1)

    # test not all pixels got the same charge
    # (could happen by chance, but *very* unlikely at 10000 photons)
    assert np.any(smeared[1:] != smeared[1])

    # if we put charge in the edges, we should loose some
    image = np.full(7, 1e4)
    smeared = random_psf_smearer(image, fraction, neighbor_matrix.indices, neighbor_matrix.indptr)
    assert smeared.sum() < image.sum()

    # central pixel should roughly stay the same
    assert np.isclose(image[0], smeared[0], rtol=0.01)

    # neighbors should loose 3/6 fractions of the charge
    assert np.allclose((1 - 0.5 * fraction) * image[1:], smeared[1:], rtol=0.05)


def test_calculate_noise_parameters(mc_gamma_testfile, observed_dl1_files):
    from lstchain.image.modifier import calculate_noise_parameters
    [extra_noise_in_dim_pixels,
     extra_bias_in_dim_pixels,
     extra_noise_in_bright_pixels] = calculate_noise_parameters(
        mc_gamma_testfile,
        observed_dl1_files["dl1_file1"]
    )
    assert np.isclose(extra_noise_in_dim_pixels, 1.96, atol=0.01)
    assert np.isclose(extra_bias_in_dim_pixels, 0.0, atol=0.1)
    assert extra_noise_in_bright_pixels == 0.0


def test_calculate_required_additional_nsb(mc_gamma_testfile, observed_dl1_files):
    from lstchain.image.modifier import calculate_required_additional_nsb
    extra_nsb, data_ped_variance, mc_ped_variance = calculate_required_additional_nsb(
        mc_gamma_testfile,
        observed_dl1_files["dl1_file1"]
    )
    # Mean pixel charge variance in pedestals is 0 because there is only one
    # pedestal event in the test file
    assert np.isclose(data_ped_variance, 0.0, atol=0.1)
    assert np.isclose(mc_ped_variance, 2.3, atol=0.2)
    assert np.isclose(extra_nsb, 0.08, atol=0.02)


def test_tune_nsb_on_waveform():
    import astropy.units as u
    from ctapipe_io_lst import LSTEventSource
    from scipy.interpolate import interp1d
    from lstchain.io.io import get_resource_path
    from lstchain.image.modifier import WaveformNsbTuner
    from lstchain.data.normalised_pulse_template import NormalizedPulseTemplate

    waveform = np.zeros((2, 2, 22))
    added_nsb_rate = {1: 0.1 * u.GHz}
    subarray = LSTEventSource.create_subarray(1)
    amplitude_HG = np.zeros(40)
    amplitude_LG = np.zeros(40)
    amplitude_HG[19] = 0.25
    amplitude_HG[20] = 1.0
    amplitude_HG[21] = 0.25
    amplitude_LG[19] = 0.4
    amplitude_LG[20] = 1.0
    amplitude_LG[21] = 0.4
    time = np.linspace(-10, 30, 40)
    pulse_templates = {1: NormalizedPulseTemplate(amplitude_HG, amplitude_LG, time, amplitude_HG_err=None,
                                                  amplitude_LG_err=None)}
    gain = np.array([1, 0])
    spe = np.loadtxt(get_resource_path("data/SinglePhE_ResponseInPhE_expo2Gaus.dat")).T
    spe_integral = np.cumsum(spe[1])
    charge_spe_cumulative_pdf = interp1d(spe_integral, spe[0], kind='cubic',
                                         bounds_error=False, fill_value=0.,
                                         assume_sorted=True)
    nsb_tuner = WaveformNsbTuner(added_nsb_rate,
                                 pulse_templates,
                                 charge_spe_cumulative_pdf,
                                 pre_computed_multiplicity=0)
    nsb_tuner.tune_nsb_on_waveform(waveform, 1, gain, subarray)
    assert np.any(waveform != 0)
    assert np.isclose(np.mean(waveform), 0.0, atol=0.2)
    
    waveform = np.zeros((2, 2, 22))
    nsb_tuner = WaveformNsbTuner(added_nsb_rate,
                                 pulse_templates,
                                 charge_spe_cumulative_pdf,
                                 pre_computed_multiplicity=10)
    nsb_tuner.tune_nsb_on_waveform(waveform, 1, gain, subarray)
    assert np.any(waveform != 0)
    assert np.isclose(np.mean(waveform), 0.0, atol=0.2)
