from lstchain.reco import utils
import astropy.units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord
from astropy.table import QTable
import numpy as np
import pandas as pd


def test_camera_to_altaz():
    pos_x = np.array([0, 0]) * u.m
    pos_y = np.array([0, 0]) * u.m
    focal = 28 * u.m
    pointing_alt = np.array([1.0, 1.0]) * u.rad
    pointing_az = np.array([0.2, 0.5]) * u.rad
    sky_coords = utils.camera_to_altaz(pos_x, pos_y, focal, pointing_alt, pointing_az)
    np.testing.assert_allclose(sky_coords.alt, pointing_alt, rtol=1e-4)
    np.testing.assert_allclose(sky_coords.az, pointing_az, rtol=1e-4)

    # Test for real event with a time
    obs_time = Time(['2018-11-01T02:00', '2018-11-01T02:00'])
    sky_coords = utils.camera_to_altaz(pos_x, pos_y, focal, pointing_alt, pointing_az, obstime = obs_time)
    np.testing.assert_allclose(sky_coords.alt, pointing_alt, rtol=1e-4)
    np.testing.assert_allclose(sky_coords.az, pointing_az, rtol=1e-4)

def test_radec_to_camera():
    pointing_radec = SkyCoord.from_name('Crab')
    obstime = Time('2020-01-27T23:00', scale='utc')
    pointing_alt  = u.Quantity(1.3748, u.rad, copy=False)
    pointing_az = u.Quantity(4.0975, u.rad, copy=False)
    focal = 28*u.m
    expected_source_pos_camera = np.array([0.0, 0.0]) * u.m
    pointing_pos_camera = utils.radec_to_camera(pointing_radec, obstime, pointing_alt, pointing_az, focal)
    np.testing.assert_allclose(pointing_pos_camera.x.to_value(), expected_source_pos_camera[0].to_value(), atol=0.1)
    np.testing.assert_allclose(pointing_pos_camera.y.to_value(), expected_source_pos_camera[1].to_value(), atol=0.1)

def test_reco_source_position_sky():
    cog_x = np.array([2, 1]) * u.m
    cog_y = np.array([-1, 1]) * u.m
    disp_dx = np.array([-2, -1]) * u.m
    disp_dy = np.array([1, -1]) * u.m
    focal_length = 28 *u.m
    pointing_alt = np.array([1.0, 1.0]) * u.rad
    pointing_az = np.array([0.2, 0.5]) * u.rad
    sky_coords = utils.reco_source_position_sky(cog_x, cog_y, disp_dx, disp_dy, focal_length, pointing_alt, pointing_az)
    np.testing.assert_allclose(sky_coords.alt, pointing_alt, rtol=1e-4)
    np.testing.assert_allclose(sky_coords.az, pointing_az, rtol=1e-4)


def test_sky_to_camera():
    alt = np.array([1, 1]) * u.rad
    az = np.array([0.2, 0.5]) * u.rad
    focal = 28*u.m
    pointing_alt = np.array([1.0, 1.0]) * u.rad
    pointing_az = np.array([0.2, 0.5]) * u.rad
    camera_coords = utils.sky_to_camera(alt, az, focal, pointing_alt, pointing_az)
    np.testing.assert_allclose(camera_coords.x.value, np.array([0, 0]), rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(camera_coords.y.value, np.array([0, 0]), rtol=1e-4, atol=1e-4)


def test_linear_imputer():
    a = np.array([0.2, 0.3, np.nan, np.nan, np.nan, 0.7, np.nan, 0.8, np.nan])
    utils.linear_imputer(a, missing_values=np.nan, copy=False)
    np.testing.assert_allclose(a, [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.8])


def test_impute_pointing():
    event_id = np.arange(5, 14, dtype=int)
    a = np.array([0.2, 0.3, np.nan, np.nan, np.nan, 0.7, np.nan, 0.8, np.nan])
    df = pd.DataFrame(np.transpose([event_id, a, a]), columns=['event_id', 'alt_tel', 'az_tel'])
    df = utils.impute_pointing(df)
    np.testing.assert_allclose(df.alt_tel, [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.8])
    np.testing.assert_allclose(df.az_tel, [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.8])


def test_filter_events():
    from lstchain.reco.utils import filter_events
    df = pd.DataFrame({'a': [1, 2, 3],
                       'b': [np.nan, 2.2, 3.2],
                       'c': [1, 2, np.inf]}
                      )
    np.testing.assert_array_equal(filter_events(df, filters=dict(a=[0, np.inf], b=[0, np.inf], c=[0, np.inf]), finite_params=['b']),
                                  pd.DataFrame({'a': [2, 3], 'b': [2.2, 3.2], 'c': [2, np.inf]}))
    np.testing.assert_array_equal(filter_events(df, filters=dict(a=[0, np.inf], b=[0, np.inf], c=[0, np.inf]), finite_params=['b', 'c']),
                                  pd.DataFrame({'a': [2], 'b': [2.2], 'c': [2]}))
    np.testing.assert_array_equal(filter_events(df, filters=dict(a=[0, 1])),
                                  pd.DataFrame({'a': [1], 'b': [np.nan], 'c': 1}))
    with np.testing.assert_raises(KeyError):
        filter_events(df, filters=dict(e=[0, np.inf]))

def test_get_obstime_real():
    # times in seconds, rates in s^-1
    t_obs = 600
    dead_time_per_event = 7e-6
    cosmics_rate = 1e4
    # interleaved event rates:
    pedestal_rate = 100
    flatfield_rate = 100
    # starting times for interleaved events (arbitrary):
    t0_pedestal = 0
    t0_flatfield = 0.002
    n_cosmics = np.random.poisson(cosmics_rate * t_obs)

    timestamps = np.random.uniform(0, t_obs, n_cosmics)
    timestamps = np.append(timestamps, np.arange(t0_pedestal, t_obs,
                                                 1/pedestal_rate))
    timestamps = np.append(timestamps, np.arange(t0_flatfield, t_obs,
                                                 1/flatfield_rate))
    # sort events by timestamp:
    timestamps.sort()

    # time to previous event:
    delta_t = np.insert(np.diff(timestamps), 0, 0)

    # now remove events which are closer than dead_time_per_event
    recorded_events = delta_t > dead_time_per_event

    # true effective time:
    true_t_eff = t_obs - dead_time_per_event * recorded_events.sum()
    true_t_eff *= u.s

    # we'll write only 80% of the remaining events - this simulates triggered
    # events which are no longer present in the DL2 event list
    cut = np.random.uniform(0., 1., recorded_events.sum()) > 0.2

    events = pd.DataFrame({'delta_t': delta_t[recorded_events][cut],
                           'dragon_time': timestamps[recorded_events][cut]})
    t_eff, t_elapsed = utils.get_effective_time(events)
    print(t_obs, t_elapsed, true_t_eff, t_eff)
    # test accuracy to 0.05%:
    assert np.isclose(t_eff, true_t_eff, rtol=5e-4)

    # now test with a QTable:
    a = delta_t[recorded_events][cut]*u.s
    b = timestamps[recorded_events][cut]*u.s
    events = QTable([a, b], names=('delta_t', 'dragon_time'))
    t_eff, t_elapsed = utils.get_effective_time(events)
    print(t_obs, t_elapsed, true_t_eff, t_eff)
    # test accuracy to 0.05%:
    assert np.isclose(t_eff, true_t_eff, rtol=5e-4)