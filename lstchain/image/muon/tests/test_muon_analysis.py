import astropy.units as u


def test_pixel_to_telescope():
    from ctapipe_io_lst import load_camera_geometry
    from lstchain.image.muon import pixel_coords_to_telescope

    cam = load_camera_geometry()
    x, y = pixel_coords_to_telescope(cam, 38 * u.m)

    assert x.unit == y.unit == u.deg
