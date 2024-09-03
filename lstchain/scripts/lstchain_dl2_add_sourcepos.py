#!/usr/bin/env python

import pandas as pd
import numpy as np
import astropy.units as u
import argparse
from astropy.coordinates import SkyCoord, AltAz
from astropy.coordinates.erfa_astrom import ErfaAstromInterpolator, erfa_astrom
from astropy.time import Time
from ctapipe.coordinates import CameraFrame
from ctapipe_io_lst import LSTEventSource
from ctapipe_io_lst.constants import LST1_LOCATION

parser = argparse.ArgumentParser(description="Source position adder. Calculates x,y coordinates of a source in CameraFrame, event-wise.")
parser.add_argument('-f', '--dl2-file', dest='file', required=True,
                    type=str, help='DL2 file to be processed')
parser.add_argument('-s', '--source-name', dest='source_name',
                    type=str, default='Crab pulsar',
                    help='Source name (known to astropy; default: %(default)s)')

# NOTE: the position "Crab pulsar" is better defined than "Crab" or
# "Crab Nebula", for which astropy.coordinates.SkyCoord.form_name returns
# different values depending on the catalog it uses...


def main():
    args = parser.parse_args()

    source_pos = SkyCoord.from_name(args.source_name)
    tablename = "/dl2/event/telescope/parameters/LST_LSTCam"
    new_table_name = 'source_position'

    # Effective focal length, i.e. including the effect of aberration
    # (which affects all image parameters and hence the reconstructed source
    # position in CameraFrame):
    subarray_info = LSTEventSource.create_subarray(tel_id=1)
    focal = subarray_info.tel[1].optics.effective_focal_length
    # By using this effective focal the calculated nominal source position
    # will be consistent with the reconstructed event directions in CameraFrame

    print('Selected source name:', args.source_name)
    print('focal length for calculation:', focal)
    print('Processing', args.file, '...')

    table = pd.read_hdf(args.file, tablename)

    pointing_alt = np.array(table['alt_tel']) * u.rad
    pointing_az  = np.array(table['az_tel']) * u.rad

    time_utc = Time(table["dragon_time"], format="unix", scale="utc")
    telescope_pointing = SkyCoord(alt=pointing_alt, az=pointing_az,
                                  frame=AltAz(obstime=time_utc,
                                              location=LST1_LOCATION))

    # CameraFrame is terribly slow without the erfa interpolator below...
    with erfa_astrom.set(ErfaAstromInterpolator(5 * u.min)):
        camera_frame = CameraFrame(focal_length=focal,
                                   telescope_pointing=telescope_pointing,
                                   location=LST1_LOCATION, obstime=time_utc)
        source_pos_camera = source_pos.transform_to(camera_frame)

    # Units: m (like reco_src_x, y)
    table['src_x'] = source_pos_camera.data.x.to_value(u.m)
    table['src_y'] = source_pos_camera.data.y.to_value(u.m)

    table[['src_x', 'src_y']].to_hdf(args.file, new_table_name, mode='r+',
                                     format='table', data_columns=True)
    print('... done!')

if __name__ == '__main__':
    main()
