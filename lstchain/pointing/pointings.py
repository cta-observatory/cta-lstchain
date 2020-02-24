import os
import numpy as np
from ctapipe.core import Component
from ctapipe_io_lst.containers import LSTMonitoringContainer, LSTDriveContainer
from ctapipe.core.traits import Unicode, Int
from astropy.io import ascii
from astropy import units as u


__all__ = [
    'PointingPosition'
]


class PointingPosition(Component):
    """
      Pointion position of telescopes

    """
    drive_path = Unicode(
        '',
        allow_none=True,
        help='Path to the LST drive report file'
    ).tag(config=True)

    tel_id = Int(
        0,
        help='id of the telescope to take drive report for'
    ).tag(config=True)

    def _read_drive_report(self):
        """
        Reading drive reports

        Parameters:
        -----------
        str: drive report file

        Returns:
        data:`~astropy.table.Table`
             A table of drive reports

        """
        self.log.info("Drive report file:", self.drive_path)
        if self.drive_path:
                data = ascii.read(self.drive_path)
            # Renaming the columns, since the drive report doesn't contain
            # these information it self
                data['col6'].name = 'time'
                data['col8'].name = 'azimuth_avg'
                data['col13'].name = 'zenith_avg'
                return data
        else:
            raise Exception("No drive report file found")

    def cal_pointingposition(self, ev_time, drive_data):
        """
        Calculating pointing positions by interpolation

        Parameters:
        -----------
        time: array
            times from events

        Drivereport: Container
            a container filled with drive information
        """
        drive_container = LSTDriveContainer()
        drive_container.time = drive_data['time'].data
        drive_container.azimuth_avg = drive_data['azimuth_avg'].data
        drive_container.altitude_avg =  90.0 - drive_data['zenith_avg'].data 
      

        xp = drive_container.time
        lower_drive_time = xp[xp < ev_time].max()
        upper_drive_time = xp[xp > ev_time].min()

        time_in_window = (xp >= lower_drive_time) & (xp <= upper_drive_time)
        run_times = xp[time_in_window]

        if len(run_times) > 1:
            run_azimuth = drive_container.azimuth_avg[time_in_window]
            run_altitude = drive_container.altitude_avg[time_in_window]

            ev_azimuth = np.interp(ev_time, run_times, run_azimuth) * u.deg
            ev_altitude = np.interp(ev_time, run_times, run_altitude) * u.deg
            return ev_azimuth, ev_altitude
        else:
            raise Exception("No drive time in the range of event times")

