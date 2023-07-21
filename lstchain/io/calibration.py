"""IO functions related to calibration files"""

from ctapipe.io import HDF5TableReader
from ctapipe.containers import (
    WaveformCalibrationContainer,
    PedestalContainer,
    FlatFieldContainer,
    PixelStatusContainer,
    MonitoringCameraContainer,
)


def read_calibration_file(h5file, tel_id=1):
    """Read camera calibration container for specified telescope"""
    base = f'tel_{tel_id}'

    with HDF5TableReader(h5file) as reader:
        return MonitoringCameraContainer(
            calibration=next(reader.read(f'/{base}/calibration', WaveformCalibrationContainer)),
            pedestal=next(reader.read(f'/{base}/pedestal', PedestalContainer)),
            flatfield=next(reader.read(f'/{base}/flatfield', FlatFieldContainer)),
            pixel_status=next(reader.read(f"/{base}/pixel_status", PixelStatusContainer)),
        )
