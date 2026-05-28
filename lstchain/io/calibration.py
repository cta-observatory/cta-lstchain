"""IO functions related to calibration files"""

from pathlib import Path
from astropy.io import fits
from astropy.table import QTable
from ctapipe.io import HDF5TableReader
from ctapipe.io import metadata as meta
from lstcam_calib.io import get_dataset_keys

from ctapipe.containers import (
    WaveformCalibrationContainer,
    PedestalContainer,
    FlatFieldContainer,
    PixelStatusContainer,
    MonitoringCameraContainer,
)

CALIB_CONTAINERS = {
    "calibration": WaveformCalibrationContainer,
    "flatfield": FlatFieldContainer,
    "pedestal": PedestalContainer,
    "pixel_status": PixelStatusContainer,
}

def get_metadata(path):
    """Read metadata from calibration file, either fits or h5."""
    if path.name.endswith(".fits") or path.name.endswith(".fits.gz"):
        with fits.open(path) as hdul:
            return hdul[0].header

    elif path.name.endswith(".h5") or path.name.endswith(".hd5"):
        return meta.read_hdf5_metadata(path)

    else:
        raise ValueError(
            f"path {path} is neither a fits nor a h5 file:  %s ", path.suffix
        )


def read_calibration_file(file, row_number=0):
    """Read camera calibration container for specified telescope."""

    # flexible reading of h5 file to be compatible with old key name
    if Path(file).name.endswith(".h5"):
        mon_data = MonitoringCameraContainer(
            calibration=next(reader.read(f'/{base}/calibration', 
                                         WaveformCalibrationContainer)),
            pedestal=next(reader.read(f'/{base}/pedestal', 
                                      PedestalContainer)),
            flatfield=next(reader.read(f'/{base}/flatfield', 
                                       FlatFieldContainer)),
            pixel_status=next(reader.read(f"/{base}/pixel_status", 
                                          PixelStatusContainer)))
        # add metadata
        mon_data.meta = meta.read_reference_metadata(Path(file))

    elif Path(file).name.endswith(".fits") or Path(file).name.endswith(".fits.gz"):
        mon_data = MonitoringCameraContainer()
        with fits.open(file) as f:
            for key in CALIB_CONTAINERS.keys():
                table = QTable.read(f, hdu=key)
                row = table[row_number]
                for col in row.keys():
                    mon_data[key][col] = row[col]

            # add metadata
            mon_data.meta = f[0].header

    else:
        raise ValueError("Wrong calibration file format")

    return mon_data
