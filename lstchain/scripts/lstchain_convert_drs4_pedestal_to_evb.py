'''
Convert pedestal h5 file to a non-descriptive binary file as needed by EVB.

The output format is an array of uint16, little endian values in the order
the module ids and of shape [N_GAINS, N_PIXELS, N_CAPACITORS], with the left
most dimension varying slowest (C order).
'''
import numpy as np
import sys
from xml.etree.ElementTree import ElementTree
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from ctapipe.io import read_table
from pkg_resources import resource_filename
from pathlib import Path

default_camera_config = Path(resource_filename('lstchain', 'resources/conf.LSTCam.xml'))



parser = ArgumentParser(
    description=__doc__,
    formatter_class=ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    'drs4_baseline_file',
    help='DRS4 pedestal baseline h5 file as produced by lstchain_create_drs4_baseline_file'
)
parser.add_argument('output_file', type=Path)
parser.add_argument(
    '-c', '--camera-config',
    default=default_camera_config,
    type=Path,
    help='Camera configuration XML file'
)
parser.add_argument('-t', '--tel-id', default=1, type=int)


def read_pedestal(path, tel_id):
    '''Read the pedestal array from a h5 file'''
    table = read_table(path, f"/r1/monitoring/drs4_baseline/tel_{tel_id:03d}")
    return table[0]['baseline_mean']


def get_pixel_ids_in_module_order(path):
    '''Read pixel ids in module order from camera description XML'''
    root = ElementTree().parse(path)
    camera_description = root.find('cameraDescription')

    pixel_ids = []
    for module in camera_description.findall('module'):
        for pixel in module.findall('pixel'):
            pixel_abs_id = int(pixel.find("pixelAbsId").text)
            pixel_ids.append(pixel_abs_id)
    return np.array(pixel_ids)


def main():
    args = parser.parse_args()

    pedestal = read_pedestal(args.drs4_baseline_file, args.tel_id)

    if not (pedestal.dtype == np.uint16 or pedestal.dtype == np.int16):
        print(f'Input pedestal must have dtype (u)int16, got {pedestal.dtype}.', file=sys.stderr)
        print('Make sure the pedestal file was generated by lstchain > 0.8.2', file=sys.stderr)
        sys.exit(1)

    # reorder to module ordering and make sure it's contiguous again
    pixel_ids_in_module_order = get_pixel_ids_in_module_order(args.camera_config)
    pedestal_in_module_order = pedestal[:, pixel_ids_in_module_order]
    pedestal_in_module_order = np.ascontiguousarray(pedestal_in_module_order)

    with args.output_file.open(mode='wb') as f:
        f.write(pedestal_in_module_order)


if __name__ == '__main__':
    main()