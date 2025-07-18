#!/usr/bin/env python
"""
Script to apply data volume reduction to R0G files in CTAR1 format (aka R1v1).
Reduction is only applied to shower events (not to interleaved pedestal
and flatfield).

In principle it can work on R0 data (2 gains), but no gain selection will be 
applied, and it does not make much sense to keep both gains on reduced 
(pixel-selected) data!

If the option --fix-pixel-status is used, then the input file has to be an R0V
file, and the pixel selection file must be the same that was used to produce 
it. this is a special option to fix R0V that were created with wrong pixel 
status (but correct waveform selection).

"""
import logging
import protozfits
import argparse
import sys
import re

from pathlib import Path
from ctapipe.containers import EventType
from ctapipe.io import read_table
from ctapipe_io_lst import PixelStatus

import lstchain.paths as paths
import numpy as np
from contextlib import ExitStack

parser = argparse.ArgumentParser(description="Volume reducer program (R0G to "
                                             "R0V)")
parser.add_argument('-f', '--R0G-file', dest='input_file', required=True,
                    type=Path, help='Input R0G file name (of stream 1)')

parser.add_argument('-o', '--output-dir', dest='output_dir',
                    type=Path, default='./',
                    help='Output directory')

parser.add_argument('--pixselection-file', dest='pix_file', required=True,
                    type=Path, help='Pixel_selection .h5 file '
                    'produced by lstchain_dvr_pixselector')

parser.add_argument('--log', dest='log_file',
                    type=Path, default=None,
                    help='Log file name')

parser.add_argument('--fix-pixel-status', dest='fix_pixel_status',
                    action='store_true', required=False, 
                    default=False,
                    help='Just set the pixel status flags (input file should be R0V)')

# Events for which gain selection will be applied:
EVENT_TYPES_TO_REDUCE = [EventType.SUBARRAY, EventType.UNKNOWN]
UNSET_DVR_BIT_MASK = ~np.uint8(PixelStatus.DVR_STATUS_0 | PixelStatus.DVR_STATUS_1)
SET_DVR_BIT_0 = np.uint8(PixelStatus.DVR_STATUS_0)

def main():
    args = parser.parse_args()

    input_file = args.input_file
    output_dir = args.output_dir

    log = logging.getLogger(__name__)
    log.setLevel(logging.INFO)

    log_file = args.log_file
    runinfo = paths.parse_r0_filename(input_file)
    if log_file is None:
        log_file = output_dir / f'R0g_to_R0v_Run{runinfo.run:05d}.{runinfo.subrun:04d}.log'
    formatter = logging.Formatter('%(asctime)s - '
                                  '%(levelname)s - %(message)s',
                                  '%Y-%m-%d %H:%M:%S')
    handler = logging.FileHandler(log_file, mode='w')
    handler.setFormatter(formatter)
    logging.getLogger().addHandler(handler)
    logging.getLogger().addHandler(logging.StreamHandler(sys.stderr))

    pix_file = args.pix_file
    if not pix_file.is_file():
        log.error('File %s does not exist!', pix_file) 
        sys.exit(1)

    log.info('Using %s to select pixels to be saved.', pix_file)

    # Get the event-wise lists of pixels to be saved in the R0V output:
    pixtable = read_table(pix_file, "selected_pixels_masks")
    pixel_mask = dict(zip(pixtable['event_id'].data,
                          pixtable['pixmask'].data))
    # Get also the event_type from the Pixel selection file:
    event_type = dict(zip(pixtable['event_id'].data,
                          [EventType(et) for et 
                           in pixtable['event_type'].data]))

    # Loop over the files (4 streams) to perform the gain selection:
    
    input_stream_names = [Path(input_file.parent,
                               re.sub("LST-1...Run", 
                                      f"LST-1.{id_stream+1}.Run",
                                      input_file.name))
                          for id_stream in range(4)]
  
    output_stream_names = [Path(output_dir, Path(inputsn).name) 
                           for inputsn in input_stream_names]

    input_streams = []
    for name in input_stream_names:
        input_streams.append(protozfits.File(str(name), pure_protobuf=True))

    try:
        camera_config = input_streams[0].CameraConfiguration[0]
    except (AttributeError, IndexError):
        log.error('CameraConfiguration not found! Is this R1v1 data?')
        sys.exit(1)

    num_pixels = camera_config.num_pixels
    num_samples = camera_config.num_samples_nominal
    pixel_id_map = protozfits.any_array_to_numpy(camera_config.pixel_id_map)

    num_events = 0

    with ExitStack() as stack:
        for i, name in enumerate(output_stream_names):
            header = input_streams[i].Events.header
            n_tiles = header["NAXIS2"]
            rows_per_tile = header["ZTILELEN"]

            if args.fix_pixel_status:
                r0v_input = header.get("LSTDVR", False)
                if not r0v_input:
                    raise RuntimeError('Input file must be R0V if --fix-pixel-status option is used!')

            stream = stack.enter_context(protozfits.ProtobufZOFits(
                    n_tiles=n_tiles,
                    rows_per_tile=rows_per_tile,
                    compression_block_size_kb=64*1024,
                    default_compression="lst-r1v1-uncalibrated"))
            stream.open(str(name))

            stream.move_to_new_table("DataStream")
            stream.write_message(input_streams[i].DataStream[0])
          
            stream.move_to_new_table("CameraConfiguration")
            stream.write_message(input_streams[i].CameraConfiguration[0])

            stream.move_to_new_table("Events")
            stream.set_bool("LSTDVR", True, "LST offline DVR applied")

            for event in input_streams[i].Events:
                # skip corrupted events:
                if event.event_id == 0:
                    continue
                num_events += 1

                # Check if this event is known
                if event.event_id not in pixel_mask:
                    log.warning('Event id %d not found in pixel selection file!',
                             event.event_id)
                    log.warning('    ==> SKIPPING it!')
                    continue

                num_gains = event.num_channels

                evtype = event_type[event.event_id]
                pixel_status = protozfits.any_array_to_numpy(event.pixel_status)
                ordered_pix_mask = np.array(num_pixels*[True])
              
                if evtype in EVENT_TYPES_TO_REDUCE:
                    pixmask = pixel_mask[event.event_id]
                    ordered_pix_mask = pixmask[pixel_id_map]

                    wf = protozfits.any_array_to_numpy(event.waveform)
                    # Keep only selected waveforms - unless the option
                    # --fix-pixel-status has been selected, in which case
                    # it is not needed because input file is already R0V:
                    if not args.fix_pixel_status:
                        wf = wf.reshape((num_gains, num_pixels, num_samples))
                        new_wf = wf[:, ordered_pix_mask, :]
                        event.waveform.data = new_wf.tobytes()
                    else:
                        # Check that the number of saved pixels in the mask 
                        # is consistent with the size of the waveform (will
                        # be the case, if the pixel selection file is the same
                        # that was used in the creation of the input R0V file
                        if len(wf) != num_gains * num_samples * ordered_pix_mask.sum():
                            raise RuntimeError('The pixel selection file is not consistent with the input R0V file!')

                # Modify pixel status as needed
                new_status = np.where(ordered_pix_mask,
                                      pixel_status | SET_DVR_BIT_0,
                                      pixel_status & UNSET_DVR_BIT_MASK)
                event.pixel_status.data = new_status.tobytes()

                stream.write_message(event)

            stream.close()
            input_streams[i].close()

    log.info('Number of processed events: %d', num_events)
    log.info('R0G to R0V conversion finished successfully!')
