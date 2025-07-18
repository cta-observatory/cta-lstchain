#!/usr/bin/env python
"""
Script to apply gain selection to the raw fits.fz produced by EvB6 in CTAR1
format (aka R1v1).

Gain selection is only applied to shower events (not to interleaved pedestal
and flatfield)

It uses heuristic identification of the interleaved flatfield (FF) events (
given the occasional problems we have with the FF event tagging)

"""
import logging
import protozfits
import argparse
import sys
import re

from pathlib import Path
from ctapipe.containers import EventType

import lstchain.paths as paths
import numpy as np
from contextlib import ExitStack

parser = argparse.ArgumentParser(description="Gain selector program (R0 to "
                                             "R0G)")
parser.add_argument('-f', '--R0-file', dest='input_file',
                    type=str, help='Input R0 file name (of stream 1)')

parser.add_argument('-o', '--output-dir', dest='output_dir',
                    type=str, default='./',
                    help='Output directory')

parser.add_argument('--log', dest='log_file',
                    type=str, default=None,
                    help='Log file name')

parser.add_argument('--no-flatfield-heuristic', action='store_false', 
                    dest="use_flatfield_heuristic",
                    help=("If given, do *not* identify flatfield events"
                          " heuristically from the raw data. "
                          "Trust event_type."))

# Range of waveform to be checked (for gain selection & heuristic FF
# identification)
SAMPLE_START = 3
SAMPLE_END = 39

# Level of high gain (HG) required for switching to low gain (LG)
THRESHOLD = 3900

# Events for which gain selection will be applied:
EVENT_TYPES_TO_REDUCE = [EventType.SUBARRAY, EventType.UNKNOWN]

def main():
    args = parser.parse_args()
  
    input_file = args.input_file
    output_dir = args.output_dir
    use_flatfield_heuristic =  args.use_flatfield_heuristic

    log = logging.getLogger(__name__)
    log.setLevel(logging.INFO)

    log_file = args.log_file
    runinfo = paths.parse_r0_filename(input_file)

    if log_file is None:
        log_file = output_dir
        log_file += f'/R0_to_R0g_Run{runinfo.run:05d}.{runinfo.subrun:04d}.log'

    formatter = logging.Formatter('%(asctime)s - '
                                  '%(levelname)s - %(message)s',
                                  '%Y-%m-%d %H:%M:%S')
    handler = logging.FileHandler(log_file, mode='w')
    handler.setFormatter(formatter)
    logging.getLogger().addHandler(handler)

    # Loop over the files (4 streams) to perform the gain selection:
    
    input_stream_names = [Path(Path(input_file).parent,
                               re.sub("LST-1...Run", 
                                      f"LST-1.{id_stream+1}.Run",
                                      Path(input_file).name))
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

    # Counters to keep track of how many FF-like events are
    # tagged as FF, and how many are not (and vice-versa):
    num_FF_like_with_FF_type = 0
    num_FF_like_with_no_FF_type = 0
    num_FF_unlike_with_FF_type  = 0
    num_events = 0

    with ExitStack() as stack:
        for i, name in enumerate(output_stream_names):
            header = input_streams[i].Events.header
            n_tiles = header["NAXIS2"]
            rows_per_tile = header["ZTILELEN"]

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

            wf_offset = input_streams[i].DataStream[0].waveform_offset

            for event in input_streams[i].Events:
                # skip corrupted events:
                if event.event_id == 0:
                    continue
                num_events += 1

                num_gains = event.num_channels
                if num_gains != 2:
                    log.error('You are attempting gain selection on '
                              'data with only one gain!')
                    sys.exit(1)

                wf = protozfits.any_array_to_numpy(event.waveform)
                wf = wf.reshape((num_gains, num_pixels, num_samples))
                hg = wf[0]
                lg = wf[1]

                evtype = EventType(event.event_type)
                evtype_heuristic = get_event_type(hg, wf_offset, evtype)

                if evtype == EventType.FLATFIELD:
                    if evtype_heuristic == EventType.FLATFIELD:
                        num_FF_like_with_FF_type += 1
                    else:
                        num_FF_unlike_with_FF_type += 1
                elif evtype_heuristic == EventType.FLATFIELD:
                    num_FF_like_with_no_FF_type += 1

                if use_flatfield_heuristic:
                    evtype = evtype_heuristic

                if evtype in EVENT_TYPES_TO_REDUCE:
                    # Find pixels with HG above gain switch threshold:
                    use_lg = (np.max(hg[:, SAMPLE_START:SAMPLE_END], axis=1)
                              > THRESHOLD)
                    new_wf = np.where(use_lg[:, None], lg, hg) # gain-selected
                    event.waveform.data = new_wf.tobytes()
                    event.num_channels = 1  # Just one gain stored!

                    pixel_status = protozfits.any_array_to_numpy(event.pixel_status)
                    # Set to 0 the status bit of the removed gain:
                    new_status = np.where(use_lg,
                                          pixel_status & 0b11111011,
                                          pixel_status & 0b11110111)
                    event.pixel_status.data = new_status.tobytes()

                stream.write_message(event)

            stream.close()
            input_streams[i].close()


    log.info('Number of processed events: %d', num_events)
    if num_FF_like_with_FF_type > 0:
        log.info('FF-like events tagged as FF: %d', 
                 num_FF_like_with_FF_type)
    else:
        log.warning('FF-like events tagged as FF: %d !!', 
                    num_FF_like_with_FF_type)

    log.info('FF-like events not tagged as FF: %d', 
             num_FF_like_with_no_FF_type)

    log.info('FF-unlike events tagged as FF: %d', 
             num_FF_unlike_with_FF_type)


    num_FF_like = num_FF_like_with_no_FF_type + num_FF_like_with_FF_type

    # If a relevant fraction of FF-like events were not tagged as FF...:
    max_frac = 0.1
    if num_FF_like > 0:
        frac_untagged_ff_like = num_FF_like_with_no_FF_type / num_FF_like
        if frac_untagged_ff_like > max_frac:
            log.warning('%d percent of FlatField(FF)-like events are not tagged as FF', 
                     int(100*frac_untagged_ff_like))
            log.warning('This may be due to anomalous events, like car flashes or LIDAR shots')
            log.warning('More rarely, it could result from mis-tagging of FF events')
            log.warning('A thorough inspection of the data check plots is recommended\n')

    log.info('R0 to R0G conversion finished successfully!')

def get_event_type(wf_hg, offset, evtype):

    # For heuristic flat field identification (values refer to
    # baseline-subtracted HG integrated waveforms):
    MIN_FLATFIELD_ADC = 3000
    MAX_FLATFIELD_ADC = 12000
    MIN_FLATFIELD_PIXEL_FRACTION = 0.8

    evtype_heuristic = evtype

    # pixel-wise integral:
    wf_hg_sum = np.sum(wf_hg - offset, axis=1)
    ff_pix = ((wf_hg_sum > MIN_FLATFIELD_ADC) &
              (wf_hg_sum < MAX_FLATFIELD_ADC))

    # Check fraction of pixels with HG in "FF-like range"
    if ff_pix.sum() / len(ff_pix) > MIN_FLATFIELD_PIXEL_FRACTION:
        # Looks like a FF event:
        evtype_heuristic = EventType.FLATFIELD
    elif evtype == EventType.FLATFIELD:
        # Does not look like a FF, but was tagged as such:
        evtype_heuristic = EventType.UNKNOWN

    return evtype_heuristic
