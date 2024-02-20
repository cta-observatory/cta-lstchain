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

from ctapipe.io import EventSource
from ctapipe.containers import EventType
from lstchain.io import standard_config
import lstchain.paths as paths
from traitlets.config import Config

import numpy as np
from contextlib import ExitStack

parser = argparse.ArgumentParser(description="Gain selector program (R0 to "
                                             "R0G)")
parser.add_argument('-f', '--R0-file', dest='input_file',
                    type=str, help='Input R0 file name (of stream 1)')

parser.add_argument('-o', '--output-dir', dest='output_dir',
                    type=str, default='./',
                    help='Output directory')

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
  
    log = logging.getLogger("lstchain_r0_to_r0g")
  
    # First identify properly interleaved pedestals (also in case there are
    # ucts jumps) and FF events (heuristically):
    event_type = get_event_types(input_file)
    event_type_val = np.array([x.value for x in event_type.values()])

    log.info('Identified event types and number of events:')
    evtype, counts = np.unique(event_type_val, return_counts=True)
    for j, n in zip(evtype, counts):
        log.info('%s: %d', EventType(j), n)

    # Now loop over the files (4 streams) again to perform the actual gain
    # selection:
    
    run_info = pth.parse_r0_filename(input_file)
    input_stream_names = [pth.Path(pth.Path(input_file).parent,
                                   pth.run_to_r0_filename(run_info.tel_id, 
                                                          run_info.run,
                                                          run_info.subrun, 
                                                          id_stream))
                          for id_stream in range(4)]
    output_stream_names = [pth.Path(output_dir, pth.Path(inputsn).name) 
                           for inputsn in input_stream_names]

    input_streams = []
    for name in input_stream_names:
        input_streams.append(protozfits.File(name, pure_protobuf=True))

    try:
      camera_config = input_streams[0].CameraConfiguration[0]
    except (AttributeError, IndexError):
      log.error('CameraConfiguration not found! Is this R1v1 data?')
      sys.exit(1)

    num_pixels = camera_config.num_pixels
    num_samples = camera_config.num_samples_nominal


    with ExitStack() as stack:
        for i, name in enumerate(output_stream_names):
            header = input_streams[i].Events.header
            n_tiles = header["NAXIS2"]
            rows_per_tile = header["ZTILELEN"]

            stream = stack.enter_context(protozfits.ProtobufZOFits(
                    n_tiles=n_tiles,
                    rows_per_tile=rows_per_tile,
                    compression_block_size_kb=64*1024,
                    defaul_compression="lst"))
            stream.open(name)

            stream.move_to_new_table("DataStream")
            stream.write_message(input_streams[i].DataStream[0])
          
            stream.move_to_new_table("CameraConfiguration")
            stream.write_message(input_streams[i].CameraConfiguration[0])

            stream.move_to_new_table("Events")

            for event in input_streams[i].Events:
                # skip corrupted events:
                if event.event_id == 0:
                    continue
                # Get the event type, from the first loop over the files:
                evtype = event_type[event.event_id]
              
                if evtype in EVENT_TYPES_TO_REDUCE:
                    if event.num_channels != 2:
                        log.error('You are attempting gain selection on '
                                      'data with only one gain!')
                        sys.exit(1)

                    # Find pixels with HG above gain switch threshold:
                    wf = protozfits.any_array_to_numpy(event.waveform)
                    wf = wf.reshape((num_gains, num_pixels, num_samples))
                    hg = wf[0]
                    lg = wf[1]
                    use_lg = (np.max(hg[:, SAMPLE_START:SAMPLE_END], axis=1)
                              > THRESHOLD)
                    new_wf = np.where(use_lg[:, None], lg, hg) # gain-selected
                    event.waveform.data = new_wf.tobytes()
                    event.num_channels = 1  # Just one gain stored!

                    pixel_status = protozfits.any_array_to_numpy(event.pixel_status)
                    # Set to 0 the status bit of the removed gain:
                    new_status = np.where(use_lg,
                                          pixel_status & 0b1011,
                                          pixel_status & 0b0111)
                    event.pixel_status.data = new_status.tobytes()

                stream.write_message(event)

            stream.close()
            input_streams[i].close()

    log.info('R0 to R0G conversion finished successfully!')
    


def get_event_types(input_file):

    log = logging.getLogger("lstchain_r0_to_r0g")

    # For heuristic flat field identification (values refer to
    # baseline-subtracted HG integrated waveforms):
    MIN_FLATFIELD_ADC = 3000
    MAX_FLATFIELD_ADC = 12000
    MIN_FLATFIELD_PIXEL_FRACTION = 0.8

    standard_config['source_config']['LSTEventSource'][
        'apply_drs4_corrections'] = False
    standard_config['source_config']['LSTEventSource'][
        'pointing_information'] = False

    event_type = dict()
  
    with EventSource(input_url=input_file,
                     config=Config(standard_config['source_config'])) as source:
        source.pointing_information = False
        source.trigger_information = True
        source.log.setLevel(log.WARNING)

        offset = source.data_stream.waveform_offset 
        try:
            for event in source:
                if event.r0.tel[1].waveform is None:
                    log.error('The data seem to contain no R0 waveforms. '
                                  'Is this already gain-selected data?')
                    sys.exit(1)

                # Check if this may be a FF event
                # Subtract baseline offset (convert to signed integer to
                # allow for negative fluctuations!)
                wf_hg = event.r0.tel[1].waveform[0][:,
                        SAMPLE_START:SAMPLE_END].astype('int16') - offset
                # pixel-wise integral:
                wf_hg_sum = np.sum(wf_hg, axis=1)
                ff_pix = ((wf_hg_sum > MIN_FLATFIELD_ADC) &
                          (wf_hg_sum < MAX_FLATFIELD_ADC))

                evtype = event.trigger.event_type
                # Check fraction of pixels with HG in "FF-like range"
                if ff_pix.sum() / event.r0.tel[1].waveform.shape[1] > \
                        MIN_FLATFIELD_PIXEL_FRACTION:
                    # Looks like a FF event:
                    evtype = EventType.FLATFIELD
                elif evtype == EventType.FLATFIELD:
                    # If originally tagged as FF, but not looking like FF:
                    evtype = EventType.UNKNOWN

                event_type[event.index.event_id] = evtype

            log.info('Finished first loop over input files - all ok!')

        except Exception as err:
            log.error(err)
            log.error('Something went wrong!')
            sys.exit(1)

    return event_type



if __name__ == '__main__':
    main()
