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

from ctapipe.io import EventSource
from ctapipe.containers import EventType
from lstchain.io import standard_config
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
# Baseline offset:
OFFSET = 400

def main():
    args = parser.parse_args()

    # Level of high gain (HG) required for switching to low gain (LG)
    THRESHOLD = 3500 + OFFSET

    input_file = args.input_file
    output_dir = args.output_dir

    # First identify properly interleaved pedestals (also in case there are
    # ucts jumps) and FF events (heuristically):
    event_id, event_type = get_event_types(input_file)

    event_id = np.array(event_id)
    event_type_val = np.array([x.value for x in event_type])

    logging.info('Identified event types and number of events:')
    for j in np.unique(event_type_val):
        logging.info(f'{EventType(j)}:, {np.sum(event_type_val == j)}\n')

    # Now loop over the files (4 streams) again to perform the actual gain
    # selection:
    k = input_file.find('/LST-1.')
    input_stream_names = [input_file[:k+7]+str(id_stream+1)+input_file[k+8:]
                          for id_stream in range(4)]
    output_stream_names = [output_dir + name[k:] for name in input_stream_names]

    input_streams = []
    for name in input_stream_names:
        input_streams.append(protozfits.File(name, pure_protobuf=True))

    try:
        num_pixels = input_streams[0].CameraConfiguration[0].num_pixels
        num_samples = input_streams[0].CameraConfiguration[
            0].num_samples_nominal
    except:
        logging.error('CameraConfiguration not found! Is this CTAR1 data?')
        exit(1)

    with ExitStack() as stack:
        for i, name in enumerate(output_stream_names):

            n_tiles = input_streams[i].Events.protobuf_i_fits.header[
                'NAXIS2'].value
            rows_per_tile = input_streams[i].Events.protobuf_i_fits.header[
                'ZTILELEN'].value

            stream = stack.enter_context(protozfits.ProtobufZOFits(
                    n_tiles=n_tiles,
                    rows_per_tile=rows_per_tile,
                    compression_block_size_kb=64*1024,
                    defaul_compression="lst"))
            stream.open(name)

            stream.move_to_new_table("CameraConfiguration")
            stream.write_message(input_streams[i].CameraConfiguration[0])
            stream.move_to_new_table("DataStream")
            stream.write_message(input_streams[i].DataStream[0])

            stream.move_to_new_table("Events")

            for event in input_streams[i].Events:
                # skip corrupted events:
                if event.event_id == 0:
                    continue
                # Get the event type, from the first loop over the files:
                evtype = event_type_val[event_id==event.event_id][0]

                if ((evtype == EventType.SUBARRAY.value) |
                    (evtype == EventType.UNKNOWN.value)):
                    # Find pixels with HG above gain switch threshold:
                    wf = protozfits.any_array_to_numpy(event.waveform)
                    num_gains = int(wf.size / num_pixels / num_samples)

                    if num_gains != 2:
                        logging.error('You are attempting gain selection on '
                                      'data with only one gain!')
                        exit(1)

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

    logging.info('R0 to R0G conversion finished successfully!')
    return(0)


def get_event_types(input_file):

    # For heuristic flat field identification (values refer to
    # baseline-subtracted HG integrated waveforms):
    MIN_FLATFIELD_ADC = 3000
    MAX_FLATFIELD_ADC = 12000
    MIN_FLATFIELD_PIXEL_FRACTION = 0.8

    standard_config['source_config']['LSTEventSource'][
        'apply_drs4_corrections'] = False
    standard_config['source_config']['LSTEventSource'][
        'pointing_information'] = False

    event_type = []
    event_id = []
    with EventSource(input_url=input_file,
                     config=Config(standard_config['source_config'])) as source:
        source.pointing_information = False
        source.trigger_information = True
        source.log.setLevel(logging.WARNING)
        try:
            for event in source:
                if event.r0.tel[1].waveform is None:
                    logging.error('The data seem to contain no R0 waveforms. '
                                  'Is this already gain-selected data?')
                    exit(1)

                # Check if this may be a FF event
                # Subtract baseline offset (convert to signed integer to
                # allow for negative fluctuations!)
                wf_hg = event.r0.tel[1].waveform[0][:,
                        SAMPLE_START:SAMPLE_END].astype('int16') - OFFSET
                # pixel-wise integral:
                wf_hg_sum = np.sum(wf_hg, axis=1)
                ff_pix = ((wf_hg_sum > MIN_FLATFIELD_ADC) &
                          (wf_hg_sum < MAX_FLATFIELD_ADC))
                event_type.append(event.trigger.event_type)

                # Check fraction of pixels with HG in "FF-like range"
                if ff_pix.sum() / event.r0.tel[1].waveform.shape[1] > \
                        MIN_FLATFIELD_PIXEL_FRACTION:
                    # Looks like a FF event:
                    event_type[-1] = EventType.FLATFIELD
                elif event_type[-1] == EventType.FLATFIELD:
                    # If originally tagged as FF, but not looking like FF:
                    event_type[-1] = EventType.UNKNOWN

                event_id.append(event.index.event_id)

            logging.info('Finished first loop over input files - all ok!')

        except Exception as err:
            logging.error(err)
            logging.error('Something went wrong!')
            exit(1)

    return event_id, event_type



if __name__ == '__main__':
    main()