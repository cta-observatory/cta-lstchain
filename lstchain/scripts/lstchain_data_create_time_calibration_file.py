import argparse
import numpy as np
from traitlets.config.loader import Config
from ctapipe.io import event_source
from lstchain.calib.camera.r0 import LSTR0Corrections
from lstchain.io.config import read_configuration_file
from lstchain.calib.camera.time_correction_calculate import TimeCorrectionCalculate


''' 
Script to create drs4 time correction coefficients

'''

parser = argparse.ArgumentParser()

# Required arguments
parser.add_argument("--input_file",
                    help="Path to fitz.fz file to create pedestal file.",
                    type=str)

parser.add_argument("--output_file",
                    help="Path where script create pedestal file",
                    type=str)

# Optional argument
parser.add_argument("--max_events",
                    help="Maximum numbers of events to read. Default = 20000",
                    type=int,
                    default=20000)

parser.add_argument('--config_file', '-conf', action='store', type=str,
                    dest='config_file',
                    help='Path to a configuration file. If none is given, a standard configuration is applied',
                    default=None
                    )

parser.add_argument('--pedestal_file', '-ped', action='store', type=str,
                    dest='pedestal_file',
                    help='Path to drs4 pedestal file ',
                    default=None
                    )

args = parser.parse_args()


def main():
    print("--> Input file: {}".format(args.input_file))
    print("--> Number of events: {}".format(args.max_events))
    reader = event_source(input_url=args.input_file, max_events=args.max_events)
    print("--> Number of files", reader.multi_file.num_inputs())

    config_dic = {}
    if args.config_file is not None:
        try:
            config_dic = read_configuration_file(args.config_file)

        except("Custom configuration could not be loaded !!!"):
            pass
    # read the configuration file
    config = Config(config_dic)

    # declare the pedestal calibrator
    lst_r0 = LSTR0Corrections(pedestal_path=args.pedestal_file, config=config)

    # declare the time corrector
    timeCorr = TimeCorrectionCalculate(calib_file_path=args.output_file,
                                       config=config)

    tel_id = timeCorr.tel_id

    for i, event in enumerate(reader):
        if event.r0.event_id % 5000 == 0:
            print(event.r0.event_id)

        lst_r0.calibrate(event)

        # Cut in signal to avoid cosmic events
        if event.r1.tel[tel_id].trigger_type == 4 or (
                np.median(np.sum(event.r1.tel[tel_id].waveform[0], axis=1))> 300):

            timeCorr.calibrate_pulse_time(event)
    # write output
    timeCorr.finalize()


if __name__ == '__main__':
    main()

