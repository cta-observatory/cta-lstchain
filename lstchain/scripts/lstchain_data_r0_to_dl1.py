import argparse
from ctapipe.utils import get_dataset_path
from lstchain.reco import r0_to_dl1
from lstchain.io.config import read_configuration_file
import os

parser = argparse.ArgumentParser(description="R0 to DL1")


parser.add_argument('--infile', '-f', type=str,
                    dest='infile',
                    help='path to the .fits.fz file with the raw events',
                    default=None, required=True)

parser.add_argument('--outdir', '-o', action='store', type=str,
                    dest='outdir',
                    help='Path where to store the reco dl2 events',
                    default='./dl1_data/')

parser.add_argument('--config_file', '-conf', action='store', type=str,
                    dest='config_file',
                    help='Path to a configuration file. If none is given, a standard configuration is applied',
                    default=None
                    )

parser.add_argument('--pedestal_path', '-pedestal', action='store', type=str,
                    dest='pedestal_path',
                    help='Path to a pedestal file',
                    default=None, required=True
                    )

parser.add_argument('--calibration_path', '-calib', action='store', type=str,
                    dest='calibration_path',
                    help='Path to a calibration file',
                    default=None, required=True
                    )

parser.add_argument('--time_calibration_path', '-time_calib', action='store', type=str,
                    dest='time_calibration_path',
                    help='Path to a calibration file for pulse time correction',
                    default=None, required=True
                    )

parser.add_argument('--pointing_file_path', '-pointing', action='store', type=str,
                    dest='pointing_file_path',
                    help='Path to the Drive log file with the pointing information.',
                    default=None
                    )

parser.add_argument('--ucts_t0_dragon', action='store', type=float,
                    dest='ucts_t0_dragon',
                    help='UCTS timestamp in nsecs, unix format and TAI scale of the \
                          first event of the run with valid timestamp. If none is \
                          passed, the start-of-the-run timestamp is provided, hence \
                          Dragon timestmap is not reliable.',
                    default="NaN"
                    )

parser.add_argument('--dragon_counter0', action='store', type=float,
                    dest='dragon_counter0',
                    help='Dragon counter (pps + 10MHz) in nsecs corresponding \
                          to the first reliable UCTS of the run. To be provided \
                          along with ucts_t0_dragon.',
                    default="NaN"
                    )

parser.add_argument('--ucts_t0_tib', action='store', type=float,
                    dest='ucts_t0_tib',
                    help='UCTS timestamp in nsecs, unix format and TAI scale of the \
                          first event of the run with valid timestamp. If none is \
                          passed, the start-of-the-run timestamp is provided, hence \
                          TIB timestmap is not reliable.',
                    default="NaN"
                    )

parser.add_argument('--tib_counter0', action='store', type=float,
                    dest='tib_counter0',
                    help='First valid TIB counter (pps + 10MHz) in nsecs corresponding \
                          to the first reliable UCTS of the run when TIB is available. \
                          To be provided along with ucts_t0_tib.',
                    default="NaN"
                    )

parser.add_argument('--max_events', '-maxevts', action='store', type=int,
                    dest='max_events',
                    help='Maximum number of events to be processed.',
                    default=int(1e15)
                    )

args = parser.parse_args()


def main():
    os.makedirs(args.outdir, exist_ok=True)

    r0_to_dl1.allowed_tels = {1, 2, 3, 4}
    output_filename = args.outdir + '/dl1_' + os.path.basename(args.infile).rsplit('.', 1)[0] + '.h5'

    config = {}
    if args.config_file is not None:
        try:
            config = read_configuration_file(args.config_file)
        except("Custom configuration could not be loaded !!!"):
            pass

    config["max_events"] = args.max_events
    
    r0_to_dl1.r0_to_dl1(args.infile,
                         output_filename=output_filename,
                         custom_config=config,
                         pedestal_path=args.pedestal_path,
                         calibration_path=args.calibration_path,
                         time_calibration_path=args.time_calibration_path,
                         pointing_file_path=args.pointing_file_path,
                         ucts_t0_dragon=args.ucts_t0_dragon,
                         dragon_counter0=args.dragon_counter0,
                         ucts_t0_tib=args.ucts_t0_tib,
                         tib_counter0=args.tib_counter0
                         )


if __name__ == '__main__':
    main()
