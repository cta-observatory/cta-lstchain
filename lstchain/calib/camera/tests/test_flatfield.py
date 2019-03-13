from ctapipe.utils import get_dataset_path

from ctapipe.io import event_source
from .. import FlasherFlatFieldCalculator

"""
#example_file_path = get_dataset_path("/ctadata/franca/LST/LST-1.4.Run00167.0001.fits.fz")
input_reader = event_source("/ctadata/franca/LST/LST-1.4.Run00167.0001.fits.fz", max_events=10)


def test_flasherflatfieldcalculator():

    ff_calculator = FlasherFlatFieldCalculator(sample_size=3, tel_id=0)

    for event in input_reader:

        ff_data = ff_calculator.calculate_relative_gain(event)

        if ff_calculator.num_events_seen == ff_calculator.sample_size:
            assert ff_data
"""