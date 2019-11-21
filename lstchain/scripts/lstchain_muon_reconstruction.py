"""
Example to load raw data (hessio format), calibrate and reconstruct muon
ring parameters, and write some parameters to an output table
"""
import numpy as np
import warnings
from astropy.table import Table
from ctapipe.calib import CameraCalibrator
from ctapipe.core import Tool
from ctapipe.core import traits as t
from ctapipe.image.muon.muon_diagnostic_plots import plot_muon_event
from ctapipe.image.muon.muon_reco_functions import analyze_muon_event
from ctapipe.io import event_source
from ctapipe.utils import get_dataset

warnings.filterwarnings("ignore")  # Supresses iminuit warnings


def print_muon(event, printer=print):
    for tid in event['TelIds']:
        idx = event['TelIds'].index(tid)
        if event['MuonIntensityParams'][idx]:
            printer("MUON: Run ID {} Event ID {} \
                    Impact Parameter {} Ring Width {} Optical Efficiency {}".format(
                event['MuonRingParams'][idx].obs_id,
                event['MuonRingParams'][idx].event_id,
                event['MuonIntensityParams'][idx].impact_parameter,
                event['MuonIntensityParams'][idx].ring_width,
                event['MuonIntensityParams'][idx].optical_efficiency_muon)
            )
    pass


class MuonDisplayerTool(Tool):
    name = 'ctapipe-display-muons'
    description = t.Unicode(__doc__)

    infile = t.Unicode(
        help='input file name',
        default=get_dataset('gamma_test_large.simtel.gz')
    ).tag(config=True)

    outfile = t.Unicode(help='output file name',
                        default=None).tag(config=True)

    display = t.Bool(
        help='display the camera events', default=False
    ).tag(config=True)

    classes = t.List([CameraCalibrator, ])

    aliases = t.Dict({'infile': 'MuonDisplayerTool.infile',
                      'outfile': 'MuonDisplayerTool.outfile',
                      'display': 'MuonDisplayerTool.display'
                      })


    def setup(self):
        self.calib = CameraCalibrator()

    def start(self):

        output_parameters = {'MuonEff': [],
                             'ImpactP': [],
                             'RingWidth': [],
                             'RingCont': [],
                             'RingComp': [],
                             'RingPixComp': [],
                             'Core_x': [],
                             'Core_y': [],
                             'Impact_x_arr': [],
                             'Impact_y_arr': [],
                             'MCImpactP': [],
                             'ImpactDiff': [],
                             'RingSize': [],
                             'RingRadius': [],
                             'NTels': []}

        numev = 0
        num_muons_found = 0

        for event in event_source(self.infile):
            self.log.info("Event Number: %d, found %d muons", numev, num_muons_found)
            self.calib.calibrate(event)
            muon_evt = analyze_muon_event(event)

            numev += 1

            if not muon_evt['MuonIntensityParams']:  # No telescopes contained a good muon
                continue
            else:
                if self.display:
                    plot_muon_event(event, muon_evt)

                ntels = len( event.r0.tels_with_data)
                #if(len( event.r0.tels_with_data) <= 1):
                    #continue
                #print("event.r0.tels_with_data", event.r0.tels_with_data)
                for tid in muon_evt['TelIds']:
                    idx = muon_evt['TelIds'].index(tid)
                    if muon_evt['MuonIntensityParams'][idx] is not None:
                        print("pos, tid", event.inst.subarray.positions[tid], tid)
                        tel_x = event.inst.subarray.positions[tid][0]
                        tel_y = event.inst.subarray.positions[tid][1]
                        core_x = event.mc.core_x  # MC Core x
                        core_y = event.mc.core_y  # MC Core y
                        rec_impact_x = muon_evt['MuonIntensityParams'][idx].impact_parameter_pos_x
                        rec_impact_y = muon_evt['MuonIntensityParams'][idx].impact_parameter_pos_y
                        print("rec_impact_x, rec_impact_y", rec_impact_x, rec_impact_y)
                        print("event.mc.core_x, event.mc.core_y",
                              event.mc.core_x, event.mc.core_y)
                        impact_mc = np.sqrt(np.power(core_x - tel_x, 2) +
                                         np.power(core_y - tel_y, 2))
                        print("simulated impact", impact_mc)
                        # Coordinate transformation to move the impact point to array coordinates
                        rec_impact_x_arr = tel_x + rec_impact_x 
                        rec_impact_y_arr = tel_y + rec_impact_y
                        print("impact_x_arr, impact_y_arr", rec_impact_x_arr, rec_impact_y_arr)
                        # Distance between core of the showe and impact parameter
                        impact_diff = np.sqrt(np.power(core_x - rec_impact_x_arr, 2) +
                                              np.power(core_y - rec_impact_y_arr, 2))
                        print("impact_diff ",impact_diff )
                        self.log.info("** Muon params: %s",
                                      muon_evt['MuonIntensityParams'][idx])

                        output_parameters['MuonEff'].append(
                            muon_evt['MuonIntensityParams'][idx].optical_efficiency_muon
                        )
                        output_parameters['ImpactP'].append(
                            muon_evt['MuonIntensityParams'][idx].impact_parameter.value
                        )
                        output_parameters['RingWidth'].append(
                            muon_evt['MuonIntensityParams'][idx].ring_width.value
                        )
                        output_parameters['RingCont'].append(
                            muon_evt['MuonRingParams'][idx].ring_containment
                        )
                        output_parameters['RingComp'].append(
                            muon_evt['MuonIntensityParams'][idx].ring_completeness
                        )
                        output_parameters['RingPixComp'].append(
                            muon_evt['MuonIntensityParams'][idx].ring_pix_completeness
                        )
                        output_parameters['Core_x'].append(
                            event.mc.core_x.value
                        )
                        output_parameters['Core_y'].append(
                            event.mc.core_y.value
                        )
                        output_parameters['Impact_x_arr'].append(
                            rec_impact_x_arr.value
                        )
                        output_parameters['Impact_y_arr'].append(
                            rec_impact_y_arr.value
                        )
                        output_parameters['MCImpactP'].append(
                            impact_mc.value
                        )
                        output_parameters['ImpactDiff'].append(
                            impact_diff.value
                        )
                        output_parameters['RingSize'].append(
                            muon_evt['MuonIntensityParams'][idx].ring_size
                        )
                        output_parameters['RingRadius'].append(
                            muon_evt['MuonRingParams'][idx].ring_radius.value
                        )
                        output_parameters['NTels'].append(
                            ntels
                        )

                        print_muon(muon_evt, printer=self.log.info)
                        num_muons_found += 1



        t = Table(output_parameters)
        t['ImpactP'].unit = 'm'
        t['RingWidth'].unit = 'deg'
        #t['MCImpactP'].unit = 'm'
        if self.outfile:
            t.write(self.outfile)


if __name__ == '__main__':
    tool = MuonDisplayerTool()
    tool.run()
