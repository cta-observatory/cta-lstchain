#!/bin/bash

dl1_gammas_tailcuts_6_3='/fefs/aswg/data/mc/DL1/20190415/gamma/south_pointing/20191128_v.0.3.1_v00/dl1_gamma_south_pointing_20191128_v.0.3.1_v00_DL1_testing.h5'

dl1_protons_tailcuts_6_3='/fefs/aswg/data/mc/DL1/20190415/proton/south_pointing/20191128_v.0.3.1_v00/dl1_proton_south_pointing_20191128_v.0.3.1_v00_DL1_testing.h5'

dl2_gammas_tailcuts_6_3='/fefs/aswg/data/mc/DL2/20190415/gamma/south_pointing/20191128_v.0.3.1_v00/dl2_dl1_gamma_south_pointing_20191128_v.0.3.1_v00_DL1_testing.h5'

dl2_protons_tailcuts_6_3='/fefs/aswg/data/mc/DL2/20190415/proton/south_pointing/20191128_v.0.3.1_v00/dl2_dl1_proton_south_pointing_20191128_v.0.3.1_v00_DL1_testing.h5'

dl2_best_cuts_tailcuts_6_3='/fefs/aswg/workspace/maria.bernardos/LSTanalysis/sensitivity/20190415/south_pointing/20191128_v.0.3.1_v00/dl2_best_cuts_south_pointing_20191128_v.0.3.1_v00_DL1_testing.h5'

result_cuts_tailcuts_6_3='/fefs/aswg/workspace/maria.bernardos/LSTanalysis/sensitivity/20190415/south_pointing/20191128_v.0.3.1_v00/results_south_pointing_20191128_v.0.3.1_v00_DL1_testing.h5'

dl1_gammas_tailcuts_8_4='/fefs/aswg/data/mc/DL1/20190415/gamma/south_pointing/20200110_v.0.3.1_tailcuts_8_4/dl1_gamma_south_pointing_20200110_v.0.3.1_tailcuts_8_4_DL1_testing.h5'

dl1_protons_tailcuts_8_4='/fefs/aswg/data/mc/DL1/20190415/proton/south_pointing/20200110_v.0.3.1_tailcuts_8_4/dl1_proton_south_pointing_20200110_v.0.3.1_tailcuts_8_4_DL1_testing.h5'

dl2_gammas_tailcuts_8_4='/fefs/aswg/data/mc/DL2/20190415/gamma/south_pointing/20200110_v.0.3.1_tailcuts_8_4/dl2_dl1_gamma_south_pointing_20200110_v.0.3.1_tailcuts_8_4_DL1_testing.h5'

dl2_protons_tailcuts_8_4='/fefs/aswg/data/mc/DL2/20190415/proton/south_pointing/20200110_v.0.3.1_tailcuts_8_4/dl2_dl1_proton_south_pointing_20200110_v.0.3.1_tailcuts_8_4_DL1_testing.h5'

dl2_best_cuts_tailcuts_8_4='/fefs/aswg/workspace/maria.bernardos/LSTanalysis/sensitivity/20190415/south_pointing/20200110_v.0.3.1_tailcuts_8_4/dl2_best_cuts_south_pointing_20200110_v.0.3.1_tailcuts_8_4_DL1_testing.h5'

result_cuts_tailcuts_8_4='/fefs/aswg/workspace/maria.bernardos/LSTanalysis/sensitivity/20190415/south_pointing/20200110_v.0.3.1_tailcuts_8_4/results_south_pointing_20200110_v.0.3.1_tailcuts_8_4_DL1_testing.h5'

dl1_gammas_tailcuts_10_5='/fefs/aswg/data/mc/DL1/20190415/gamma/south_pointing/20200110_v.0.3.1_tailcuts_10_5/dl1_gamma_south_pointing_20200110_v.0.3.1_tailcuts_10_5_DL1_testing.h5'

dl1_protons_tailcuts_10_5='/fefs/aswg/data/mc/DL1/20190415/proton/south_pointing/20200110_v.0.3.1_tailcuts_10_5/dl1_proton_south_pointing_20200110_v.0.3.1_tailcuts_10_5_DL1_testing.h5'

dl2_gammas_tailcuts_10_5='/fefs/aswg/data/mc/DL2/20190415/gamma/south_pointing/20200110_v.0.3.1_tailcuts_10_5/dl2_dl1_gamma_south_pointing_20200110_v.0.3.1_tailcuts_10_5_DL1_testing.h5'

dl2_protons_tailcuts_10_5='/fefs/aswg/data/mc/DL2/20190415/proton/south_pointing/20200110_v.0.3.1_tailcuts_10_5/dl2_dl1_proton_south_pointing_20200110_v.0.3.1_tailcuts_10_5_DL1_testing.h5'

dl2_best_cuts_tailcuts_10_5='/fefs/aswg/workspace/maria.bernardos/LSTanalysis/sensitivity/20190415/south_pointing/20200110_v.0.3.1_tailcuts_10_5/dl2_best_cuts_south_pointing_20200110_v.0.3.1_tailcuts_10_5_DL1_testing.h5'

result_cuts_tailcuts_10_5='/fefs/aswg/workspace/maria.bernardos/LSTanalysis/sensitivity/20190415/south_pointing/20200110_v.0.3.1_tailcuts_10_5/results_south_pointing_20200110_v.0.3.1_tailcuts_10_5_DL1_testing.h5'

python lstchain_mc_sensitivity.py -gd1 $dl1_gammas_tailcuts_6_3 -pd1 $dl1_protons_tailcuts_6_3 -gd2-cuts $dl2_gammas_tailcuts_6_3 -pd2-cuts $dl2_protons_tailcuts_6_3 -gd2-sens $dl2_gammas_tailcuts_6_3 -pd2-sens $dl2_protons_tailcuts_6_3 -dl2o $dl2_best_cuts_tailcuts_6_3 -c $result_cuts_tailcuts_6_3

python lstchain_mc_sensitivity.py -gd1 $dl1_gammas_tailcuts_8_4 -pd1 $dl1_protons_tailcuts_8_4 -gd2-cuts $dl2_gammas_tailcuts_8_4 -pd2-cuts $dl2_protons_tailcuts_8_4 -gd2-sens $dl2_gammas_tailcuts_8_4 -pd2-sens $dl2_protons_tailcuts_8_4 -dl2o $dl2_best_cuts_tailcuts_8_4 -c $result_cuts_tailcuts_8_4

python lstchain_mc_sensitivity.py -gd1 $dl1_gammas_tailcuts_10_5 -pd1 $dl1_protons_tailcuts_10_5 -gd2-cuts $dl2_gammas_tailcuts_10_5 -pd2-cuts $dl2_protons_tailcuts_10_5 -gd2-sens $dl2_gammas_tailcuts_10_5 -pd2-sens $dl2_protons_tailcuts_10_5 -dl2o $dl2_best_cuts_tailcuts_10_5 -c $result_cuts_tailcuts_10_5
