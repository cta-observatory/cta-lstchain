from ctapipe.utils import get_dataset_path

def test_import_calib():
    from lstchain import calib

def test_import_reco():
    from lstchain import reco

def test_import_visualization():
    from lstchain import visualization

def test_import_lstio():
    from lstchain import io


def test_dl0_to_dl1():
    from lstchain.reco.dl0_to_dl1 import r0_to_dl1
    infile = get_dataset_path('gamma_test_large.simtel.gz')
    r0_to_dl1(infile)

def test_buildModels():
    from lstchain.reco.reco_dl1_to_dl2 import buildModels
    infile = 'dl1_gamma_test_large.h5'
    features = ['intensity', 'width', 'length']

    RFreg_Energy, RFreg_disp_, RFcls_GH = buildModels(
        infile, infile,
        features,
        SaveModels=True)

    import joblib
    joblib.dump(RFreg_Energy, 'rf_energy.pkl')
    joblib.dump(RFreg_disp_, 'rf_disp.pkl')
    joblib.dump(RFcls_GH, 'rf_gh.pkl')


def test_ApplyModels():
    from lstchain.reco.reco_dl1_to_dl2 import ApplyModels
    import pandas as pd
    import joblib

    dl1_file = 'dl1_gamma_test_large.h5'
    dl1 = pd.read_hdf(dl1_file)
    features = ['intensity', 'width', 'length']
    # Load the trained RF for reconstruction:
    fileE = 'rf_energy.pkl'
    fileD = 'rf_disp.pkl'
    fileH = 'rf_gh.pkl'

    RFreg_Energy = joblib.load(fileE)
    RFreg_disp_ = joblib.load(fileD)
    RFcls_GH = joblib.load(fileH)

    dl2 = ApplyModels(dl1, features, RFcls_GH, RFreg_Energy, RFreg_disp_)



