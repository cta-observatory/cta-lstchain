import h5py
import json
import importlib
import torch
import indexedconv.utils as ic_utils


def load_camera_parameters(camera_type, camera_parameters_path):
    """Loads camera parameters : nbCol and injTable.

    Args:
        camera_type (str): the type of camera to load data for ; eg 'LSTCAM'

    Returns:
        A dictionary
    """
    camera_parameters = {}
    if camera_type in ['LSTCAM', 'LST', 'NECTAR', 'MSTN']:
        camera_parameters['layout'] = 'Hex'
    else:
        camera_parameters['layout'] = 'Square'
    with h5py.File(camera_parameters_path, 'r') as hdf5_file:
        camera_parameters['nbRow'] = hdf5_file[camera_type].attrs['nbRow']
        camera_parameters['nbCol'] = hdf5_file[camera_type].attrs['nbCol']
        camera_parameters['injTable'] = hdf5_file[camera_type]['injTable'][()]
        camera_parameters['pixelsPosition'] = hdf5_file[camera_type]['pixelsPosition'][()]

    return camera_parameters


def load_model(experiments_path, experiment_name, checkpoint, camera_parameters_path):
    """Loads a trained model from an experiment.

    Args:
    experiments_path (str): the path to the experiments folder
    experiment_name (str): the name of the experiment. The experiment folder contains the network definition,
        parameters and checkpoints
    checkpoint (int): the checkpoint version
    camera_parameters_path (str): the path to the file containing the info of the different cameras
    """
    exp_path = experiments_path + '/' + experiment_name
    exp_info = json.load(open(exp_path + '/' + experiment_name + '_settings.json'))
    net_name = [k for k in exp_info['network'].keys()][0]

    spec = importlib.util.spec_from_file_location("nets", exp_path + '/nets.py')
    nets = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(nets)

    model_net = eval('nets.' + net_name)

    saved_model = torch.load(f=exp_path + '/checkpoint_' + str(checkpoint) + '.tar',
                             map_location=lambda storage, loc: storage)

    camera_parameters = load_camera_parameters(exp_info['dataset_parameters']['camera_type'], camera_parameters_path)
    net_parameters = exp_info['network'][net_name]
    net = model_net(net_parameters, camera_parameters)

    net.load_state_dict(saved_model)
    net.eval()

    return net
