Code snippets
=============

Some code snippets that can be useful.


Print lstchain version used to produce a file
---------------------------------------------

The lstchain version is stored in the file metadata. You can print it with the following command:

In python: 

.. code-block:: python

    import tables
    filename = 'dl1_file.h5'
    file = tables.open_file(filename)
    print(file.root._v_attrs['LSTCHAIN_VERSION'])

In bash:

.. code-block:: bash

    h5dump -a /LSTCHAIN_VERSION dl1_file.h5

.. note::
    ctapipe and ctapipe_io_lst versions are also stored in the metadata.
    You can print them with the same commands, replacing LSTCHAIN_VERSION by CTAPIPE_VERSION or CTAPIPE_IO_LST_VERSION.

Print configuration parameters used by the script/tool
------------------------------------------------------
The configuration file used by a given script/tool is stored in the metadata of the output file. Note that the
configuration file can contain more parameters than the ones used in a given analysis stage as the same configuration
file can be used for several stages.

.. code-block:: python

    import tables
    import yaml
    filename = 'dl1_file.h5'
    with tables.open_file(filename) as file:
        config = yaml.safe_load(file.root.dl1.event.telescope.parameters.LST_LSTCam.attrs["config"])
    print(config)
    # Or for a given parameter section e.g.:
    print(config['tailcut_clean_with_pedestal_threshold'])

Print calibration and auxiliary files used by the script/tool
-------------------------------------------------------------
For observed data, the calibration and auxiliary files used to produce DL1 data are stored
in the metadata of the DL1 files. You can print them with the following command:

.. code-block:: python

    import tables
    import yaml
    filename = 'dl1_file.h5'
    with tables.open_file(filename) as file:
        config = yaml.safe_load(file.root.dl1.event.telescope.monitoring.calibration.attrs["config"])
    # Drive log with pointing information
    print(config['source_config']['LSTEventSource']['PointingSource'])
    # Run summary file with reference timestamps and counters for timestamp calculation
    print(config['source_config']['LSTEventSource']['EventTimeCalculator'])
    # Calibration files (DRS4 baseline corrections, calibration coefficients, time calibration, etc)
    print(config['source_config']['LSTEventSource']['LSTR0Corrections'])
    print(config['LSTCalibrationCalculator']['systematic_correction_path'])

All these metadata information can be accessed interactively using ViTables.