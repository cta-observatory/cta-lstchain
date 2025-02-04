.. _introduction:

Introduction
============

``lstchain`` is the analysis library for the observed and simulated LST-1 data.

Dependencies
------------
``lstchain`` heavily depends on ``ctapipe``

Data levels
===========

+------+---------------------------------------------------------------------------+-------------+
| Level| Description                                                               | File Format |
+======+===========================================================================+=============+
| R0   | Uncalibrated RAW waveforms from the camera                                | ZFITS       |
+------+---------------------------------------------------------------------------+-------------+
| R1   | Calibrated waveforms from the camera                                      |             |
+------+---------------------------------------------------------------------------+-------------+
| DL1a | Integrated charge and peak position of the waveform                       | HDF5        |
+------+---------------------------------------------------------------------------+-------------+
| DL1b | Image parameters (width, length, intensity, etc.)                         | HDF5        |
+------+---------------------------------------------------------------------------+-------------+
| DL2  | Event parameters (energy, direction, time, etc.)                          | HDF5        |
+------+---------------------------------------------------------------------------+-------------+
| DL3  | Lists of reconstructed events after event selection                       | FITS        |
|      | with corresponding IRFs (AEFF, EDISP, PSF, etc.)                          |             |
+------+---------------------------------------------------------------------------+-------------+


Analysis steps
==============

These individuals steps can be run one by one but are also integrated
in more complete workflows presented in :doc:`lst_analysis_workflow`.

R1 to DL1
---------

MC data
^^^^^^^

Use ``lstchain.scripts.lstchain_mc_r0_to_dl1``. 

For more information, try ``--help`` or see the :doc:`lstchain_api/index`.


Real data
^^^^^^^^^

Use ``lstchain.scripts.lstchain_data_r0_to_dl1``.

For more information, try ``--help`` or see the :doc:`lstchain_api/index`.


DL1 to DL1a and DL1b
^^^^^^^^^^^^^^^^^^^^

If you already have a DL1 file containing images and parameters (DL1a and DL1b), you can recalculate the parameters
using a different cleaning by using ``lstchain.scripts.lstchain_dl1ab``.

For more information, try ``--help`` or see the :doc:`lstchain_api/index`.


Configuration file
^^^^^^^^^^^^^^^^^^

Here is an example configuration file for this step.

.. toggle:: 

    .. include:: ../lstchain/data/lstchain_standard_config.json
       :code: json


DL1 to DL2
----------

Use ``lstchain.tools.lstchain_dl1_to_dl2`` for real data and MC.

For more information, try ``--help`` or see the :doc:`lstchain_api/index`.

Configuration file
^^^^^^^^^^^^^^^^^^

Here is an example configuration file for this step.

.. toggle:: 

    .. include:: ../lstchain/data/lstchain_standard_config.json
       :code: json


DL2 to DL3
----------

For a quick look into the data and perform :math:`{\theta}^2/{\alpha}` plots from DL2 files,
you can use the notebook:

https://github.com/cta-observatory/cta-lstchain/blob/main/notebooks/explore_DL2.ipynb

or the script: ``lstchain_significance_calculation``

IRF creation
^^^^^^^^^^^^

To write IRF files, you should use ``lstchain.tools.lstchain_create_irf_files``.

For more information, try ``--help`` or see the :doc:`lstchain_api/index`.


Here is an example configuration file for the IRF creation step.

.. toggle:: 

    .. include:: examples/irf_dl3_tool_config.json
        :code: json


Event list creation
^^^^^^^^^^^^^^^^^^^

To write DL3 files, you should use:

- ``lstchain.tools.lstchain_create_dl3_file``
- ``lstchain.tools.lstchain_create_dl3_index_files``

For more information, try ``--help`` or see the :doc:`lstchain_api/index`.

You should use the same configuration file used for the IRF creation (hence you have the same cuts).


Post DL3 analysis
-----------------

You can analyze the resulting DL3 files using Gammapy https://docs.gammapy.org/.

There is a notebook in ``cta-lstchain/notebooks`` that shows how to use Gammapy to perform
a 1D on-off analysis for point-like sources.

https://github.com/cta-observatory/cta-lstchain/blob/main/notebooks/post_DL3_analysis.ipynb

Other analyses require additional packages. For example, for the 3D analysis of extended
sources and creating sky maps, you can create a background model using one of the following packages:

    * https://github.com/mdebony/BAccMod
    * https://github.com/cta-observatory/pybkgmodel

For doing pulsar analysis we suggest using:

    * https://github.com/cta-observatory/PulsarTimingAnalysis
