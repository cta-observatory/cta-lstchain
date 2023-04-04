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
|      | with corresponding IRFs (PSF, EDISP, BKG, etc.)                           |             |
+------+---------------------------------------------------------------------------+-------------+


Analysis steps
==============

R1 to DL1
---------
Usage of

``lstchain.scripts.lstchain_data_r0_to_dl1``

for real data and

``lstchain.scripts.lstchain_mc_r0_to_dl1``

for MC.

if you already have a DL1 file containing images and parameters (DL1a and DL1b), you can recalculate the parameters
using a different cleaning by using:
``lstchain.scripts.lstchain_dl1ab``


DL1 to DL2
----------

Usage of

``lstchain.scripts.lstchain_dl1_to_dl2``

for real data and MC

DL2 to DL3
----------

To write DL3 and IRF files, you should use:
``lstchain.tools.lstchain_create_dl3_file``
``lstchain.tools.lstchain_create_dl3_index_files``
``lstchain.tools.lstchain_create_irf_files``

and analyze the results using ``gammapy``

For a quick look into the data and perform :math:`{\theta}^2/{\alpha}` plots, you can also use:
``lstchain.scripts.lstchain_post_dl2``

