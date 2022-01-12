.. _introduction:

Introduction
============

``lstchain`` is the analysis library for the observed and simulated LST-1 data.

Dependencies
------------
``lstchain`` heavily depends on ``ctapipe``

Data levels
===========

* R0
* R1
* DL1a
* DL1b
* DL2
* DL3

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
