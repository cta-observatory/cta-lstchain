.. _datachecks:

=========================
Datachecks (`datachecks`)
=========================

.. currentmodule:: lstchain.datachecks

Introduction
============

Module containing functions producing the LST datachecks. Currently the checks are done at the DL1 level.
The DL1 datacheck files are produced using the following scripts:

* :py:obj:`lstchain.scripts.lstchain_check_dl1`

  This takes as input subrun-wise DL1 files, e.g.:

  .. code-block:: bash

    lstchain_check_dl1 --input-file dl1_LST-1.1.Run01881.0000.h5 --output-dir OUTPUT_DIR --omit-pdf


  and produces a subrun-wise file, ``datacheck_dl1_LST-1.Run01881.0000.h5`` which contains many quantities that can be
  used to judge the quality of the data (see class :py:obj:`lstchain.datachecks.containers.DL1DataCheckContainer`)


* :py:obj:`lstchain.scripts.lstchain_check_dl1`

  The same script is used (by providing as input the subrun-wise datacheck files produced above) to produce a run-wise
  datacheck file. It also needs to know where the subrun-wise .fits files (produced in the R0 to DL1 analysis step)
  containing the muon ring information are stored ("MUONS_DIR"):

  .. code-block:: bash

    lstchain_check_dl1 --input-file "datacheck_dl1_LST-1.Run01881.*.h5" --output-dir OUTPUT_DIR --muons-dir MUONS_DIR


  The output is now a run-wise file, ``datacheck_dl1_LST-1.Run01881.h5`` which contains the information from the
  subrun-wise files. It also produces a .pdf file ``datacheck_dl1_LST-1.Run01881.pdf`` with many plots of the quantities
  stored in the DL1DataCheckContainer objects.

* b
* c


Using the datacheck files for selecting good-quality data
=========================================================

Reference/API
=============

.. automodapi:: lstchain.datachecks.containers
   :no-inheritance-diagram:
.. automodapi:: lstchain.datachecks.dl1_checker
   :no-inheritance-diagram:
