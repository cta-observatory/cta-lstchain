.. _datachecks:

=========================
Datachecks (`datachecks`)
=========================

.. currentmodule:: lstchain.datachecks

Introduction
============

Module containing functions for checking the quality of the LST data.

DL1 data checks
===============

Currently the checks are done at the DL1 level.
The DL1 datacheck files are produced by running the following scripts sequentially:

* :py:obj:`lstchain.scripts.lstchain_check_dl1`

  This takes as input a DL1 file (including DL1a information, i.e. camera images & times) from a data subrun, e.g.:

  .. code-block:: bash

    lstchain_check_dl1 --input-file dl1_LST-1.1.Run14619.0000.h5 --output-dir OUTPUT_DIR --omit-pdf


  The script produces a data check file for the subrun, ``datacheck_dl1_LST-1.Run14619.0000.h5`` which contains many
  quantities that can be used to judge the quality of the data (see class :py:obj:`~lstchain.datachecks.containers.DL1DataCheckContainer`)

|

* :py:obj:`lstchain.scripts.lstchain_check_dl1`

  The same script is run again, but now providing as input the **subrun-wise datacheck files** produced earlier (all subruns
  of a given run must be provided). It also needs to know where the subrun-wise ``muons_LST-1.*.fits files`` (produced in
  the R0 to DL1 analysis step) which contain the muon ring information are stored ("MUONS_DIR"):

  .. code-block:: bash

    lstchain_check_dl1 --input-file "datacheck_dl1_LST-1.Run14619.*.h5" --output-dir OUTPUT_DIR --muons-dir MUONS_DIR


  The output is now a data check file for the whole run, ``datacheck_dl1_LST-1.Run14619.h5`` which contains all the
  information from the subrun-wise files. The script also produces a .pdf file ``datacheck_dl1_LST-1.Run14619.pdf`` with
  various plots of the quantities stored in the DL1DataCheckContainer objects, plus others obtained from the muon ring
  analysis. Note that the muon ring information is not propagated to the run-wise datacheck files, it is just used for
  the plotting.

|

* :py:obj:`lstchain.scripts.lstchain_longterm_dl1_check`

  This script merges the run-wise datacheck files of (typically) one night, stored in INPUT_DIR, and produces **a
  single .h5 file for the whole night** as output (e.g. ``DL1_datacheck_20230920.h5``). The "longterm" in the script
  name is a bit of an overstatement - in principle it can be run over data from many nights, but the output html (see
  below) becomes too heavy and some of the interactive features work poorly.

  .. code-block:: bash

    lstchain_longterm_dl1_check --input-dir INPUT_DIR --muons-dir MUONS_DIR --output-file DL1_datacheck_20230920.h5 --batch

  The output .h5 file contains a (run-wise) summarized version of the information in the input files, including the muon
  ring .fits files. It also creates an .html file (e.g. ``DL1_datacheck_20230920.html``) which can be opened with any
  web browser and which contains various interactive plots which allow to make a quick check of the data of a night. See
  an example of the .html file `here <https://www.lst1.iac.es/datacheck/dl1/v0.10/20230920/DL1_datacheck_20230920.html>`_
  (password protected).

|

.. Heading below be replaced by this once script is merged!  :py:obj:`lstchain.scripts.lstchain_cherenkov_transparency`

* ``lstchain_cherenkov_transparency``


  This script analyzes the image intensity histograms (one per subrun) stored in the **run-wise** datacheck files (which
  must exist in INPUT_DIR)

  .. code-block:: bash

    lstchain_cherenkov_transparency --update-datacheck-file DL1_datacheck_20230920.h5 --input-dir INPUT_DIR

  The script updates the **night-wise** datacheck .h5 file ``DL1_datacheck_20230920.h5`` with a new table (with one entry
  per subrun) containing parameters related to the image intensity spectra for cosmic ray events (i.e., a
  Cherenkov-transparency - like approach, see e.g. https://arxiv.org/abs/1310.1639).




Using the datacheck files for selecting good-quality data
=========================================================

The night-wise datacheck .h5 files, ``DL1_datacheck_YYYYMMDD.h5`` can be used to select a subsample of good quality data
from a large sample of observations. The files are relatively light, 6 MB per night in average. A large sample of them
can be processed with the notebook ``cta_lstchain/notebooks/data_quality.ipynb`` (instructions can be found inside the
notebook)


Reference/API
=============

.. automodapi:: lstchain.datachecks.containers
   :no-inheritance-diagram:
.. automodapi:: lstchain.datachecks.dl1_checker
   :no-inheritance-diagram:
