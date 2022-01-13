.. _image:

===============
Image (`image`)
===============

.. currentmodule:: lstchain.image

Introduction
============ 

Module containing image processing customly implemented in lstchain. This includes:

* Dynamic image cleaning.
* Noise adder at DL1 level to mimic real NSB conditions.
* Muon fit and analysis.



Muon analysis
=============

IACTs use camera calibration systems to estimate the conversion between the measured signal
and the total number of received photons. This method, does not take into account effects
of the optical system and the efficiency of the mirror of the telescope is not considered. To carry
out an absolute calibration of light throughput, it is essential to analyze a signal of known nature,
as for example the peculiar ring-shaped images produced by cosmic ray muons.

To perform this analysis:

1. We fit ellipses to the bright images in the camera and extract the fitted parameters such as the image ``intensity``
or the ring ``width``

2. We compare the fit results with those obtained using MC simulations and different optical efficiencies.

3. The overall optical efficiency of the telescope corresponds to the fitted value that is closer to

Muon analysis is automatically performed by LSTOSA when running `lstchain.scripts.lstchain_data_r0_to_dl1`.
You can find the table with the outputs in muons_LST-1.Runxxxxx.yyyy.fits, where xxxxx is the run number and
yyyy the subrun number analyzed.

If you want to perform a custom muon analysis, you can run the script `lstchain.scripts.lstchain_dl1_muon_analysis`,
that performs the analysis of DL1 files and writes out a table with the results for the fitted muons. In order to
reduce the differences between the analysis of Real data and MC, muon analysis uses a different signal extractor
(`GlobalPeakWindowSum`) than real data (`LocalPeakWindowSum`), that can be specified in the `image_extractor_for_muons`
keyword from the standard input card located in `lstchain.data.lstchain_standard_config.json`

Muon analysis can also be performed to check the Point Spread Function of the telescope, although small changes in
the point spread function may not be spotted using it.

Reference/API
=============

.. automodapi:: lstchain.image
   :no-inheritance-diagram:
.. automodapi:: lstchain.image.cleaning
   :no-inheritance-diagram:
.. automodapi:: lstchain.image.modifier
   :no-inheritance-diagram:
.. automodapi:: lstchain.image.muon.muon_analysis
   :no-inheritance-diagram:
.. automodapi:: lstchain.image.muon.plot_muon
   :no-inheritance-diagram:
