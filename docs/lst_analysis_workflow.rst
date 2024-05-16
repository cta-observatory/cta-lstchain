============================
LST-1 data analysis workflow
============================

Pipeline
========

Here is a simplified version of the LST-1 data analysis pipeline and the different tools used for each step:

.. this image can be modified going to mermaid.live and loading it using its URL in Actions/LOAD GIST
.. image:: https://mermaid.ink/img/pako:eNptkk2LwjAQhv9KyGkXLOh662EvusKCXuqxKTLbjlrIR0lSliL-9500DXTVHNJ8PO87M-nceG0a5Dk_S_NbX8F6ti-EFprRKJYnVb8d-5-Lhe76zrJMOq_qru0wyz6ZIqV0ZbGbVtVM9chu9x-nw6akDztsqhTATdZRUxbLeMmmcQGloBxnIXRnjTeaFiix9tboiUTdJL8YJcamYlodQn8XO5eAmOgDEFQNeBizswiShV31rHgsZ_ZOQREh44CI7X6VPFfJLuW4mtFPScxLeYmtJyxAoTQCxhfqhnAPGuTgWpdM1snkJcMXXKFV0DbUAbegEdxfUaHgOS0bPEMvveBC3wmF3pvjoGuee9vjgvcdeeO2BfqH6v_hV9N6Y3l-BunoEMftIXba2HD3P0O7zCs?type=png)](https://mermaid.live/edit#pako:eNptkk2LwjAQhv9KyGkXLOh662EvusKCXuqxKTLbjlrIR0lSliL-9500DXTVHNJ8PO87M-nceG0a5Dk_S_NbX8F6ti-EFprRKJYnVb8d-5-Lhe76zrJMOq_qru0wyz6ZIqV0ZbGbVtVM9chu9x-nw6akDztsqhTATdZRUxbLeMmmcQGloBxnIXRnjTeaFiix9tboiUTdJL8YJcamYlodQn8XO5eAmOgDEFQNeBizswiShV31rHgsZ_ZOQREh44CI7X6VPFfJLuW4mtFPScxLeYmtJyxAoTQCxhfqhnAPGuTgWpdM1snkJcMXXKFV0DbUAbegEdxfUaHgOS0bPEMvveBC3wmF3pvjoGuee9vjgvcdeeO2BfqH6v_hV9N6Y3l-BunoEMftIXba2HD3P0O7zCs
    :width: 1080
    :align: center
    :alt: LST-1 data analysis pipeline


The steps marked with lstchain must be run by the analysers using the corresponding lstchain command (see :ref:`the Analysis Steps <introduction>`) .
Other tools are described below.


lstmcpipe
---------
lstmcpipe handles the analysis of MC data on the cluster at La Palma. 
It produces the trained models and the MC DL2 files required for data analysis.
Its usage is recommended in order to:
- avoid common mistakes and pitfalls
- do not worry about resources management for this part of the analysis which requires specific needs
- have a reproducible analysis
- have common analysis production and therefore save computing resources

If you need a specific MC analysis production, you can request one by opening a pull-request on the GitHub repository.

repository: https://github.com/cta-observatory/lstmcpipe

documentation: https://cta-observatory.github.io/lstmcpipe

lstosa
------
lstosa handles the analysis of real data on the cluster at La Palma.
It automatically produces the DL1 files required for data analysis.
Analysers should not need to use this package directly (it is maintained and handled by LSTOSA team), but use the DL1 files it produces.

repository: https://github.com/cta-observatory/lstosa    

documentation: https://lstosa.readthedocs.io/


Files and configuration
=======================

The DL1 files to use obviously depend on the source you want to analyse.
Unless you have a good reason, the latest version of the DL1 files should be used.

Selection of the real-data DL1 files
------------------------------------

The selection of the DL1 files (run-wise, i.e. those produced by lstosa by merging the subrun-wise DL1 files) for a
specific analysis (i.e., a given source and time period) can be performed using the notebook
``cta_lstchain/notebooks/data_quality.ipynb``. The selection also takes into account the quality of the data, mostly in
terms of atmospheric conditions - evaluated using the rate of cosmic-ray showers as a proxy. Data taken under poor
conditions will not be included in the list of selected runs. Instructions and further details can be found inside the
notebook.


RF models
---------

The list of available trained RF models can be found in the lstmcpipe documentation and production list, 
along with a description of each production:
https://cta-observatory.github.io/lstmcpipe/productions.html

The RF models are stored in the following directory:
``/fefs/aswg/data/models/...``


Tuning of MC DL1 files and RF models
------------------------------------

The default MC production is generated with a level of noise in the images which corresponds to the level of diffuse
night-sky background ("NSB") in a "dark" field of view (i.e. for observations with moon below the horizon, at not-too-low
galactic latitudes and not affected by other sources of noise, like the zodiacal light). In general, observations of
**extragalactic** sources in dark conditions can be properly analyzed with the default MC (i.e. with the standard RF models).

The median of the standard deviation of the pixel charges recorded in interleaved pedestal events (in which  a camera
image is recorded in absence of a physics trigger) is a good measure of the NSB level in a given data run. This is computed
by the data selection notebook ``cta_lstchain/notebooks/data_quality.ipynb`` (see above). For data with an NSB level
significantly higher than the "dark field" one, it is possible to tune (increase) the noise in the MC files, and produce
from them RF models (and "test MC" for computing instrument response functions) which improve the performance of the
analysis (relative to using the default, low-NSB MC).

This is done by changing the configuration file for the MC processing, producing new DL1(b) files, and training new RF models.
To produce a config tuned to the data you want to analyse, you first have to obtain the standard analysis configuration
file (for MC) for the desired version of lstchain (= the version with which the real DL1 files you will use were produced).
This can be done with the script :py:obj:`~lstchain.scripts.lstchain_dump_config`:

.. code-block::

    lstchain_dump_config --mc --output-file standard_lstchain_config.json

Now you have to update the file with the parameters needed to increase the NSB level. For this you need a simtel.gz MC
file from the desired production (any will do, it can be either a gamma or a proton file), and a "typical" subrun DL1 file
from your **real data**  sample. "Typical" means one in which the NSB level is close to the median value for the sample
to be analyzed. The data selection notebook ``cta_lstchain/notebooks/data_quality.ipynb`` (see above) provides a list of
a few such subruns for your selected sample - you can use any of them. To update the config file you have to use the
script :py:obj:`~lstchain.scripts.lstchain_tune_nsb` , e.g. :

.. code-block::

    lstchain_tune_nsb.py --config standard_lstchain_config.json \
                         --input-mc .../simtel_corsika_theta_6.000_az_180.000_run10.simtel.gz \
                         --input-data .../dl1_LST-1.Run10032.0069.h5 \
                         -o tuned_nsb_lstchain_config.json

To request a **new production of RF models**, you can open a pull-request on the lstmcpipe repository, providing
the .json configuration file produced following the steps above.


Keeping track of lstchain configurations
----------------------------------------

The lstchain configuration file used to process the  simulations and produce the RF models of a given MC production is
provided in the lstmcpipe repository, as well as in the models directory.

It is important that the software version, and the configuration used for processing real data and MC are the same. For a
given lstchain version, this should be guaranteed by following the procedure above which makes use of
:py:obj:`~lstchain.scripts.lstchain_dump_config`.


DL3/IRF config files
--------------------


An example config file for IRF/DL3 creation is provided in `docs/example`:

  - `irf_dl3_tool_config.json <irf_dl3_tool_config.json>`_


Such files should be used to produce DL3 files and IRFs from DL2 (see :ref:`the Analysis Steps <introduction>`)

Environment
===========

Conda environment are used to manage the dependencies of the analysis on the cluster at La Palma.
**It is recommended to use the same environment for all the analysis steps.**

To use the main conda, add this to your `.bashrc`:

.. code-block::
    :caption: conda setup

    # >>> conda initialize >>>
    # !! Contents within this block are managed by 'conda init' !!
    __conda_setup="$('/fefs/aswg/software/virtual_env/anaconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
    if [ $? -eq 0 ]; then
        eval "$__conda_setup"
    else
        if [ -f "/fefs/aswg/software/virtual_env/anaconda3/etc/profile.d/conda.sh" ]; then
            . "/fefs/aswg/software/virtual_env/anaconda3/etc/profile.d/conda.sh"
        else
            export PATH="/fefs/aswg/software/virtual_env/anaconda3/bin:$PATH"
        fi
    fi
    unset __conda_setup
    # <<< conda initialize <<<


Then, you can use the conda environment used to produce the MC files:

.. code-block::

    conda activate /fefs/aswg/software/conda/envs/ENV_NAME


The `ENV_NAME` used for MC production is provided in the lstmcipe config file.


Note: you may also activate the environment defined here using your own conda installation.
