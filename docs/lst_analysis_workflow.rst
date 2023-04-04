============================
LST-1 data analysis workflow
============================

Pipeline
========

Here is a simplified version of the LST-1 data analysis pipeline and the different tools used for each step:

.. mermaid::
    :alt: LST-1 data analysis pipeline
    :caption: LST-1 data analysis pipeline

    flowchart LR

        R0_mc(Subgraph) --lstmcpipe--> models[RF models]
        R0_mc --lstmcpipe--> DL2_MC[DL2 MC]

        subgraph R0_mc[R0 MC]
            gamma[gamma\nproton\nelectron]
        end

        DL2_MC --lstchain--> IRFs

        models --lstchain--> DL2_data[DL2 real data]
        models --lstmcpipe--> DL2_MC

        R0_data --lstosa-->DL1_data[DL1 data]

        DL1_data --lstchain--> DL2_data

        DL2_data --lstchain--> DL3_data
        IRFs --gammapy--> analysis
        DL3_data --gammapy--> analysis




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

RF models
---------

The list of available trained RF models can be found in the lstmcpipe documentation and production list, 
along with a description of each production:
https://cta-observatory.github.io/lstmcpipe/productions.html

The RF models are stored in the following directory:
``/fefs/aswg/data/models/...``


Tuning of DL1 files and RF models
---------------------------------

In case of high NSB in the data, it is possible to tune the DL1 files and the RF models to improve the performance of the analysis.      
This is done by changing the `config` file of the RF models and producing new DL1 files and training new RF models.
To produce a config tuned to the data you want to analyse, you may run ``lstchain_tune_nsb`` function that will produce a ``tuned_nsb_config.json`` file.

To request a new production of RF models, you can open a pull-request on the lstmcpipe repository, producing a complete MC config, using:

.. code-block::

    lstchain-dump-config --mc --update-with tuned_nsb_config.json --output-file PATH_TO_OUTPUT_FILE


lstchain config
---------------

The lstchain config used to produce the RF models of a production is provided in the lstmcpipe repository, as well as in the models directory.
It is a good idea to use the same config for the data analysis.
You can also produce a config using `lstchain-dump-config`.


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
    :caption: conda activate

    conda activate /fefs/aswg/software/conda/envs/ENV_NAME


The `ENV_NAME` used for MC production is provided in the lstmcipe config file.


Note: you may also activate the environment defined here using your own conda installation.