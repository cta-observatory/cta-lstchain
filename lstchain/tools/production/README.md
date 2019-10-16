# LST_scripts
Scripts to analyse files on LST cluster at La Palma



## Notes:

### Monte-Carlo analysis

Steps:

- `mc_dl0_to_dl1.py`
    - provide input directory you want to analyse. e.g. `/fefs/aswg/data/mc/DL0/20190909/proton/North_pointing`
    - launch parallel jobs
- `mc_dl1_merge.py`
    - once all jobs from `mc_dl0_to_dl1.py` are finished
    - check that jobs finished without error from the logs
    - merge the DL1 files for training and testing
    - clean and move the `running_analysis`
- `mc_dl1_to_dl2.py`
    - input:
    - reco
    
    
    
### Real Data analysis

- must be done in *real time*:
    - compute calibration coefficient (create calibration file) = script from Pawel and Franca
    - check these calibration files ?!
    - how to pass this file to EventBuilder ? EventBuilder access cp disks ?
    - run flat-field pedestal interleaved --> script produce calibration files

- real_r0_to_dl2.py