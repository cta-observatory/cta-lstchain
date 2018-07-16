This directory is for keeping the development vesrion of analysis code for MAGIC and LST joint data analysis

In order to feed MAGIC data in ctapipe we need to convert MAGIC DL1 data to a specific format. Currently we are converting MAGIC DL1 root files to HDF5 format. The data being stored in the HDF5 files are: 1) image charge, 2) photon arraival time in each pixel, 3) stereo trigger number, 4) arrival time of the event, 5) pointing positions of the telescope

The convertions of the MAGIC DL1 root files are done in two steps

**Step 1:**
 * Addding pointing position container to MAGIC DL1 root files

**Step 2:**
 *  Converting the files obtained in **Step 1** to HDF5 format
 *  Then remove the files created in **Step 1**.

Further details:

Since MAGIC DL1 root files doesn't contain the pointing information for all events, we need to add first that information to the root files. Then we can convert the file to HDF5 format. One can skip step 1 and do every thing in one single step.  However, one may need to work a bit more to do so. Since it was easier for us to proceed with these two-step procedure, we will move ahead with this. Any smart changes are always welcome!


**Code for step 1:**

I have one c macro, once shell script and one python script. To run the code MARs library is required. 
  * add_pointingpos.cc
  * script_addpointingpos.sh
  * script_highlevel.py

All scripts should be kept at same directory. One should only run the python script changing the location of data directory and output directory. The python script works on all *_Y_* files of the data directory. **Note**: Data directory and output directory should be different. 


**Code for step 2:**
 * convert.py
 * script_convert.py
 
 Covertion can be made on all subruns separately. In general we would like to have one DL1 file for one run. Hence we prefer to add all subruns and then write one DL1 file in HDF5 format for all available subruns. One should only run script_convert.py file changing the location of the directory where the files (modified *_Y_*) are stored. After the converstion the modified *Y_* files will be removed to save space.
 

