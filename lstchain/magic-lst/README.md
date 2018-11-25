This directory is for keeping the development version of analysis code for MAGIC and LST joint data analysis

In order to feed MAGIC data in ctapipe we need to convert MAGIC DL1 data to a specific format. Currently we are converting MAGIC DL1 root files to HDF5 format. The data being stored in the HDF5 files are: 1) image charge, 2) photon arraival time in each pixel, 3) stereo trigger number, 4) arrival time of the event, 5) pointing positions of the telescope

The convertions of the MAGIC DL1 root files are done in two steps

# Step 1:
Addding pointing position container to MAGIC DL1 root files

# step 2:
1. Converting the files obtained in "Step 1" to HDF5 format
2. Then remove the files created in step 1.

Further details:

Since MAGIC DL1 root files doesn't contain the pointing information for all events, we need to add first that information to the root files. Then we can convert the file to HDF5 format. One can skip step one and do every thing in one step.  However, one may need to work a bit more to do so. Since it was easier for us to proceed with these two-step procedure, we will move ahead with this. Any smart changes is always welcome!
