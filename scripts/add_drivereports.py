import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import sys
from ctapipe.io import HDF5TableWriter,HDF5TableReader
from ctapipe.core import Container, Field, Map
import tables
from tables import *
import ctapipe
from lstchain.io import DL1ParametersContainer
from lstchain.io import DriveReport
import argparse

parser = argparse.ArgumentParser(description="Adding drive reports to DL1 files")
parser.add_argument('--infile')

args = parser.parse_args()

if __name__ == '__main__':

#     filename =
 
       
     # DL1 files
 #    filename='./dl1_LST-1.3.Run00889.0000.fits.h5'
     print(args) 
     if args is  None:
          raise Exception('DL1 data file should be provided')
          
     else:
          filename=args.infile
          print("File used:",filename)
     # The data is considered here following the format in the drive report

     drive_report = DriveReport()
     drive_report.date ='Tue July 2'
     drive_report.time_stamp ='13:59:02'
     drive_report.epoch = 2019
     drive_report.time  = 1561766334
     drive_report.Az_avg = -0.87399
     drive_report.Az_min = -0.873999
     drive_report.Az_max = -0.873999
     drive_report.Az_rms = 0.00336961
     drive_report.El_avg = 95.026
     drive_report.El_min = 95.026 
     drive_report.El_max = 95.026
     drive_report.El_rms = 0.0

# Drive report table
#<date> <timestamp epoch time> Az <average> <min> <max> <rms> El <average> <min> <max> <rms>
#Tue Jul  2 13:59:02 2019 1561766334 Az -0.87399 -0.873999 -0.873999 0.00336961 El 95.026 95.026 95.026 0 RA 0 Dec 0

#NOTE: At present, it just adds the drive report to a DL1 file. However, this is not doing an event-by-event basis. The idea is to take the timing information of the first event and last event of a run (or subrun) and then take the portion of the drive report relevant to that time slot of the run. Adding the pointing information event-wise is part of a separate code which includes also a correction to the pointing information.

     with HDF5TableWriter(filename=filename, group_name="DriveReport", mode="a", overwrite=True) as writer:
           writer.write("drive", drive_report)
           writer.close()



