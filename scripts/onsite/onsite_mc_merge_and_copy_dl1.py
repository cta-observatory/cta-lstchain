#!/usr//bin/env python3

# T. Vuillaume, 12/09/2019
# merge and copy DL1 data after production


# 1. check job_logs
# 2. check that all files have been created in DL1 based on training and testing lists
# 3. move DL1 files in final place
# 4. merge DL1 files
# 5. move running_dir 


import os
import sys
from lstchain.io.data_management import *
from lstchain.io import smart_merge_h5files


input_dir = sys.argv[1]

JOB_LOGS = os.path.join(input_dir, 'job_logs')
training_filelist = os.path.join(input_dir, 'training.list')
testing_filelist = os.path.join(input_dir, 'testing.list')
running_DL1_dir = os.path.join(input_dir, 'DL1')
DL1_training_dir = os.path.join(running_DL1_dir, 'training')
DL1_testing_dir = os.path.join(running_DL1_dir, 'testing')
final_DL1_dir = input_dir.replace('running_analysis', 'DL1')
logs_destination_dir = input_dir.replace('running_analysis', 'analysis_logs')



def check_files_in_dir_from_file(dir, file):
    """
    Check that a list of files from a file exist in a dir

    Parameters
    ----------
    dir
    file

    Returns
    -------

    """
    with open(file) as f:
        lines = f.readlines()

    files_in_dir = os.listdir(dir)
    files_not_in_dir = []
    for line in lines:
        filename = os.path.basename(line.rstrip('\n'))
        if filename not in files_in_dir:
            files_not_in_dir.append(filename)

    return files_not_in_dir


def readlines(file):
    with open(file) as f:
        lines = [line.rstrip('\n') for line in f]
    return lines

def move_dir_content(src, dest):
    files = os.listdir(src)
    for f in files:
        shutil.move(os.path.join(src,f), dest)
    os.rmdir(src)
    

print("\n ==== START {} ==== \n".format(sys.argv[0]))

# 1. check job logs
check_job_logs(JOB_LOGS)


# 2. check that all files have been created in DL1 based on training and testing lists
## just check number of files first:
if not len(os.listdir(DL1_training_dir)) == len(readlines(training_filelist)):
    tf = check_files_in_dir_from_file(DL1_training_dir, training_filelist)
    if  tf != []:
        query_continue("{} files from the training list are not in the `DL1/training` directory:\n{} "
                     "Continue ?".format(len(tf),tf))
        
if not len(os.listdir(DL1_testing_dir)) == len(readlines(testing_filelist)):
    tf = check_files_in_dir_from_file(DL1_testing_dir, testing_filelist)
    if tf != []:
        query_continue("{} files from the testing list are not in the `DL1/training` directory:\n{} "
                     "Continue ?".format(len(tf), tf))

# 3. merge DL1 files
for t in ['testing', 'training']:
    output_filename = 'dl1_'
    for i in [-4, -3, -2, -1]:
        output_filename += running_DL1_dir.split('/')[i]
        output_filename += '_'
    output_filename += t
    output_filename += '.h5'

    filelist = [running_DL1_dir + f for f in os.listdir(os.path.join(running_DL1_dir, t))]
    smart_merge_h5files(filelist, output_filename)


# 4. move DL1 files in final place
check_and_make_dir(final_DL1_dir)
move_dir_content(running_DL1_dir, final_DL1_dir)
print("DL1 files have been moved in {}".format(final_DL1_dir))

# 5. move running_dir as logs
check_and_make_dir(logs_destination_dir)
move_dir_content(input_dir, logs_destination_dir)
print("LOGS have been moved to {}".format(logs_destination_dir))

print("\n ==== END {} ==== \n".format(sys.argv[0]))