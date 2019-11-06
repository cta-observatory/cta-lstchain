#!/usr//bin/env python

# T. Vuillaume, 20/06/2019

# Choose the selected options


## TODO:
#   - clean the code
#   - prints --> logs


####################################### OPTIONS #######################################

## env ##
# source / fefs / aswg / software / scripts / source_env.sh
# lstchain_repo = "/fefs/home/thomas.vuillaume/software/cta-observatory/cta-lstchain"


import sys
import os
import shutil
import random

from lstchain.io.data_management import *



DL0_DATA_DIR = sys.argv[1]




if __name__ == '__main__':

    BASE_DIR = '/fefs/aswg/'
    PROD_ID = '20190923'
    TRAIN_TEST_RATIO = 0.25
    RANDOM_SEED = 42
    NFILES_PER_DL1 = 0  # IF 0, the number of files per DL1 is computed based on the size of the DL0 files and the expected reduction factor of 50. Else, use fixed number of files.
    
    DESIRED_DL1_SIZE_MB = 100
    
    
    DL0_DATA_DIR = sys.argv[1]
    
    print("\n ==== START {} ==== \n".format(sys.argv[0]))
    
    print("Working on DL0 files in {}".format(DL0_DATA_DIR))
    
    check_data_path(DL0_DATA_DIR)

    # make_output_data_dirs(DL0_DATA_DIR)

    raw_files_list = get_input_filelist(DL0_DATA_DIR)
    
    if NFILES_PER_DL1 == 0:
        size_dl0 = os.stat(raw_files_list[0]).st_size/1e6
        reduction_dl0_dl1 = 50
        size_dl1 = size_dl0/reduction_dl0_dl1
        NFILES_PER_DL1 = max(1, int(DESIRED_DL1_SIZE_MB/size_dl1))
    
    random.seed(RANDOM_SEED)
    random.shuffle(raw_files_list)

    number_files = len(raw_files_list)
    ntrain = int(number_files * TRAIN_TEST_RATIO)
    ntest = number_files - ntrain

    training_list = raw_files_list[:ntrain]
    testing_list = raw_files_list[ntrain:]

    print("{} raw files".format(number_files))
    print("{} files in training dataset".format(ntrain))
    print("{} files in test dataset".format(ntest))

    with open('training.list', 'w+') as newfile:
        for f in training_list:
            newfile.write(f)
            newfile.write('\n')

    with open('testing.list', 'w+') as newfile:
        for f in testing_list:
            newfile.write(f)
            newfile.write('\n')


    RUNNING_DIR = os.path.join(DL0_DATA_DIR.replace('DL0', 'running_analysis'), PROD_ID)

    # LOG_DIR = os.path.join(DL0_DATA_DIR.replace('DL0', 'analysis_logs'), PROD_ID) - this one should be handled by the cleaning script
    JOB_LOGS = os.path.join(RUNNING_DIR, 'job_logs')
    # DIR_LISTS_BASE = os.path.join(RUNNING_DIR, 'file_lists')
    DL1_DATA_DIR = os.path.join(RUNNING_DIR, 'DL1')
    # ADD CLEAN QUESTION
    
    print("RUNNING_DIR: ", RUNNING_DIR)
    print("JOB_LOGS DIR: ", JOB_LOGS)
    print("DL1 DATA DIR: ", DL1_DATA_DIR)
    
    for dir in [RUNNING_DIR, DL1_DATA_DIR, JOB_LOGS]:
        check_and_make_dir(dir)

    ## dumping the training and testing lists and spliting them in sublists for parallel jobs

    for l in 'training', 'testing':
        if l == 'training':
            list = training_list
        else:
            list = testing_list
        dir_lists = os.path.join(RUNNING_DIR, 'file_lists_'+l)
        output_dir = os.path.join(RUNNING_DIR, 'DL1')
        output_dir = os.path.join(output_dir, l)
        check_and_make_dir(dir_lists)
        check_and_make_dir(output_dir)
        print("output dir: ", output_dir)

        number_of_sublists = len(list)//NFILES_PER_DL1+int(len(list)%NFILES_PER_DL1>0)
        for i in range(number_of_sublists):
            output_file = os.path.join(dir_lists, '{}_{}.list'.format(l, i))
            print("dir_lists:", dir_lists)
            with open(output_file, 'w+') as out:
                for line in list[i*NFILES_PER_DL1:NFILES_PER_DL1*(i+1)]:
                    out.write(line)
                    out.write('\n')
            print('{} files generated for {} list'.format(number_of_sublists, l))


        ### LSTCHAIN ###
        counter = 0

        for file in os.listdir(dir_lists):
            jobo = os.path.join(JOB_LOGS, "job{}.o".format(counter))
            jobe = os.path.join(JOB_LOGS, "job{}.e".format(counter))
            cmd = 'sbatch -e {} -o {} lstchain_core.sh {} {}'.format(
                jobe,
                jobo,
                os.path.join(dir_lists, file),
                output_dir,
            )
            os.system(cmd)
            print(cmd)
            counter+=1

        print("{} jobs submitted".format(counter))
        
    shutil.copyfile(sys.argv[0], os.path.join(RUNNING_DIR, sys.argv[0]))
    shutil.move('testing.list', os.path.join(RUNNING_DIR, 'testing.list'))
    shutil.move('training.list', os.path.join(RUNNING_DIR, 'training.list'))
    
    print("\n ==== END {} ==== \n".format(sys.argv[0]))       


