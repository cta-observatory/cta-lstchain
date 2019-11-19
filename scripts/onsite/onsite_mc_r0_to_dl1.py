#!/usr//bin/env python

## Code to reduce R0 data to DL1 onsite (La Palma cluster)


####################################### OPTIONS #######################################



import sys
import os
import shutil
import random
import argparse
import calendar
import lstchain
from lstchain.io.data_management import *

 

parser = argparse.ArgumentParser(description="R0 to DL1")


parser.add_argument('input_dir', type=str,
                    help='path to the files directory to analyse',
                   )

parser.add_argument('--config_file', '-conf', action='store', type=str,
                    dest='config_file',
                    help='Path to a configuration file. If none is given, a standard configuration is applied',
                    default=None
                    )

parser.add_argument('--pedestal_path', '-pedestal', action='store', type=str,
                    dest='pedestal_path',
                    help='Path to a pedestal file',
                    default=None
                    )

parser.add_argument('--calibration_path', '-calib', action='store', type=str,
                    dest='calibration_path',
                    help='Path to a calibration file',
                    default=None
                    )

parser.add_argument('--train_test_ratio', '-ratio', action='store', type=str,
                    dest='train_test_ratio',
                    help='Ratio of training data',
                    default=0.25
                    )

parser.add_argument('--random_seed', '-seed', action='store', type=str,
                    dest='random_seed',
                    help='Random seed for random processes',
                    default=42,
                    )

parser.add_argument('--n_files_per_dl1', '-nfdl1', action='store', type=str,
                    dest='n_files_per_dl1',
                    help='Number of input files merged in one DL1. If 0, the number of files per DL1 is computed based '
                         'on the size of the DL0 files and the expected reduction factor of 50 '
                         'to obtain DL1 files of ~100 MB. Else, use fixed number of files',
                    default=0,
                    )

today = calendar.datetime.date.today()
default_prod_id = f'{today.year:04d}{today.month:02d}{today.day:02d}_v{lstchain.__version__}_v00'

parser.add_argument('--prod_id', action='store', type=str,
                    dest='prod_id',
                    help="Production ID",
                    default=default_prod_id,
                   )

args = parser.parse_args()


if __name__ == '__main__':

    PROD_ID = args.prod_id
    TRAIN_TEST_RATIO = args.train_test_ratio
    RANDOM_SEED = args.random_seed
    NFILES_PER_DL1 = args.n_files_per_dl1
    
    DESIRED_DL1_SIZE_MB = 100
    
    DL0_DATA_DIR = args.input_dir
    
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
            cc = ' -conf {}'.format(args.config_file) if args.config_file is not None else ' '
            base_cmd = 'slurm_core.sh "lstchain_mc_r0_to_dl1.py -o {} {}"'.format(output_dir, cc)
            cmd = 'sbatch -e {} -o {} {} {}'.format(jobe, jobo, base_cmd, os.path.join(dir_lists, file))

            os.system(cmd)
#             print(cmd)
            counter+=1

        print("{} jobs submitted".format(counter))
    
    # copy this script itself into logs
    shutil.copyfile(sys.argv[0], os.path.join(RUNNING_DIR, os.path.basename(sys.argv[0])))
    # copy config file into logs
    if args.config_file is not None:
        shutil.copy(args.config_file, os.path.join(RUNNING_DIR, os.path.basename(args.config_file)))
    
    # save file lists into logs
    shutil.move('testing.list', os.path.join(RUNNING_DIR, 'testing.list'))
    shutil.move('training.list', os.path.join(RUNNING_DIR, 'training.list'))
    
    print("\n ==== END {} ==== \n".format(sys.argv[0]))       


