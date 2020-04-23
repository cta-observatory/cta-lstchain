# functions used for LST onsite analysis data management

import sys
import os
from distutils.util import strtobool
import shutil

__all__ = [
    'query_yes_no',
    'query_continue',
    'check_data_path',
    'get_input_filelist',
    'check_and_make_dir',
    'check_job_logs',
]

def query_yes_no(question, default="yes"):
    """
    Ask a yes/no question via raw_input() and return their answer.

    Parameters
    ----------
    question: str
        question to the user
    default: str - "yes", "no" or None
        resumed answer if the user just hits <Enter>.
        "yes" or "no" will set a default answer for the user
        None will require a clear answer from the user
    Returns
    -------
    bool - True for "yes", False for "no"
    """
    valid = {"yes": True, "y": True, "ye": True,
             "no": False, "n": False}
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        sys.stdout.write(question + prompt)
        choice = input().lower()
        if default is not None and choice == '':
            return valid[default]
        else:
            try:
                return bool(strtobool(choice))
            except:
                sys.stdout.write("Please respond with 'yes' or 'no' "
                                 "(or 'y' or 'n').\n")

                
def query_continue(question, default="no"):
    """
    Ask a question and if the answer is no, exit the program.
    Calls `query_yes_no`.

    Parameters
    ----------
    question: str
    default: str

    Returns
    -------
    bool - answer from query_yes_no
    """
    answer = query_yes_no(question, default=default)
    if not answer:
        sys.exit("Program stopped by user")
    else:
        return answer



def check_data_path(data_path):
    """
    Check if the path to some data exists.
    Raise an Error if the path does not exist, is not a directory or does not contain data.

    Parameters
    ----------
    data_path: str
    """
    if not os.path.exists(data_path):
        raise ValueError("The input directory must exist")
    if get_input_filelist(data_path) == []:
        raise ValueError("The input directory is empty")


def get_input_filelist(data_path):
    """
    Return list of files in `data_path`

    Parameters
    ----------
    data_path: str

    Returns
    -------
    list of str
    """
    return [os.path.abspath(os.path.join(data_path, f)) for f in os.listdir(data_path) if os.path.isfile(os.path.join(data_path, f))]


def check_and_make_dir(dir):
    """
    Check if a directory exists or contains data before to makedir.
    If exists, query the user to remove its content.

    Parameters
    ----------
    dir: str
        path to a directory
    """
    if os.path.exists(dir) and os.listdir(dir)!=[]:
        clean = query_yes_no("The directory {} is not empty. Do you want to remove its content?".format(dir), default='yes')
        if clean:
            shutil.rmtree(dir)
    os.makedirs(dir, exist_ok=True)

    
def check_job_logs(job_logs_dir):
    """
    Check all the job logs named `*.e` for Errors
    Query the user to continue or not in case of errors.
    If not, the program exits.

    Parameters
    ----------
    job_logs_dir: str
        path to the directory including job logs
    """
    job_logs = [os.path.join(job_logs_dir, f) for f in os.listdir(job_logs_dir) if f.endswith('.e')]
    logs_with_error = []
    for log_filename in job_logs:
        with open(log_filename) as log_file:
            for line in log_file.readlines():
                if 'Error' in line:
                    logs_with_error.append(os.path.basename(log_filename))
                    break
    if not logs_with_error == []:
        query_continue("There are errors in the following log files:\n {}"
                       "\n Are you sure you want to continue?".format(logs_with_error), default="no")

