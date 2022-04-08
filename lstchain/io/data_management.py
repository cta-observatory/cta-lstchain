# functions used for LST onsite analysis data management

import os
import shutil
import sys


__all__ = [
    'check_and_make_dir',
    'check_data_path',
    'check_job_logs',
    'get_input_filelist',
    'query_continue',
    'query_yes_no',
]


def str_to_bool(answer):
    if answer.lower() in {'y', 'yes'}:
        return True

    if answer.lower() in {'n', 'no'}:
        return False

    raise ValueError('Invalid choice, use one of [y, yes, n, no]')


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

    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
        default = True
    elif default == "no":
        prompt = " [y/N] "
        default = False
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        sys.stdout.write(question + prompt)
        choice = input().lower()
        if choice == '' and default is not None:
            return default
        else:
            try:
                return str_to_bool(choice)
            except ValueError:
                print(
                    "Please respond with 'yes' or 'no' (or 'y' or 'n').",
                    file=sys.stderr,
                )


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

