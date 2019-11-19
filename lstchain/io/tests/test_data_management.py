from lstchain.io.data_management import *

def test_check_prod_id():
    path1 = '/go/to/path1/abcd/'
    path2 = 'p2/abcd'
    check_prod_id(path1, path2)
