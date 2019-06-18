from lstchain.io import config


def test_get_standard_config():
    config.get_standard_config()

def test_replace_config():
    a = dict(toto=1, tata=2)
    b = dict(tata=3, tutu=4)
    c = config.replace_config(a, b)
    c["toto"] == 1
    c["tata"] == 3
    c["tutu"] == 4