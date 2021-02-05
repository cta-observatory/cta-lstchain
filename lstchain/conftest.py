import pytest
import tempfile
from pathlib import Path


# add a marker for the tests that need private data and don't run them
# by default
def pytest_configure(config):
    print(config.option.markexpr)

    config.addinivalue_line(
        "markers", "private_data: mark tests that needs the private test data"
    )

    if 'private_data' not in config.option.markexpr:
        if config.option.markexpr:
            config.option.markexpr += ' and '
        else:
            config.option.markexpr +=  'not private_data'

    print(config.option.markexpr)


@pytest.fixture(scope='session')
def temp_dir():
    with tempfile.TemporaryDirectory(prefix='test_lstchain') as d:
        yield Path(d)
