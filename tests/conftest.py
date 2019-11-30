import pytest


def pytest_addoption(parser):
    parser.addoption("--include_performance_tests", action="store_true",
                     help="Will run the performance tests which do full model testing. This could take a few"
                             "days to fully accomplish.")

@pytest.fixture()
def include_performance_tests(pytestconfig):
    return pytestconfig.getoption("include_performance_tests")


@pytest.fixture()
def skip_performance_check(include_performance_tests):
    if not include_performance_tests:
        pytest.skip('Skipping due to performance argument not specified. Add --include_performance_tests to not skip')
