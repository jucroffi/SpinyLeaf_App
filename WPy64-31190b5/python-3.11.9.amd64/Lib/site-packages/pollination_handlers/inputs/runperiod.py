"""Handlers to check ladybug analysis period inputs."""
from ladybug.analysisperiod import AnalysisPeriod


def run_period_to_str(value):
    """Translate an AnalysisPeriod into a string.

    Args:
        value: Either an AnalysisPeriod object or a string representation
            of an AnalysisPeriod.

    Returns:
        str -- string version of the run period.
    """
    if value == '' or value is None:
        r_per, r_per_str = AnalysisPeriod(), ''
    elif isinstance(value, str):  # ensure the string is of an appropriate type
        r_per, r_per_str = AnalysisPeriod.from_string(value), value
    elif isinstance(value, AnalysisPeriod):
        r_per, r_per_str = value, str(value)
    else:
        raise ValueError('Run period must be a string or an AnalysisPeriod '
                         'object. Got {}.'.format(type(value)))
    assert r_per.timestep == 1, 'Run period must be for a timestep of 1. '  \
        'Finer timesteps are not yet supported.'
    return r_per_str
