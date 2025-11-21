"""Handlers to interpret point-in-time metrics to correct sting inputs."""
_BOOL_STRINGS = ('True', 'False')


def point_in_time_metric_to_str(value):
    """Translate a string of any acceptable metric inputs to a recipe input.

        Args:
            value: Either an integer or text to indicate the following.

            * 0 = illuminance
            * 1 = irradiance
            * 2 = luminance
            * 3 = radiance

        Returns:
            str -- point in time recipe text.
    """
    acceptable_dict = {
        '0': 'illuminance',
        '1': 'irradiance',
        '2': 'luminance',
        '3': 'radiance',
        'illuminance': 'illuminance',
        'irradiance': 'irradiance',
        'luminance': 'luminance',
        'radiance': 'radiance'
    }
    try:
        return acceptable_dict[value.lower()]
    except KeyError:
        raise ValueError('Point in time metric "{}" is not recognized.'.format(value))
