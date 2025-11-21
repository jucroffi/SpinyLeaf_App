"""Handlers for wea file."""
import os

from ladybug.epw import EPW
from ladybug.wea import Wea
from ladybug.dt import DateTime

from .helper import get_tempfile


def wea_handler(wea_obj):
    """Translate a Wea object to a wea file.

        Args:
            wea_obj: Either a Wea python object or the path to a wea or an epw file.
                In case the wea_obj is a path to wea file it will be returned as is.
                For epw files they will be converted to an annual wea.

        Returns:
            str -- Path to a wea file.
    """

    if isinstance(wea_obj, str):
        if not os.path.isfile(wea_obj):
            raise ValueError('Invalid file path: %s' % wea_obj)
        if wea_obj.lower().endswith('.wea'):
            wea_file = wea_obj
        elif wea_obj.lower().endswith('.epw'):
            # translate epw to wea
            wea = Wea.from_epw_file(wea_obj)
            file_path = get_tempfile('wea', _wea_file_name(wea))
            wea_file = wea.write(file_path)
        else:
            raise ValueError(
                'File path should end with wea or epw not %s' % wea_obj.split('.')[-1]
            )
    elif isinstance(wea_obj, Wea):
        file_path = get_tempfile('wea', _wea_file_name(wea_obj))
        wea_file = wea_obj.write(file_path)
    elif isinstance(wea_obj, EPW):
        file_path = get_tempfile('wea', _wea_file_name(wea_obj))
        wea_file = wea_obj.to_wea(file_path)
    else:
        raise ValueError(
            'Wea input should be a string, a Wea object, or an EPW object. '
            'Not {}.'.format(type(wea_obj))
        )
    return wea_file


def wea_handler_timestep_check(wea_obj):
    """Translate a Wea object to a wea file while checking to be sure the timestep is 1.

        Args:
            wea_obj: Either a Wea python object or the path to a wea or an epw file.
                In case the wea_obj is a path to wea file it will be returned as is.
                For epw files they will be converted to an annual wea.

        Returns:
            str -- Path to a wea file.
    """
    if isinstance(wea_obj, Wea):
        assert wea_obj.timestep == 1, 'Wea timestep must be 1 for this recipe.'
    return wea_handler(wea_obj)


def wea_handler_timestep_annual_check(wea_obj):
    """Translate a Wea object to a wea file while checking to be sure the timestep is 1
    and the Wea is annual.

        Args:
            wea_obj: Either a Wea python object or the path to a wea or an epw file.
                In case the wea_obj is a path to wea file it will be returned as is.
                For epw files they will be converted to an annual wea.

        Returns:
            str -- Path to a wea file.
    """
    wea_file = wea_handler(wea_obj)
    wea_obj = Wea.from_file(wea_file)
    assert wea_obj.timestep == 1, 'Wea timestep must be 1 for this recipe.'
    assert wea_obj.is_annual, 'Wea must be annual for this recipe.'
    return wea_file


def _wea_file_name(wea_obj):
    """Generate a file name from a Wea object."""
    try:
        dts = wea_obj.datetimes
    except AttributeError:  # it's an EPW object
        dts = (DateTime(1, 1, 0), DateTime(12, 11, 23))
    wea_obj.location.city = ''.join(i for i in wea_obj.location.city if ord(i) < 128)
    return '{}_{}_{}'.format(wea_obj.location.city, dts[0].int_hoy, dts[-1].int_hoy)
