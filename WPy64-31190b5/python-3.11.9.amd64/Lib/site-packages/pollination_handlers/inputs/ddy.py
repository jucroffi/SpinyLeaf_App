"""Handlers for ddy file."""
import os

from ladybug.epw import EPW
from ladybug.ddy import DDY

from .helper import get_tempfile


def ddy_handler(ddy_obj):
    """Translate a DDY object to a ddy file.

        Args:
            ddy_obj: Either a DDY python object or the path to a ddy or an epw file.
                In case the ddy_obj is a path to ddy file it will be returned as is.
                For epw files, a heating and cooling design day will be approximated
                from statistical analysis of the EPW data.

        Returns:
            str -- Path to a ddy file.
    """

    if isinstance(ddy_obj, str):
        if not os.path.isfile(ddy_obj):
            raise ValueError('Invalid file path: %s' % ddy_obj)
        if ddy_obj.lower().endswith('.ddy'):
            ddy_file = ddy_obj
        elif ddy_obj.lower().endswith('.epw'):
            epw_obj = EPW(ddy_obj)
            file_path = get_tempfile('ddy', epw_obj.location.city)
            ddy_file = epw_obj.to_ddy(file_path)
        else:
            raise ValueError(
                'File path should end with ddy or epw not %s' % ddy_obj.split('.')[-1]
            )
    elif isinstance(ddy_obj, DDY):
        ddy_file = get_tempfile('ddy', ddy_obj.location.city)
        ddy_obj.save(ddy_file)
    elif isinstance(ddy_obj, EPW):
        file_path = get_tempfile('ddy', ddy_obj.location.city)
        ddy_file = ddy_obj.to_ddy(file_path)
    else:
        raise ValueError(
            'DDY input should be a string, a DDY object, or an EPW object. '
            'Not {}.'.format(type(ddy_obj))
        )
    return ddy_file
