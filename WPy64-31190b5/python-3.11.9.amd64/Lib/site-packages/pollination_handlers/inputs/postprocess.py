"""Handlers for post-processing options."""
import os
import json

from .helper import get_tempfile


def grid_metrics(gm_obj):
    """Validate the file for custom grid metrics.

        Args:
            gm_obj: An object with custom grid metrics. This can be either a
                JSON file, a string, or a list of grid metrics.

        Returns:
            str -- Path to a the custom grid metrics file.
    """
    if isinstance(gm_obj, str):
        if os.path.isfile(gm_obj):
            with open(gm_obj) as file:
                grid_metrics = json.load(file)
        else:
            grid_metrics = json.loads(gm_obj)
    elif isinstance(gm_obj, list):
        grid_metrics = gm_obj
    else:
        raise TypeError(
            'Unexpected type of input gm_obj. Valid types are str and list. '
            'Type of input is: %s.' % type(gm_obj)
            )

    keywords = ['minimum', 'maximum', 'exclusiveMinimum', 'exclusiveMaximum']
    of_keywords = ['allOf', 'anyOf']
    # validate
    for grid_metric in grid_metrics:
        assert isinstance(grid_metric, dict), \
        'Each item in grid metrics must be a dictionary.'
        for key, value in grid_metric.items():
            if key in keywords:
                assert isinstance(value, (float, int)), \
                    'Expected float or integer. Received: %s' % type(value)
            elif key in of_keywords:
                assert isinstance(value, list)
                for of_obj in value:
                    for of_k, of_v in of_obj.items():
                        if of_k in keywords:
                            assert isinstance(of_v, (float, int)), \
                                'Expected float or integer. Received: %s' % type(of_v)
                        else:
                            raise ValueError('Valid keywords are %s.' % keywords)
            else:
                raise ValueError('Valid keywords are %s.' % (keywords + of_keywords))

    file_path = get_tempfile('json', 'grid_metrics')
    with open(file_path, 'w') as f:
        json.dump(grid_metrics, f)

    return file_path
