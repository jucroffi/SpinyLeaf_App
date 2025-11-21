"""Handlers for honeybee and dragonfly models."""
import os
import sys
import json
import shutil

if (sys.version_info < (3, 0)):
    readmode = 'rb'
    writemode = 'wb'
else:
    readmode = 'r'
    writemode = 'w'

from honeybee.model import Model
from dragonfly.model import Model as ModelDF

from honeybee_energy.hvac.idealair import IdealAirSystem

from .helper import get_tempfile


def model_to_json(model_obj):
    """Translate a Honeybee model to a HBJSON file.

    Args:
        model_obj: Either a Honeybee model or the path to the HBJSON file.
            In case the model_obj is a path, it will be returned as is. For a
            Model object, it will be saved to a HBJSON file in a temp folder.

    Returns:
        str -- Path to HBJSON file.
    """
    if isinstance(model_obj, str):
        if not os.path.isfile(model_obj):
            raise ValueError('Invalid file path: %s' % model_obj)
        hb_file = model_obj
    elif isinstance(model_obj, Model):
        hb_file = get_tempfile('hbjson', model_obj.identifier)
        obj_dict = model_obj.to_dict()
        if (sys.version_info < (3, 0)):  # we need to manually encode it as UTF-8
            with open(hb_file, writemode) as fp:
                workflow_str = json.dumps(obj_dict, ensure_ascii=False)
                fp.write(workflow_str.encode('utf-8'))
        else:
            with open(hb_file, writemode, encoding='utf-8') as fp:
                json.dump(obj_dict, fp, ensure_ascii=False)
    else:
        raise ValueError(
            'Model input should be a string or a Honeybee Model. '
            'Not {}.'.format(type(model_obj))
        )
    return hb_file


def model_to_json_room_check(model_obj):
    """Translate a Honeybee model to HBJSON with checks for Rooms.

    If no Rooms are found in the model, a ValueError will be raised with an
    explicit error message.

    Args:
        model_obj: Either a Honeybee model or the path to the HBJSON file.
            In case the model_obj is a path, it will be returned as is. For a
            Model object, it will be saved to a HBJSON file in a temp folder.

    Returns:
        str -- Path to HBJSON file.
    """
    if isinstance(model_obj, Model):
        if len(model_obj.rooms) == 0:
            raise ValueError(
                'Model contains no Rooms. This is required for this recipe.')
    return model_to_json(model_obj)


def model_to_json_hvac_check(model_obj):
    """Translate a Honeybee model to HBJSON with checks for Rooms with HVAC.

    If no Rooms with HVAC are found in the model, a ValueError will be
    raised with an explicit error message.

    Args:
        model_obj: Either a Honeybee model or the path to the HBJSON file.
            In case the model_obj is a path, it will be returned as is. For a
            Model object, it will be saved to a HBJSON file in a temp folder.

    Returns:
        str -- Path to HBJSON file.
    """
    if isinstance(model_obj, Model):
        if len(model_obj.rooms) == 0:
            raise ValueError(
                'Model contains no Rooms. This is required for this recipe.')
        model_hvacs = model_obj.properties.energy.hvacs
        if len(model_hvacs) == 0:
            raise ValueError(
                'Model contains no HVAC Systems. This is required for this recipe.')
        ideal_air_count = 0
        for hvac in model_hvacs:
            if isinstance(hvac, IdealAirSystem):
                ideal_air_count += 1
        if ideal_air_count != 0:
            raise ValueError(
                'Model contains {} Ideal Air Systems.\nThis recipe requires all '
                'conditioned rooms\nto use detailed or template '
                'systems.'.format(ideal_air_count))
    return model_to_json(model_obj)


def model_to_json_grid_check(model_obj):
    """Translate a Honeybee model to HBJSON and perform a check for SensorGrids.

    If no SensorGrids are found in the model, a ValueError will be raised with
    an explicit error message. Note that this check will be bypassed if a
    HBJSON/HBpkl file is connected or if the path to a Radiance Folder.

    Args:
        model_obj: Either a Honeybee model or the path to the Honeybee model.
            Paths can either be to a HBJSON/HBpkl file or to a Honeybee Radiance
            folder that serves as direct input for a Radiance recipe.

    Returns:
        str -- Path to HBJSON file.
    """
    if isinstance(model_obj, Model):
        if len(model_obj.properties.radiance.sensor_grids) == 0:
            raise ValueError(
                'Model contains no sensor girds. This is required for this recipe.')
    if isinstance(model_obj, str) and os.path.isdir(model_obj):
        return _process_model_rad_folder(model_obj)
    return model_to_json(model_obj)


def model_to_json_grid_room_check(model_obj):
    """Translate a Honeybee model to HBJSON with checks for Rooms and SensorGrids.

    If no Rooms or SensorGrids are found in the model, a ValueError will be
    raised with an explicit error message.

    Args:
        model_obj: Either a Honeybee model or the path to the HBJSON file.
            In case the model_obj is a path, it will be returned as is. For a
            Model object, it will be saved to a HBJSON file in a temp folder.

    Returns:
        str -- Path to HBJSON file.
    """
    if isinstance(model_obj, Model):
        if len(model_obj.rooms) == 0:
            raise ValueError(
                'Model contains no Rooms. This is required for this recipe.')
    return model_to_json_grid_check(model_obj)


def model_to_json_view_check(model_obj):
    """Translate a Honeybee model to HBJSON and perform a check for Views.

    If no Views are found in the model, a ValueError will be raised with
    an explicit error message. Note that this check will be bypassed if a
    HBJSON/HBpkl file is connected or if the path to a Radiance Folder.

    Args:
        model_obj: Either a Honeybee model or the path to the HBJSON file.
            Paths can either be to a HBJSON/HBpkl file or to a Honeybee Radiance
            folder that serves as direct input for a Radiance recipe.

    Returns:
        str -- Path to HBJSON file.
    """
    if isinstance(model_obj, Model):
        if len(model_obj.properties.radiance.views) == 0:
            raise ValueError(
                'Model contains no views. This is required for this recipe.')
    if isinstance(model_obj, str) and os.path.isdir(model_obj):
        return _process_model_rad_folder(model_obj)
    return model_to_json(model_obj)


def model_dragonfly_to_json(model_obj):
    """Translate a Dragonfly model to a DFJSON file.

    Args:
        model_obj: Either a Dragonfly model or the path to the DFJSON file.
            In case the model_obj is a path, it will be returned as is.  For a
            Model object, it will be saved to a DFJSON file in a temp folder.

    Returns:
        str -- Path to DFJSON file.
    """
    if isinstance(model_obj, str):
        if not os.path.isfile(model_obj):
            raise ValueError('Invalid file path: %s' % model_obj)
        df_file = model_obj
    elif isinstance(model_obj, ModelDF):
        df_file = get_tempfile('dfjson', model_obj.identifier)
        # write the dictionary into a file
        obj_dict = model_obj.to_dict()
        with open(df_file, 'w') as fp:
            json.dump(obj_dict, fp)
    else:
        raise ValueError(
            'Model input should be a string or a Dragonfly Model. '
            'Not {}.'.format(type(model_obj))
        )
    return df_file


def _process_model_rad_folder(model_obj):
    """Zip a Radiance folder for input to a Radiance recipe."""
    assert os.path.isdir(os.path.join(model_obj, 'model')), \
        'File path "{}" does not contain a model subfolder for the Radiance folder.'
    try:
        model_path, model_id = os.path.split(model_obj)
        if model_id == 'radiance':  # the model ID is probably in the folder above
            model_path, model_id = os.path.split(model_path)
        else:
            model_id = None
    except Exception:  # shallow model folder
        model_id = None
    hb_file = get_tempfile('', model_id)
    return shutil.make_archive(hb_file, 'zip', model_obj)
