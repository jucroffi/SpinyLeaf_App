"""Post-processing classes for dynamic schedules."""
import json
import os
import sys
from itertools import islice, cycle

from honeybee.config import folders


class ApertureGroupSchedule(object):
    """ApertureGroupSchedule.

    This class contains the dynamic states schedule of a single ApertureGroup.

    Args:
        identifier: Identifier of the ApertureGroup.
        schedule: A list or tuple of integer values.

    Properties:
        * identifier
        * schedule
        * is_static
    """
    __slots__ = ('_identifier', '_schedule', '_is_static')

    def __init__(self, identifier, schedule, is_static=None):
        """Initialize ApertureGroupSchedule."""
        self._identifier = identifier
        self.schedule = schedule
        self.is_static = is_static

    @classmethod
    def from_dict(cls, data):
        """Initialize a ApertureGroupSchedule from a dictionary.

        Args:
            data: A dictionary representation of an ApertureGroupSchedule
                object.
        """
        identifier = data['identifier']
        schedule = data['schedule']
        is_static = data['is_static'] if 'is_static' in data else None
        return cls(identifier, schedule, is_static)

    @property
    def identifier(self):
        """Return identifier."""
        return self._identifier

    @property
    def schedule(self):
        """Return ApertureGroup schedule."""
        return self._schedule

    @schedule.setter
    def schedule(self, schedule):
        assert isinstance(schedule, (list, tuple)), \
            ('Failed to add schedule. Received input of type: '
            '%s. Expected input of type: list or tuple.' % type(schedule))
        if isinstance(schedule, tuple):
            schedule = list(schedule)
        if len(schedule) < 8760:
            schedule = list(islice(cycle(schedule), 8760))
        elif len(schedule) > 8760:
            error_message = (
                'The light path %s has %s values in '
                'its states schedule. Maximum allowed number of values '
                'is 8760.' % (self.identifier, len(schedule))
            )
            raise ValueError(error_message)
        self._schedule = schedule

    @property
    def is_static(self):
        """Return True if schedule is static."""
        return self._is_static

    @is_static.setter
    def is_static(self, value):
        if value is not None:
            assert isinstance(value, bool)
            self._is_static = value
        else:
            if len(set(self.schedule)) == 1:
                self._is_static = True
            else:
                self._is_static = False

    def to_dict(self):
        """Return ApertureGroupSchedule as a dictionary."""
        base = {}
        base['identifier'] = self.identifier
        base['schedule'] = self.schedule
        base['is_static'] = self.is_static
        return base

    def ToString(self):
        """Overwrite .NET ToString."""
        return self.__repr__()

    def __repr__(self):
        return '{}: {}'.format(self.__class__.__name__, self.identifier)


class DynamicSchedule(object):
    """DynamicSchedule.

    This class contains a single property (dynamic_schedule). This property is
    a dictionary where keys are ApertureGroup identifiers and the value for
    each key is the dynamic schedule for the ApertureGroup.

    Args:
        dynamic_schedule: A dictionary of ApertureGroup identifier and
            schedules.

    Properties:
        * dynamic_schedule
    """
    __slots__ = ('_dynamic_schedule',)

    def __init__(self, dynamic_schedule = None):
        self.dynamic_schedule = dynamic_schedule

    @classmethod
    def from_group_schedules(cls, group_schedules):
        """Initialize a DynamicSchedule from a list of ApertureGroupSchedules.

        The method will automatically sense if there are duplicated groups in
        the list and ensure the group schedule only appears once.

        Args:
            data: A dictionary representation of a DynamicSchedule objects.
        """
        dyn_sch = cls()
        dyn_sch_ids = set()
        for _ap_group in group_schedules:
            assert isinstance(_ap_group, ApertureGroupSchedule), \
                'Expected Aperture Group Schedule. Got {}'.format(type(_ap_group))
            if _ap_group.identifier not in dyn_sch_ids:
                dyn_sch_ids.add(_ap_group.identifier)
                dyn_sch.add_aperture_group_schedule(_ap_group)
        return dyn_sch

    @classmethod
    def from_dict(cls, data):
        """Initialize a DynamicSchedule from a dictionary.

        Args:
            data: A dictionary representation of a DynamicSchedule objects.
        """
        dynamic_schedule = {}
        for identifier, group in data.items():
            dynamic_schedule[identifier] = ApertureGroupSchedule.from_dict(group)
        return cls(dynamic_schedule)

    @classmethod
    def from_json(cls, json_file):
        """Initialize a DynamicSchedule from a JSON file.

        Args:
            json_file: Path to JSON file.
        """
        assert os.path.isfile(json_file), 'Failed to find %s' % json_file
        if sys.version_info < (3, 0):
            with open(json_file) as inf:
                data = json.load(inf)
        else:
            with open(json_file, encoding='utf-8') as inf:
                data = json.load(inf)
        return cls.from_dict(data)

    @property
    def dynamic_schedule(self):
        """Return dynamic schedule as a dictionary."""
        return self._dynamic_schedule

    @dynamic_schedule.setter
    def dynamic_schedule(self, dynamic_schedule):
        if not dynamic_schedule:
            dynamic_schedule = {}
        assert isinstance(dynamic_schedule, dict), \
            'Expected dictionary. Got %s.' % type(dynamic_schedule)
        self._dynamic_schedule = dynamic_schedule

    def add_aperture_group_schedule(self, aperture_group_schedule):
        """Add an ApertureGroupSchedule to the DynamicSchedule instance.

        Args:
            aperture_group_schedule: An ApertureGroupSchedule object.
        """
        assert isinstance(aperture_group_schedule, ApertureGroupSchedule), \
            ('Failed to add ApertureGroupSchedule. Received input of type: '
            '%s. Expected input of type: ApertureGroupSchedule.' \
            % type(aperture_group_schedule))
        identifier = aperture_group_schedule.identifier
        self.dynamic_schedule[identifier] = aperture_group_schedule

    def filter_by_identifiers(self, identifiers):
        """Filter the DynamicSchedule by identifiers.

        This method returns a filtered DynamicSchedule object.

        Args:
            identifiers: A list of identifiers.

        Returns:
            A filtered DynamicSchedule object.
        """
        filter_dyn_sch = DynamicSchedule()
        for identifier in identifiers:
            if identifier in self.dynamic_schedule:
                filter_dyn_sch.add_aperture_group_schedule(
                    self.dynamic_schedule[identifier]
                )
            else:
                filter_dyn_sch.add_aperture_group_schedule(
                    ApertureGroupSchedule(identifier, [0])
                )
        return filter_dyn_sch

    def to_dict(self, simplified=False):
        """Return DynamicSchedule as a dictionary."""
        base = {}
        for identifier, group in self.dynamic_schedule.items():
            if not simplified:
                base[identifier] = group.to_dict()
            else:
                base[identifier] = group.to_dict()['schedule']
        return base

    def to_json(self, folder=None, file_name=None, indent=None, simplified=False):
        """Write a DynamicSchedule to JSON.

        Args:
            folder: A text string for the directory where the JSON file will be
                written. If unspecified, the default simulation folder will be
                used. This is usually at "C:\\Users\\USERNAME\\simulation."
            file_name: _description_. Defaults to None.
            indent: A positive integer to set the indentation used in the
                resulting JSON file. (Default: None).

        Returns:
            json_file: Path to JSON file.
        """
        # create dictionary of the DynamicSchedule
        dyn_sch_dict = self.to_dict(simplified=simplified)

        # set up name and folder for the JSON
        file_name = file_name if file_name else 'dynamic_schedule'
        if not file_name.endswith('.json'):
            file_name += '.json'
        folder = folder if folder is not None else folders.default_simulation_folder
        json_file = os.path.join(folder, file_name)

        # write JSON
        with open(json_file, 'w') as fp:
            json.dump(dyn_sch_dict, fp, indent=indent)
        return json_file

    def duplicate(self):
        """Get a copy of this object."""
        return self.__copy__()

    def ToString(self):
        """Overwrite .NET ToString."""
        return self.__repr__()

    def __copy__(self):
        new_dyn_sch = DynamicSchedule(self.dynamic_schedule)
        return new_dyn_sch

    def __repr__(self):
        return '{}'.format(self.__class__.__name__)
