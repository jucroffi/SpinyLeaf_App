# coding=utf-8
"""honeybee-doe2 simulation parameters."""
from __future__ import division

from honeybee.typing import clean_doe2_string, int_in_range
from honeybee_energy.simulation.runperiod import RunPeriod

from .config import GEO_CHARS
from .util import generate_inp_string, header_comment_minor


class SimulationPar(object):
    """Complete set of DOE-2 Simulation Settings.

    Args:
        title: Text for the title of the project. (Default: *Unnamed*).
        run_period: A honeybee-energy RunPeriod object to describe the time period
            over which to run the simulation. (Default: Run for the whole year).
        site: A SiteData object describing the site where the simulation is
            run. (Default: HArtford, CT).

    Properties:
        * title
        * run_period
        * site
    """
    __slots__ = ('_title', '_run_period', '_site')

    def __init__(self, title='Unnamed', run_period=None, site=None):
        """Initialize SimulationPar."""
        self.title = title
        self.run_period = run_period
        self.site = site

    @classmethod
    def from_dict(cls, data):
        """Create a SimulationPar object from a dictionary.

        Args:
            data: A SimulationPar dictionary in following the format below.

        .. code-block:: python

            {
            "type": "SimulationPar",
            "title": 'sample_project', # Text for the title
            "run_period": {}, # Honeybee RunPeriod dictionary
            "site": {}, # Honeybee SiteData dictionary
            }
        """
        assert data['type'] == 'SimulationPar', \
            'Expected SimulationPar dictionary. Got {}.'.format(data['type'])
        title = data['title'] if 'title' in data else 'Unnamed'
        run_period = None
        if 'run_period' in data and data['run_period'] is not None:
            run_period = RunPeriod.from_dict(data['run_period'])
        site = None
        if 'site' in data and data['site'] is not None:
            site = SiteData.from_dict(data['site'])
        return cls(title, run_period, site)

    @property
    def title(self):
        """Get or set text for the title of the project."""
        return self._title

    @title.setter
    def title(self, value):
        if value:
            value = clean_doe2_string(value, GEO_CHARS)
        else:
            value = None
        self._title = value

    @property
    def run_period(self):
        """Get or set a RunPeriod object for the time period to run the simulation."""
        return self._run_period

    @run_period.setter
    def run_period(self, value):
        if value is not None:
            assert isinstance(value, RunPeriod), 'Expected RunPeriod for ' \
                'SimulationPar run_period. Got {}.'.format(type(value))
            self._run_period = value
        else:
            self._run_period = RunPeriod()

    @property
    def site(self):
        """Get or set a SiteData object for the project site of the simulation."""
        return self._site

    @site.setter
    def site(self, value):
        if value is not None:
            assert isinstance(value, SiteData), 'Expected SiteData for ' \
                'SimulationPar site. Got {}.'.format(type(value))
            self._site = value
        else:
            self._site = SiteData()

    def to_inp(self):
        """Get an DOE-2 INP string representation of the SimulationPar."""
        # add the starting headers
        sp_str = []
        sp_str.append(header_comment_minor('Abort, Diagnostics'))
        sp_str.append(header_comment_minor('Global Parameters'))
        sp_str.append(header_comment_minor('Title, Run Periods, Design Days, Holidays'))
        # add the title and run period
        title_str = \
            'TITLE\n' \
            '   LINE-1           = *{}*\n' \
            '   ..\n'.format(self.title)
        sp_str.append(title_str)
        sp_str.append(self.run_period.to_inp())
        # add the site and building data
        sp_str.append(header_comment_minor('Compliance Data'))
        sp_str.append(header_comment_minor('Site and Building Data'))
        sp_str.append(self.site.to_inp())
        return '\n'.join(sp_str)

    def to_dict(self):
        """SimulationPar dictionary representation."""
        return {
            'type': 'SimulationPar',
            'title': self.title,
            'run_period': self.run_period.to_dict(),
            'site': self.site.to_dict(),
        }

    def duplicate(self):
        """Get a copy of this object."""
        return self.__copy__()

    def ToString(self):
        """Overwrite .NET ToString."""
        return self.__repr__()

    def __copy__(self):
        return SimulationPar(
            self.title, self.run_period.duplicate(), self.site.duplicate())

    def __key(self):
        """A tuple based on the object properties, useful for hashing."""
        return (self.title, hash(self.run_period), hash(self.site))

    def __hash__(self):
        return hash(self.__key())

    def __eq__(self, other):
        return isinstance(other, SimulationPar) and self.__key() == other.__key()

    def __ne__(self, other):
        return not self.__eq__(other)

    def __repr__(self):
        return 'DOE-2 SimulationPar: {}'.format(self.title)


class SiteData(object):
    """Object to describe the project site of the simulation.

    Args:
        altitude: A number for the altitude of the location above sea level
            in Feet. (Default: 150).

    Properties:
        * altitude
    """
    __slots__ = ('_altitude',)

    def __init__(self, altitude=150):
        """Initialize SimulationPar."""
        self.altitude = altitude

    @classmethod
    def from_dict(cls, data):
        """Create a SiteData object from a dictionary.

        Args:
            data: A SiteData dictionary in following the format below.

        .. code-block:: python

            {
            "type": "SiteData",
            "altitude": 100  # altitude above sea level (ft)
            }
        """
        assert data['type'] == 'SiteData', \
            'Expected SiteData dictionary. Got {}.'.format(data['type'])
        altitude = data['altitude'] if 'altitude' in data else 150
        return cls(altitude)

    @property
    def altitude(self):
        """A number for the altitude of the location above sea level in Feet."""
        return self._altitude

    @altitude.setter
    def altitude(self, value):
        self._altitude = int_in_range(value, input_name='site data altitude')

    def to_inp(self):
        """Get an DOE-2 INP string representation of the SiteData."""
        # create the INP string for the site
        keywords = ('ALTITUDE', 'C-STATE', 'C-WEATHER-FILE',
                    'C-COUNTRY', 'C-901-LOCATION')
        values = (self.altitude, '21', '*TMY2\\HARTFOCT.bin*', '1', '1092')
        site_str = generate_inp_string('Site Data', 'SITE-PARAMETERS', keywords, values)
        # create the INP string for the building data
        keywords, values = ('HOLIDAYS',), ('"Standard US Holidays"',)
        bldg_str = generate_inp_string(
            'Building Data', 'BUILD-PARAMETERS', keywords, values)
        return site_str + bldg_str

    def to_dict(self):
        """SiteData dictionary representation."""
        return {
            'type': 'SiteData',
            'altitude': self.altitude
        }

    def duplicate(self):
        """Get a copy of this object."""
        return self.__copy__()

    def ToString(self):
        """Overwrite .NET ToString."""
        return self.__repr__()

    def __copy__(self):
        return SiteData(self.altitude)

    def __key(self):
        """A tuple based on the object properties, useful for hashing."""
        return (self.altitude,)

    def __hash__(self):
        return hash(self.__key())

    def __eq__(self, other):
        return isinstance(other, SiteData) and self.__key() == other.__key()

    def __ne__(self, other):
        return not self.__eq__(other)

    def __repr__(self):
        return self.to_inp()


def run_period_to_inp(run_period):
    """Translate a honeybee-energy RunPeriod object to a DOE-2 INP string"""
    # create the string for the run period
    year = 2020 if run_period.is_leap_year else 2021
    keywords = ('BEGIN-MONTH', 'BEGIN-DAY', 'BEGIN-YEAR',
                'END-MONTH', 'END-DAY', 'END-YEAR')
    values = (run_period.start_date.month, run_period.start_date.day, year,
              run_period.end_date.month, run_period.end_date.day, year)
    rp_str = generate_inp_string('Default Run Period', 'RUN-PERIOD-PD', keywords, values)
    # create the string for the holidays
    keywords, values = ('LIBRARY-ENTRY',), ('"US"',)
    hol_str = generate_inp_string('Standard US Holidays', 'HOLIDAYS', keywords, values)
    return rp_str + hol_str
