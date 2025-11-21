import re

from .optionbase import BoolOption
from .rcontrib import RcontribOptions

import honeybee_radiance_command._exception as exceptions


class RfluxmtxOptions(RcontribOptions):
    """rcontrib command options."""

    __slots__ = ('_v',)

    def __init__(self):
        """rcontrib command options."""
        RcontribOptions.__init__(self)
        self._on_setattr_check = False
        self._v = BoolOption('v', 'verbose report - default: off')
        self._protected = ('f', 'e', 'p', 'b', 'bn', 'm', 'M')
        self._on_setattr_check = True

    def _on_setattr(self):
        """This method executes after setting each new attribute.

        Use this method to add checks that are necessary for OptionCollection. For
        instance in rtrace option collection -ti and -te are exclusive. You can include a
        check to ensure this is always correct.
        """
        RcontribOptions._on_setattr(self)
        for opt in self._protected:
            if getattr(self, opt).is_set:
                raise exceptions.ProtectedOptionError('rfluxmtx', opt)

    @property
    def v(self):
        """Verbose report - default: off"""
        return self._v

    @v.setter
    def v(self, value):
        self._v.value = value


class RfluxmtxControlParameters(object):
    """Rfluxmtx ControlParameters.

    Set the values for hemisphere sampling type, hemisphere up direction and output file
    location. This class generates a string for rflumtx which should be included
    with a receiver aperture group.

    Here is a sample rfluxmtx control parmaters: ``#@rfluxmtx u=0,1,0 h=kf o=output.vmx``

    Args:
        sampling_type: Set hemisphere sampling type. Acceptable inputs for hemisphere
            sampling type are:

              * u for uniform.(Usually applicable for ground).
              * kf for klems full.
              * kh for klems half.
              * kq for klems quarter.
              * rN for Reinhart - Tregenza type skies. N stands for subdivisions and
               defaults to 1.
              * scN for shirley-chiu subdivisions.

            Add a ``-`` in front of the input for left-handed coordinates. For more
            information see rfluxmtx docs.
            https://www.radiance-online.org/learning/documentation/manual-pages/pdfs/rfluxmtx.pdf/at_download/file
        up_direction: Orient the "up" direction for the hemisphere using the indicated
            axis or direction vector using a tuple of 3 numbers. Valid string inputs are
            [-]{X|Y|Z|ux,uy,uz}. Default: Y
        output_spec: Send the matrix data for this receiver to the indicated file or 
            command. Single or double quotes may be used to contain strings with spaces,
            and commands must begin with an exclamation mark (`!`). The file format
            will be determined by the command-line -fio option and will include an
            information header unless the -h option was used to turn headers off. (The
            output file specification is ignored for senders.)

    """
    __slots__ = ('_sampling_type', '_up_direction', '_output_spec')

    def __init__(self, sampling_type='u', up_direction='Y', output_spec=None):
        self.sampling_type = sampling_type
        self.up_direction = up_direction
        self._output_spec = output_spec

    @classmethod
    def from_string(cls, value):
        """Create RfluxmtxControlParameters from a string.

        Args:
            value: A string in the format: #@rfluxmtx u=0,1,0 h=kf o=output.vmx
        """
        assert value.startswith('#@rfluxmtx'), \
            'Rfulxmtx control parameters should start with #@rfluxmtx.'
        # get sampling type
        segments = value.replace('#@rfluxmtx', '').strip().split(' ')
        sampling_type = 'u'
        up_direction = 'Y'
        output_spec = None
        for seg in segments:
            if seg.startswith('h='):
                sampling_type = seg.replace('h=', '').strip()
            elif seg.startswith('u='):
                up_direction = seg.replace('u=', '').strip()
            elif seg.startswith('o='):
                output_spec = seg.replace('o=', '').strip()
        if ',' in up_direction:
            up_direction = up_direction.split(',')
        return cls(sampling_type, up_direction, output_spec)

    @classmethod
    def from_file(cls, path):
        """Create RfluxmtxControlParameters from a file.

        Args:
            path: A file path to a receiver file containing a RfluxmtxControlParameters.

        """
        with open(path, 'r') as f:
            content = f.read()
        try:
            value = re.search(r'^#@rfluxmtx[\s\S].*$', content, re.MULTILINE).group(0)
        except AttributeError:
            raise ValueError(
                '%s is not a valid receiver file with RfluxmtxControlParameters.' % path
            )

        return cls.from_string(value)

    @property
    def sampling_type(self):
        return self._sampling_type

    @sampling_type.setter
    def sampling_type(self, value):
        if value is None:
            value = 'u'
            input_value = 'u'
        elif value.startswith('-'):
            input_value = value[1:]
        else:
            input_value = value

        if input_value in ['u', 'kf', 'kh', 'kq']:
            self._sampling_type = value
        elif input_value.startswith('r'):
            self._validate_reinhart_tregenza(input_value)
            self._sampling_type = value
        elif input_value.startswith('sc'):
            self._validate_shirley_chiu(input_value)
            self._sampling_type = value
        else:
            raise ValueError('Invalid sampling type: {}'.format(value))

    @staticmethod
    def _validate_reinhart_tregenza(value):
        """Validate Reinhart-Tregenza sampling type."""
        number_of_subdivisions = value[1:]
        if not number_of_subdivisions.isdigit():
            raise ValueError('Invalid sampling_type: {}'.format(value))

    @staticmethod
    def _validate_shirley_chiu(value):
        """Validate Shirley-Chiu sampling type."""
        number_of_subdivisions = value[2:]
        if not number_of_subdivisions.isdigit():
            raise ValueError('Invalid sampling_type: {}'.format(value))

    @property
    def up_direction(self):
        """hemisphere direction.

        The acceptable inputs for hemisphere direction are a tuple with 3 values
        or 'X', 'Y', 'Z', 'x', 'y', 'z', '-X', '-Y','-Z', '-x', '-y','-z'."""
        return self._up_direction
    
    @up_direction.setter
    def up_direction(self, value):

        if value is None:
            self._up_direction = 'Y'
        elif value in ('X', 'Y', 'Z', 'x', 'y', 'z', '-X', '-Y', '-Z', '-x', '-y', '-z',
            '+X', '+Y', '+Z', '+x', "+y", "+z"):
                self._up_direction = value
        elif isinstance(value, (tuple, list)):
            assert len(value) == 3, 'The length of the up direction vector must be 3.'
            self._up_direction = ','.join((str(v) for v in value))
        else:
            raise ValueError('Invalid up direction: {}'.format(value))

    @property
    def output_spec(self):
        """
        Send the matrix data for this receiver to the indicated file or command. Single
        or double quotes may be used to contain strings with spaces, and commands must
        begin with an exclamation mark (`!`). The file format will be determined by the
        command-line -fio option and will include an information header unless the -h
        option was used to turn headers off. (The output file specification is ignored
        for senders.)
        """
        return self._output_spec

    def __repr__(self):
        output_spec = 'o=%s' % self.output_spec if self.output_spec else ''
        spec = '#@rfluxmtx h=%s u=%s %s' % (self.sampling_type,
                                            self.up_direction,
                                            output_spec)
        return spec.strip()

    def to_radiance(self):
        """Return a radiance definition as a string."""
        return self.__repr__()
