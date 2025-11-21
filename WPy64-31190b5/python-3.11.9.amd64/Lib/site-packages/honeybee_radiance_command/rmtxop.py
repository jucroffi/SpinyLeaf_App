"""rmtxop command."""
from .options.rmtxop import RmtxopOptions
from ._command import Command
import honeybee_radiance_command._exception as exceptions
import honeybee_radiance_command._typing as typing
import os


class Rmtxop(Command):
    """Rmtxop command.

    Concatenate, add, multiply, divide, transpose, scale, and convert matrices. The
    implementation here allows for matrix operations on upto four matrices at the same
    time. While the Radiance binary allows for even more, this restriction has been
    imposed here to keep the code somewhat clean.

    In standard Radiance syntax, the entire command is specified in a single line. In
    the implementation here, the matrices and all the corresponding scaling factors,
    transformations and operators are specified through individual list. So, care
    must be taken to ensure that the order of transformations, scaling etc. is correctly
    assigned to the sequence of the matrices.

    An extensive number of input and parameter possibilities are evaluated in the test
    module for this class. A review of that module is higly recommended to understand
    how to specify inputs, scalars, transforms and transposition options for this class.

        Args:
        options: Command options. It will be set to Radiance default values
            if unspecified.
        output: File path to the output file (Default: None).
        matrices: A single file path or a list/tuple of multiple file paths of matrix
            files.
        transforms: A nested list of tuples/lists containing floating point numbers that
            are meant to transform the matrix/matrices specified through the 'matrices'
            input. Defaults to None, implying no transformations. For a calculation
            involving 3 matrices, if a specific transformation, say weighting by (1,2,3.4)
            is to be done to the second matrix, then the input for this argument will be
            (None, (1,2,3.4), None).
        scalars: A nested list of tuples/lists containing floating point numbers that
            are meant to scale the matrix/matrices specified through the 'matrices'
            input. Defaults to None, implying no scaling. For a calculation
            involving 3 matrices, if  scaling by -1.5  is to be done to the second matrix,
            then the input for this argument will be (None, -1.5, None).
        transposes: A boolean value or a list of boolean values (in case of multiple
            matrices) to specify transpose operation on the matrix/matrices. Defaults to
            None, implying no transpositions. For a calculation involving 3 matrices, if
            the first and third matrices are to be transposed, then the input for this
            argument will be [True,None,True].
        operators: A character string or list of character strings indicating the
            operation between two or more matrices. The default value is None indicating
            concatenation. If this input is specified then its length should be equal
            to one less than the number of matrices. For calculation involving 3 matrices,
            if the first and second matrices are to be added, and the result is to be
            multiplied by the third matrix, then the input will be ['+','*]

        Properties:
        * options
        * output
        * matrices
        * transforms
        * transposes
        * scalars
        * operators

    """

    __slots__ = ('_matrices', '_transforms', '_scalars', '_transposes', '_operators')

    def __init__(self, options=None, output=None, matrices=None, transforms=None,
                 transposes=None, scalars=None, operators=None):
        """Initialize Command."""
        Command.__init__(self, output=output)
        self.options = options
        self.output = output
        self.matrices = matrices
        self.transforms = transforms
        self.transposes = transposes
        self.scalars = scalars
        self.operators = operators

    @property
    def options(self):
        """Rmtxop options."""
        return self._options

    @options.setter
    def options(self, value):
        if value is None:
            value = RmtxopOptions()

        if not isinstance(value, RmtxopOptions):
            raise ValueError('Expected RmtxopOptions not {}'.format(type(value)))

        self._options = value

    @property
    def matrices(self):
        """List of matrix files on which operations are to be performed."""
        return self._matrices

    @matrices.setter
    def matrices(self, value):
        if value is not None:
            if not isinstance(value, (tuple, list)):
                value = [value]
            # check if values are Radiance commands
            if value and any([isinstance(v, Command) for v in value]):
                self._matrices = [cmd.enclose_command() if isinstance(cmd, Command) else \
                    typing.path_checker(cmd) for cmd in value]
            else:
                self._matrices = typing.path_checker_multiple(value)
        else:
            self._matrices = []

    @property
    def transforms(self):
        """Transformation coefficients that are specified as a single floating
        point number or a tuple or list of floating point numbers. If specified as tuple
        or list, then its length should be an even multiple of the original matrix
        components.
        If a single matrix is being considered for operation, then the input for this
        can be single number or a single list.
        If more than one matrix has been considered for operation then the input for this
        should be a list/tuple containing numbers or nested tuple/lists whose length is
        equal to the number of matrices.
        For the same matrix both transforms and scalars cannot be specified at the same
        time.
        """
        return self._transforms

    @transforms.setter
    def transforms(self, value):
        if value is not None:
            if type(value) in (list, tuple):
                self._transforms = [self.__scl_trnsfrm_setter(element) for element in
                                    value]
            else:
                self._transforms = [self.__scl_trnsfrm_setter(value)]
        else:
            self._transforms = []

    @property
    def scalars(self):
        """Scalar factor to scale the elements of the matrix that are specified as a
        single floating point number or a tuple or list of floating point numbers. If
        specified as a tuple or list, then its length should be equal to the original
        matrix components.
        If a single matrix is being considered for operation, then the input for this
        can be single number or a single list.
        If more than one matrix has been considered for operation then the input for this
        should be a list/tuple containing numbers or nested tuple/lists whose length is
        equal to the number of matrices.
        For the same matrix both transforms and scalars cannot be specified at the same
        time.
        """
        return self._scalars

    @scalars.setter
    def scalars(self, value):
        if value is not None:
            if type(value) in (list, tuple):
                self._scalars = [self.__scl_trnsfrm_setter(element) for element in value]
            else:
                self._scalars = [self.__scl_trnsfrm_setter(value)]
        else:
            self._scalars = []

    @property
    def transposes(self):
        """
        If a single matrix is being considered for operation, then the input for this
        can be single a single boolean.
        If more than one matrix has been considered for operation then the input for this
        should be a list/tuple containing booleans. """

        return self._transposes

    @transposes.setter
    def transposes(self, value):
        if value is not None:
            if not type(value) in (tuple, list):
                self._transposes = [value]
            else:
                self._transposes = value
        else:
            self._transposes = []

    @property
    def operators(self):
        """The operator for specifying addition('+'), subtraction('-'),
        multiplication('*'), division(/) or concatenation(.) between two matrices.
        The default operation is concatenation.
        In the case of addition, the two matrices involved must have the same number of
        components. If subtraction is desired, use addition ('+') with a scaling
        parameter of -1 for the second matrix (the -s option). For element-wise
        multiplication and division, the second matrix is permitted to have a single
        component per element, which will be applied equally to all components of the
        first matrix. If element-wise division is specified, any zero elements in the
        second matrix will result in a warning and the corresponding component(s) in the
        first matrix will be set to zero.
        In case of posix based systems the multiplication operator will be quoted '*'
        automatically by the script.
        """
        return self._operators

    @operators.setter
    def operators(self, value):
        if value is not None:
            if type(value) in (list, tuple):
                self._operators = [self._oprtr_setter(element) for element in value]
            else:
                self._operators = [self._oprtr_setter(value)]
        else:
            self._operators = []

    @staticmethod
    def __scl_trnsfrm_setter(value):
        """Check the input provided for the scalar or transform and return a list
        of floating point numbers"""
        # If the input is a single number then convert into a single element list.
        # Do a check for floating point number in both cases.
        if value is not None:
            if type(value) in (list, tuple):
                return [float(num) for num in value]
            else:
                return [float(value)]
        else:
            return None

    @staticmethod
    def _oprtr_setter(oprtr_value):
        """Check the input provided for the operator from the list of possible of
        options"""
        if oprtr_value:
            valid_values = ('+', '*', '/', '.')

            if oprtr_value not in valid_values:
                raise exceptions.InvalidValueError('rmtxop', oprtr_value,
                                                   valid_values=valid_values)
            if os.name == 'posix' and oprtr_value == '*':
                oprtr_value = "'*'"

            return oprtr_value
        else:
            return None

    def to_radiance(self, stdin_input=False):
        """Command in Radiance format.

        Args:
            stdin_input: A boolean that indicates if the input for this command
                comes from stdin. This is for instance the case when you pipe the input
                from another command (default: False).
        """
        if stdin_input:
            self.matrices = '-'
        self.validate(stdin_input)

        command_parts = [self.command, self.options.to_radiance()]

        transposes = self.transposes
        transforms = self.transforms
        scalars = self.scalars
        operators = self.operators

        for idx, matrix in enumerate(self.matrices):
            if transposes and transposes[idx]:
                command_parts.append('-t')
            if transforms and transforms[idx]:
                command_parts.extend(['-c'] + ['%s' % val for val in transforms[idx]])
            if scalars and scalars[idx]:
                command_parts.extend(['-s'] + ['%s' % val for val in scalars[idx]])
            command_parts.append(matrix)
            if operators and idx < (len(self.matrices) - 1):
                command_parts.append(operators[idx])

        cmd = ' '.join(command_parts)

        if self.pipe_to:
            cmd = ' | '.join((cmd, self.pipe_to.to_radiance(stdin_input=True)))
        elif self.output:
            cmd = ' > '.join((cmd, self.output))

        return ' '.join(cmd.split())

    def validate(self, stdin_input=False):
        Command.validate(self)

        if not stdin_input and not self.matrices:
            raise exceptions.MissingArgumentError(self.command, 'matrices')

        len_matrices = len(self.matrices)

        # Note: The checks below are only meant for commandline options and files. There
        # are several checks that rmtxop does at runtime relating to matrix shapes
        # and operator compatibility.

        # Check 1: Number of matrices is less than 5 (for performance reasons).
        assert len_matrices < 5, \
            'The number of matrices allowed for calculation has been set to four or ' \
            'lower. Currently, %s matrices have been specified.' % len_matrices

        # Check 2: If scalars, transforms and transposes are set, then they are declared for
        # every matrix.
        len_dict = {'scalars': len(self.scalars), 'transforms': len(self.transforms),
                    'transposes': len(self.transposes)}
        for matrix_param, param_list_len in len_dict.items():
            if param_list_len:
                assert param_list_len == len_matrices, \
                    'The number of %s specified should either be None or equal to the ' \
                    'number of matrices (%s). Currently a list of %s has been specified.' \
                    % (matrix_param, len_matrices, param_list_len)

        # Check 3: If operators are declared, then they are declared for every adjacent
        # matrix operation (i.e. one less than total number of matrices).
        len_operators = len(self.operators)
        if len_operators:
            assert len_operators == (len_matrices - 1), \
                'The number of specified operators (%s) should either be None or ' \
                'one less than the number of matrices (%s).' % (
                    len_operators, len_matrices)

        # Check 4: For a given matrix, scalar and transform is not set simultaneously.
        if len_dict['scalars'] and len_dict['transforms']:
            for idx, scalar in enumerate(self.scalars):
                transform = self.transforms[idx]
                if scalar and transform:
                    msg = 'For matrix %s both transform(%s) and scalar(%s) coefficients ' \
                          'have been provided. Transform and scalar coefficients cannot' \
                          'be specified simultaneously for the same matrix.' % \
                          (idx, transform, scalar)
                    raise Exception(msg)
