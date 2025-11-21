"""Commands to compute view factors to geometry."""
import click
import os
import sys
import logging
import math
try:
    import cupy as np
    is_gpu = True
except ImportError:
    is_gpu = False
    import numpy as np

from honeybee_radiance.config import folders

from honeybee_radiance_command.rcontrib import Rcontrib, RcontribOptions
from honeybee_radiance_command._command_util import run_command

from ladybug.futil import preparedir

from honeybee_radiance_postprocess.reader import binary_to_array

_logger = logging.getLogger(__name__)


@click.group(help='Commands to compute view factors to geometry.')
def view_factor():
    pass


@view_factor.command('contrib')
@click.argument(
    'octree', type=click.Path(exists=True, file_okay=True, resolve_path=True)
)
@click.argument(
    'sensor-grid', type=click.Path(exists=True, file_okay=True, resolve_path=True)
)
@click.argument(
    'modifiers', type=click.Path(exists=True, file_okay=True, resolve_path=True)
)
@click.option(
    '--ray-count', type=click.INT, default=6, show_default=True,
    help='The number of rays to be equally distributed over a sphere to compute '
    'the view factor for each of the input sensors.'
)
@click.option(
    '--rad-params', show_default=True, help='Radiance parameters.'
)
@click.option(
    '--rad-params-locked', show_default=True, help='Protected Radiance parameters. '
    'These values will overwrite user input rad parameters.'
)
@click.option(
    '--folder', default='.', help='Output folder into which the modifier and '
    'octree files will be written.'
)
@click.option(
    '--name', default='view_factor', help='File name of the view factor file.'
)
def rcontrib_command_with_view_postprocess(
        octree, sensor_grid, modifiers, ray_count, rad_params, rad_params_locked,
        folder, name
):
    """Run rcontrib to get spherical view factors from a sensor grid.

    This command is similar to the one in honeybee-radiance, but the
    post-processing is using NumPy.

    \b
    Args:
        octree: Path to octree file.
        sensor-grid: Path to sensor grid file.
        modifiers: Path to modifiers file.
    """
    try:
        # create the directory if it's not there
        if not os.path.isdir(folder):
            preparedir(folder)

        # generate the ray vectors to be used in the view factor calculation
        if ray_count == 6:
            rays = ((1, 0, 0), (0, 1, 0), (0, 0, 1), (-1, 0, 0), (0, -1, 0), (0, 0, -1))
        else:
            rays = _fibonacci_spiral(ray_count)
        ray_str = [' {} {} {}\n'.format(*ray) for ray in rays]

        # create a new .pts file with the view vectors
        ray_file = os.path.abspath(os.path.join(folder, '{}.pts'.format(name)))
        total_rays = 0
        with open(sensor_grid) as sg_file:
            with open(ray_file, 'w') as r_file:
                for line in sg_file:
                    for ray in ray_str:
                        try:
                            r_file.write(' '.join(line.split()[:3]) + ray)
                            total_rays += 1
                        except Exception:
                            pass  # we are at the end of the file

        # set up the Rcontrib options
        options = RcontribOptions()
        if rad_params:  # parse input radiance parameters
            options.update_from_string(rad_params.strip())
        if rad_params_locked:  # overwrite input values with protected ones
            options.update_from_string(rad_params_locked.strip())
        # overwrite specific options that would otherwise break the command
        options.M = modifiers
        options.update_from_string('-I -V- -y {}'.format(total_rays))

        # create the rcontrib command and run it
        mtx_file = os.path.abspath(os.path.join(folder, '{}.mtx'.format(name)))
        rcontrib = Rcontrib(options=options, octree=octree, sensors=ray_file)
        cmd = rcontrib.to_radiance().replace('\\', '/')
        cmd = '{} | rmtxop -ff - -c .333 .333 .334 > "{}"'.format(cmd, mtx_file.replace('\\', '/'))
        run_command(cmd, env=folders.env)

        # load the resulting matrix and process the results into view factors
        array = binary_to_array(mtx_file)
        view_fac_mtx = []
        for i in range(0, len(array), ray_count):
            sens_chunk = array[i:i+ray_count]
            s_facs = np.sum(sens_chunk, axis=0) / (math.pi * ray_count)
            view_fac_mtx.append(s_facs)

        view_fac_mtx = np.stack(view_fac_mtx)

        np.save(os.path.join(folder, '{}'.format(name)), view_fac_mtx)

    except Exception:
        _logger.exception('Failed to compute view factor contributions.')
        sys.exit(1)
    else:
        sys.exit(0)


def _fibonacci_spiral(point_count=24):
    """Get points distributed uniformly across a unit spherical surface.

    Args:
        point_count: Integer for the number of points to be distributed.

    Returns:
        List of tuple, each with 3 values representing the XYZ coordinates of
        the points that were generated.
    """
    points = []
    phi = math.pi * (3. - math.sqrt(5.))

    for i in range(point_count):
        y = 1 - (i / float(point_count - 1)) * 2
        radius = math.sqrt(1 - y * y)
        theta = phi * i
        x = math.cos(theta) * radius
        z = math.sin(theta) * radius
        points.append((x, y, z))

    return points
