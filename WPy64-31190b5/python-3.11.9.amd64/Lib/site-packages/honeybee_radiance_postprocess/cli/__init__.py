"""Modules for post-processing simulation outputs."""
import click

from honeybee.cli import main
from .grid import grid
from .mtxop import mtxop
from .postprocess import post_process
from .schedule import schedule
from .translate import translate
from .viewfactor import view_factor
from .merge import merge
from .datacollection import datacollection


# command group for all postprocess extension commands.
@click.group(help='honeybee radiance postprocess commands.')
@click.version_option()
def postprocess():
    pass


# add sub-commands to postprocess
postprocess.add_command(grid)
postprocess.add_command(mtxop)
postprocess.add_command(post_process, name='post-process')
postprocess.add_command(schedule)
postprocess.add_command(translate)
postprocess.add_command(view_factor)
postprocess.add_command(merge)
postprocess.add_command(datacollection, name='data-collection')

# add postprocess sub-commands to honeybee CLI
main.add_command(postprocess)
