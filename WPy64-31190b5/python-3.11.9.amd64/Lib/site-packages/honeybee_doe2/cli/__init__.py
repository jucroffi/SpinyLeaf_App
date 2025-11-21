"""honeybee-doe2 commands which will be added to the honeybee cli"""
import click
from honeybee.cli import main

from .translate import translate


@click.group(help='honeybee doe2 commands.')
@click.version_option()
def doe2():
    pass


doe2.add_command(translate)

# add doe2 sub-commands
main.add_command(doe2)
