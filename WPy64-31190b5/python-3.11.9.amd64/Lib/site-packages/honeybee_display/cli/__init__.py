"""honeybee-display commands."""
import click
import sys
import os
import logging
import json
import base64
import pickle
import tempfile
import uuid

from ladybug.color import Color
from honeybee.model import Model
from honeybee.cli import main

from honeybee_display.attr import FaceAttribute, RoomAttribute

_logger = logging.getLogger(__name__)


# command group for all display extension commands.
@click.group(help='honeybee display commands.')
@click.version_option()
def display():
    pass


@display.command('model-to-vis')
@click.argument('model-file', type=click.Path(
    exists=True, file_okay=True, dir_okay=False, resolve_path=True))
@click.option(
    '--color-by', '-c', help='Text for the property that dictates the colors of '
    'the Model geometry. Choose from: type, boundary_condition, none. '
    'If none, only a wireframe of the Model will be generated (assuming the '
    '--exclude-wireframe option is not used). None is useful when the primary '
    'purpose of  the visualization is to display results in relation to the Model '
    'geometry or display some room_attr or face_attr as an AnalysisGeometry '
    'or Text labels.', type=str, default='type', show_default=True)
@click.option(
    '--wireframe/--exclude-wireframe', ' /-xw', help='Flag to note whether a '
    'ContextGeometry dedicated to the Model Wireframe (in DisplayLineSegment3D) should '
    'be included in the output VisualizationSet.', default=True, show_default=True)
@click.option(
    '--mesh/--faces', help='Flag to note whether the colored model geometries should '
    'be represented with DisplayMesh3D objects instead of DisplayFace3D objects. '
    'Meshes can usually be rendered  faster and they scale well for large models '
    'but all geometry is triangulated (meaning that their wireframe in certain '
    'platforms might not appear ideal).', default=True, show_default=True)
@click.option(
    '--show-color-by/--hide-color-by', ' /-hcb', help='Flag to note whether the '
    'color-by geometry should be hidden or shown by default. Hiding the color-by '
    'geometry is useful when the primary purpose of the visualization is to display '
    'grid-data or room/face attributes but it is still desirable to have the option '
    'to turn on the geometry.', default=True, show_default=True)
@click.option(
    '--room-attr', '-r', help='An optional text string of an attribute that the Model '
    'Rooms have, which will be used to construct a visualization of this attribute '
    'in the resulting VisualizationSet. Multiple instances of this option can be passed '
    'and a separate VisualizationData will be added to the AnalysisGeometry that '
    'represents the attribute in the resulting VisualizationSet (or a separate '
    'ContextGeometry layer if --text-attr is True). Room attributes '
    'input here can have . that separates the nested attributes from '
    'one another. For example, properties.energy.program_type.',
    type=click.STRING, multiple=True, default=None, show_default=True)
@click.option(
    '--face-attr', '-f', help='An optional text string of an attribute that the Model '
    'Faces have, which will be used to construct a visualization of this attribute in '
    'the resulting VisualizationSet. Multiple instances of this option can be passed and'
    ' a separate VisualizationData will be added to the AnalysisGeometry that '
    'represents the attribute in the resulting VisualizationSet (or a separate '
    'ContextGeometry layer if --text-attr is True). Face attributes '
    'input here can have . that separates the nested attributes from '
    'one another. For example, properties.energy.construction.',
    type=click.STRING, multiple=True, default=None, show_default=True)
@click.option(
    '--color-attr', 'attr_display', flag_value='color', help='Flag to note whether the '
    'input room-attr and face-attr should be expressed as a colored AnalysisGeometry.',
    default=True, show_default=True)
@click.option(
    '--text-attr', 'attr_display', flag_value='text', help='Flag to note whether the '
    'input room-attr and face-attr should be expressed as a ContextGeometry with '
    'text labels.')
@click.option(
    '--both-attr', 'attr_display', flag_value='both', help='Flag to note whether the '
    'input room-attr and face-attr should be expressed as both a colored AnalysisGeometry '
    'and a ContextGeometry with text labels.')
@click.option(
    '--grid-display-mode', '-m', help='Text that dictates how the ContextGeometry '
    'for Model SensorGrids should display in the resulting visualization. The Default '
    'option will draw sensor points whenever there is no grid_data_path and will not '
    'draw them at all when grid data is provided, assuming the AnalysisGeometry of '
    'the grids is sufficient. Choose from: Default, Points, Wireframe, Surface, '
    'SurfaceWithEdges, None.',
    type=str, default='Default', show_default=True)
@click.option(
    '--hide-grid/--show-grid', ' /-sg', help='Flag to note whether the SensorGrid '
    'ContextGeometry should be hidden or shown by default.',
    default=True, show_default=True)
@click.option(
    '--grid-data', '-g', help='An optional path to a folder containing data that '
    'aligns with the SensorGrids in the model. Any sub folder within this path '
    'that contains a grids_into.json (and associated CSV files) will be '
    'converted to an AnalysisGeometry in the resulting VisualizationSet. '
    'If a vis_metadata.json file is found within this sub-folder, the '
    'information contained within it will be used to customize the '
    'AnalysisGeometry. Note that it is acceptable if data and '
    'grids_info.json exist in the root of this grid_data_path. Also '
    'note that this argument has no impact if honeybee-radiance is not '
    'installed and SensorGrids cannot be decoded.',
    default=None, show_default=True,
    type=click.Path(file_okay=False, dir_okay=True, resolve_path=True))
@click.option(
    '--grid-data-display-mode', '-dm', help='Text to set the display_mode of the '
    'AnalysisGeometry that is generated from the grid_data_path above. Note '
    'that this has no effect if there are no meshes associated with the model '
    'SensorGrids. Choose from: Surface, SurfaceWithEdges, Wireframe, Points',
    type=str, default='Surface', show_default=True)
@click.option(
    '--active-grid-data', '-ad', help='Text to specify the active data in the '
    'AnalysisGeometry. This should match the name of the sub-folder '
    'within the grid_data_path that should be active. If unspecified, the '
    'first data set in the grid-data with be active.',
    type=str, default=None, show_default=True)
@click.option(
    '--output-format', '-of', help='Text for the output format of the resulting '
    'VisualizationSet File (.vsf). Choose from: vsf, json, pkl, vtkjs, html. Note '
    'that both vsf and json refer to the the JSON version of the VisualizationSet '
    'file and the distinction between the two is only for help in coordinating file '
    'extensions (since both .vsf and .json can be acceptable). Also note that '
    'ladybug-vtk must be installed in order for the vtkjs or html options to be usable '
    'and the html format refers to a web page with the vtkjs file embedded within it.',
    type=str, default='vsf', show_default=True)
@click.option(
    '--output-file', help='Optional file to output the he string of the visualization '
    'file contents. By default, it will be printed out to stdout',
    type=click.File('w'), default='-', show_default=True)
def model_to_vis_set_cli(
        model_file, color_by, wireframe, mesh, show_color_by,
        room_attr, face_attr, attr_display, grid_display_mode, hide_grid,
        grid_data, grid_data_display_mode, active_grid_data, output_format, output_file):
    """Translate a Honeybee Model file (.hbjson) to a VisualizationSet.

    This command can also optionally translate the Honeybee Model to a .vtkjs file,
    which can be visualized in the open source Visual ToolKit (VTK) platform.

    \b
    Args:
        model_file: Full path to a Honeybee Model (HBJSON or HBpkl) file.
    """
    try:
        # process all of the CLI input so that it can be passed to the function
        exclude_wireframe = not wireframe
        faces = not mesh
        hide_color_by = not show_color_by
        room_attrs = [] if len(room_attr) == 0 or room_attr[0] == '' else room_attr
        face_attrs = [] if len(face_attr) == 0 or face_attr[0] == '' else face_attr
        show_grid = not hide_grid
        if attr_display == 'color':
            color_attr, text_attr, both_attr = True, False, False
        elif attr_display == 'text':
            color_attr, text_attr, both_attr = False, True, False
        elif attr_display == 'both':
            color_attr, text_attr, both_attr = False, False, True

        # pass the input to the function in order to convert the model to a visualization
        model_to_vis_set(model_file, color_by, exclude_wireframe, faces, hide_color_by,
                         room_attrs, face_attrs, text_attr, both_attr, grid_display_mode,
                         show_grid, grid_data, grid_data_display_mode, active_grid_data,
                         output_format, output_file, color_attr=color_attr)
    except Exception as e:
        _logger.exception('Failed to translate Model to VisualizationSet.\n{}'.format(e))
        sys.exit(1)
    else:
        sys.exit(0)


def model_to_vis_set(
    model_file, color_by='type',
    exclude_wireframe=False, faces=False, hide_color_by=False,
    room_attr=(), face_attr=(), text_attr=False, both_attr=False,
    grid_display_mode='Default', show_grid=False, grid_data=None,
    grid_data_display_mode='Surface', active_grid_data=None,
    output_format='vsf', output_file=None,
    wireframe=True, mesh=True, show_color_by=True, color_attr=True, hide_grid=True
):
    """Translate a Honeybee Model file (.hbjson) to a VisualizationSet.

    This function can also optionally translate the Honeybee Model to a .vtkjs file,
    which can be visualized in the open source Visual ToolKit (VTK) platform.

    Args:
        model_file: Path to a Honeybee Model (HBJSON or HBpkl) file.
        color_by: Text for the property that dictates the colors of the Model
            geometry. Choose from: type, boundary_condition, none. If none, only
            a wireframe of the Model will be generated (assuming the exclude_wireframe
            option is not used). None is useful when the primary purpose of  the
            visualization is to display results in relation to the Model geometry
            or display some room_attr or face_attr as an AnalysisGeometry or Text labels.
        exclude_wireframe: Boolean to note whether a ContextGeometry dedicated to
            the Model Wireframe (in DisplayLineSegment3D) should be included in
            the output visualization.
        faces: Boolean to note whether the colored model geometries should be
            represented with DisplayMesh3D objects instead of DisplayFace3D objects.
            Meshes can usually be rendered  faster and they scale well for large models
            but all geometry is triangulated (meaning that their wireframe in certain
            platforms might not appear ideal).
        hide_color_by: Boolean to note whether the color-by geometry should be
            hidden or shown by default. Hiding the color-by geometry is useful
            when the primary purpose of the visualization is to display grid_data
            or room/face attributes but it is still desirable to have the option
            to turn on the geometry.
        room_attr: An optional text string of an attribute that the Model Rooms
            have, which will be used to construct a visualization of this attribute
            in the resulting VisualizationSet. A list of text can also
            be passed and a separate VisualizationData will be added to the
            AnalysisGeometry that represents the attribute in the resulting
            VisualizationSet (or a separate ContextGeometry layer if text_attr
            is True). Room attributes input here can have . that separates the nested
            attributes from one another. For example, properties.energy.program_type.
        face_attr: An optional text string of an attribute that the Model Faces
            have, which will be used to construct a visualization of this attribute
            in the resulting VisualizationSet. A list of text can also be passed and
            a separate VisualizationData will be added to the AnalysisGeometry that '
            represents the attribute in the resulting VisualizationSet (or a separate '
            ContextGeometry layer if text_attr is True). Face attributes input
            here can have . that separates the nested attributes from one another.
            For example, properties.energy.construction.
        text_attr: Boolean to note whether to note whether the input room_attr
            and face_attr should be expressed as a ContextGeometry with text labels.
        both_attr: Boolean to note whether to note whether the input room_attr
            and face_attr should be expressed as both a colored AnalysisGeometry
            and a ContextGeometry as text labels.
        grid_display_mode: Text that dictates how the ContextGeometry for Model
            SensorGrids should display in the resulting visualization. The Default
            option will draw sensor points whenever there is no grid_data_path
            and will not draw them at all when grid data is provided, assuming
            the AnalysisGeometry of the grids is sufficient. Choose from: Default,
            Points, Wireframe, Surface, SurfaceWithEdges, None.
        show_grid: Boolean to note whether the SensorGrid ContextGeometry should
            be hidden or shown by default.
        grid_data: An optional path to a folder containing data that aligns
            with the SensorGrids in the model. Any sub folder within this path
            that contains a grids_into.json (and associated CSV files) will be
            converted to an AnalysisGeometry in the resulting VisualizationSet.
            If a vis_metadata.json file is found within this sub-folder, the
            information contained within it will be used to customize the
            AnalysisGeometry. Note that it is acceptable if data and
            grids_info.json exist in the root of this grid_data_path. Also
            note that this argument has no impact if honeybee-radiance is not
            installed and SensorGrids cannot be decoded. (Default: None).
        grid_data_display_mode: Optional text to set the display_mode of the
            AnalysisGeometry that is is generated from the grid_data_path above. Note
            that this has no effect if there are no meshes associated with the model
            SensorGrids. (Default: Surface). Choose from the following:

            * Surface
            * SurfaceWithEdges
            * Wireframe
            * Points

        active_grid_data: Optional text to specify the active data in the
            AnalysisGeometry. This should match the name of the sub-folder
            within the grid_data_path that should be active. If None, the
            first data set in the grid_data_path with be active. (Default: None).
        output_format: Text for the output format of the resulting VisualizationSet
            File (.vsf). Choose from: vsf, json, pkl, vtkjs, html. Note that both
            vsf and json refer to the the JSON version of the VisualizationSet
            file and the distinction between the two is only for help in
            coordinating file extensions (since both .vsf and .json can be
            acceptable). Also note that ladybug-vtk must be installed in order
            for the vtkjs or html options to be usable and the html format
            refers to a web page with the vtkjs file embedded within it.
        output_file: Optional file to output the string of the visualization
            file contents. If None, the string will simply be returned from
            this method.
    """
    # load the model object and process simpler attributes
    model_obj = Model.from_file(model_file)
    room_attrs = [room_attr] if isinstance(room_attr, str) else room_attr
    face_attrs = [face_attr] if isinstance(face_attr, str) else face_attr
    wireframe = not exclude_wireframe
    mesh = not faces
    hide_grid = not show_grid

    # load the room and face attributes
    if both_attr:
        color_attr, text_attr = True, True
    elif text_attr:
        color_attr = False
    face_attributes = []
    for fa in face_attrs:
        faa = FaceAttribute(name=fa, attrs=[fa], color=color_attr, text=text_attr)
        face_attributes.append(faa)
    room_attributes = []
    for ra in room_attrs:
        raa = RoomAttribute(name=ra, attrs=[ra], color=color_attr, text=text_attr)
        room_attributes.append(raa)

    # create the VisualizationSet
    vis_set = model_obj.to_vis_set(
        color_by=color_by, include_wireframe=wireframe, use_mesh=mesh,
        hide_color_by=hide_color_by, room_attrs=room_attributes,
        face_attrs=face_attributes, grid_display_mode=grid_display_mode,
        hide_grid=hide_grid, grid_data_path=grid_data,
        grid_data_display_mode=grid_data_display_mode,
        active_grid_data=active_grid_data)

    # output the VisualizationSet through the CLI
    return _output_vis_set_to_format(vis_set, output_format, output_file)


@display.command('model-comparison-to-vis')
@click.argument('base-model-file', type=click.Path(
    exists=True, file_okay=True, dir_okay=False, resolve_path=True))
@click.argument('incoming-model-file', type=click.Path(
    exists=True, file_okay=True, dir_okay=False, resolve_path=True))
@click.option(
    '--base-color', '-bc', help='An optional hexadecimal code for the color '
    'of the base model.', type=str, default='#74eded', show_default=True)
@click.option(
    '--incoming-color', '-ic', help='An optional hexadecimal code for the color '
    'of the incoming model.', type=str, default='#ed7474', show_default=True)
@click.option(
    '--output-format', '-of', help='Text for the output format of the resulting '
    'VisualizationSet File (.vsf). Choose from: vsf, json, pkl, vtkjs, html. Note '
    'that both vsf and json refer to the the JSON version of the VisualizationSet '
    'file and the distinction between the two is only for help in coordinating file '
    'extensions (since both .vsf and .json can be acceptable). Also note that '
    'ladybug-vtk must be installed in order for the vtkjs or html options to be usable '
    'and the html format refers to a web page with the vtkjs file embedded within it.',
    type=str, default='vsf', show_default=True)
@click.option(
    '--output-file', help='Optional file to output the he string of the visualization '
    'file contents. By default, it will be printed out to stdout',
    type=click.File('w'), default='-', show_default=True)
def model_comparison_to_vis_set_cli(
        base_model_file, incoming_model_file, base_color, incoming_color,
        output_format, output_file):
    """Translate two Honeybee Models to be compared to a VisualizationSet.

    This command can also optionally translate the Honeybee Model to a .vtkjs file,
    which can be visualized in the open source Visual ToolKit (VTK) platform.

    \b
    Args:
        base_model_file: Full path to a Honeybee Model (HBJSON or HBpkl) file
            representing the base model used in the comparison. Typically, this
            is the model with more data to be kept.
        incoming_model_file: Full path to a Honeybee Model (HBJSON or HBpkl) file
            representing the incoming model used in the comparison. Typically,
            this is the model with new data to be evaluated against the base model.
    """
    try:
        model_comparison_to_vis_set(
            base_model_file, incoming_model_file, base_color, incoming_color,
            output_format, output_file)
    except Exception as e:
        _logger.exception('Failed to translate Model to VisualizationSet.\n{}'.format(e))
        sys.exit(1)
    else:
        sys.exit(0)


def model_comparison_to_vis_set(
    base_model_file, incoming_model_file,
    base_color='#74eded', incoming_color='#ed7474',
    output_format='vsf', output_file=None,
):
    """Translate two Honeybee Models to be compared to a VisualizationSet.

    This command can also optionally translate the Honeybee Model to a .vtkjs file,
    which can be visualized in the open source Visual ToolKit (VTK) platform.

    Args:
        base_model_file: Full path to a Honeybee Model (HBJSON or HBpkl) file
            representing the base model used in the comparison. Typically, this
            is the model with more data to be kept.
        incoming_model_file: Full path to a Honeybee Model (HBJSON or HBpkl) file
            representing the incoming model used in the comparison. Typically,
            this is the model with new data to be evaluated against the base model.
        base_color: An optional hexadecimal code for the color of the base
            model. (Default: #74eded).
        incoming_color: An optional hexadecimal code for the color of the incoming
            model. (Default: #ed7474).
        output_format: Text for the output format of the resulting VisualizationSet
            File (.vsf). Choose from: vsf, json, pkl, vtkjs, html. Note that both
            vsf and json refer to the the JSON version of the VisualizationSet
            file and the distinction between the two is only for help in
            coordinating file extensions (since both .vsf and .json can be
            acceptable). Also note that ladybug-vtk must be installed in order
            for the vtkjs or html options to be usable and the html format
            refers to a web page with the vtkjs file embedded within it.
        output_file: Optional file to output the string of the visualization
            file contents. If None, the string will simply be returned from
            this method.
    """
    # load the model objects and process the colors from the hex codes
    base_model = Model.from_file(base_model_file)
    incoming_model = Model.from_file(incoming_model_file)
    base_color = Color.from_hex(base_color)
    incoming_color = Color.from_hex(incoming_color)
    base_color.a = 128
    incoming_color.a = 128

    # create the VisualizationSet
    vis_set = base_model.to_vis_set_comparison(
        incoming_model, base_color, incoming_color)

    # output the VisualizationSet through the CLI
    return _output_vis_set_to_format(vis_set, output_format, output_file)


def _output_vis_set_to_format(vis_set, output_format, output_file):
    """Process a VisualizationSet for output from the CLI.

    Args:
        vis_set: The VisualizationSet to be output form the CLI.
        output_format: Text for the output format of the resulting VisualizationSet File.
        output_file: Optional file to output the string of the visualization
            file contents. If None, the string will simply be returned from
            this method.
    """
    # output the visualization in the correct format
    output_format = output_format.lower()
    if output_format in ('vsf', 'json'):
        if output_file is None:
            return json.dumps(vis_set.to_dict())
        elif isinstance(output_file, str):
            with open(output_file, 'w') as of:
                of.write(json.dumps(vis_set.to_dict()))
        else:
            output_file.write(json.dumps(vis_set.to_dict()))
    elif output_format == 'pkl':
        if output_file is None:
            return pickle.dumps(vis_set.to_dict())
        elif isinstance(output_file, str):
            with open(output_file, 'w') as of:
                of.write(pickle.dumps(vis_set.to_dict()))
        elif output_file.name == '<stdout>':
            output_file.write(pickle.dumps(vis_set.to_dict()))
        else:
            out_folder, out_file = os.path.split(output_file.name)
            vis_set.to_pkl(out_file, out_folder)
    elif output_format in ('vtkjs', 'html'):
        if output_file is None or (not isinstance(output_file, str)
                                   and output_file.name == '<stdout>'):
            # get a temporary file
            out_file = str(uuid.uuid4())[:6]
            out_folder = tempfile.gettempdir()
        else:
            f_path = output_file if isinstance(output_file, str) else output_file.name
            out_folder, out_file = os.path.split(f_path)
            if out_file.endswith('.vtkjs'):
                out_file = out_file[:-6]
            elif out_file.endswith('.html'):
                out_file = out_file[:-5]
        try:
            if output_format == 'vtkjs':
                vis_set.to_vtkjs(output_folder=out_folder, file_name=out_file)
            if output_format == 'html':
                vis_set.to_html(output_folder=out_folder, file_name=out_file)
        except AttributeError as ae:
            raise AttributeError(
                'Ladybug-vtk must be installed in order to use --output-format '
                'vtkjs.\n{}'.format(ae))
        if output_file is None or (not isinstance(output_file, str)
                                   and output_file.name == '<stdout>'):
            # load file contents
            out_file_ext = out_file + '.' + output_format
            out_file_path = os.path.join(out_folder, out_file_ext)
            if output_format == 'html':
                with open(out_file_path, encoding='utf-8') as of:
                    f_contents = of.read()
            else:  # vtkjs can only be read as binary
                with open(out_file_path, 'rb') as of:
                    f_contents = of.read()
                b = base64.b64encode(f_contents)
                f_contents = b.decode('utf-8')
            if output_file is None:
                return f_contents
            output_file.write(f_contents)
    else:
        raise ValueError('Unrecognized output-format "{}".'.format(output_format))


# add display sub-group to honeybee CLI
main.add_command(display)
