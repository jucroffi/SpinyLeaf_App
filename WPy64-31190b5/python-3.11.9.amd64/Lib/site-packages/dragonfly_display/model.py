"""Method to translate a Dragonfly Model to a VisualizationSet."""
from ladybug_geometry.geometry3d import Point3D
from honeybee_display.model import model_to_vis_set as hb_model_to_vis_set
from honeybee_display.model import model_comparison_to_vis_set as \
    hb_model_comparison_to_vis_set


def model_to_vis_set(
        model, use_multiplier=True, exclude_plenums=False,
        solve_ceiling_adjacencies=False,
        color_by='type', include_wireframe=True, use_mesh=True,
        hide_color_by=False, room_attrs=None, face_attrs=None,
        grid_display_mode='Default', hide_grid=False, reset_coordinates=False):
    """Translate a Dragonfly Model to a VisualizationSet.

    Args:
        model: A Dragonfly Model object to be converted to a VisualizationSet.
        use_multiplier: If True, the multipliers on this Model's Stories will be
            passed along to the generated Honeybee Room objects, indicating the
            simulation will be run once for each unique room and then results
            will be multiplied. If False, full geometry objects will be written
            for each and every floor in the building that are represented through
            multipliers and all resulting multipliers will be 1. (Default: True).
        exclude_plenums: Boolean to indicate whether ceiling/floor plenum depths
            assigned to Room2Ds should be ignored during translation. This
            results in each Room2D translating to a single Honeybee Room at
            the full floor_to_ceiling_height instead of a base Room with (a)
            plenum Room(s). (Default: False).
        solve_ceiling_adjacencies: Boolean to note whether adjacencies should be
            solved between interior stories when Room2D floor and ceiling
            geometries are coplanar. This ensures that Surface boundary
            conditions are used instead of Adiabatic ones. Note that this input
            has no effect when the object_per_model is Story. (Default: False).
        color_by: Text that dictates the colors of the Model geometry.
            If none, only a wireframe of the Model will be generated, assuming
            include_wireframe is True. This is useful when the primary purpose of
            the visualization is to display results in relation to the Model
            geometry or display some room_attrs or face_attrs as an AnalysisGeometry
            or Text labels. (Default: type). Choose from the following:

            * type
            * boundary_condition
            * None

        include_wireframe: Boolean to note whether a ContextGeometry dedicated to
            the Model Wireframe (in DisplayLineSegment3D) should be included
            in the output VisualizationSet. (Default: True).
        use_mesh: Boolean to note whether the colored model geometries should
            be represented with DisplayMesh3D objects (True) instead of DisplayFace3D
            objects (False). Meshes can usually be rendered faster and they scale
            well for large models but all geometry is triangulated (meaning that
            the wireframe in certain platforms might not appear ideal). (Default: True).
        hide_color_by: Boolean to note whether the color_by geometry should be
            hidden or shown by default. Hiding the color-by geometry is useful
            when the primary purpose of the visualization is to display grid_data
            or room/face attributes but it is still desirable to have the option
            to turn on the geometry.
        room_attrs: An optional list of room attribute objects from the
            honeybee_display.attr module.
        face_attrs: An optional list of face attribute objects from the
            honeybee_display.attr module.
        grid_display_mode: Text that dictates how the ContextGeometry for Model
            SensorGrids should display in the resulting visualization. The Default
            option will draw sensor points. Choose from the following:

            * Default
            * Points
            * Wireframe
            * Surface
            * SurfaceWithEdges
            * None

        hide_grid: Boolean to note whether the SensorGrid ContextGeometry should be
            hidden or shown by default. (Default: False).
        reset_coordinates: Boolean to note whether the coordinate system of the
            model should be reset in the resulting visualization set such that
            the model sits at the origin. This is useful when the resulting
            visualization platform is meant to orbit around the world
            origin. (Default: False).

    Returns:
        A VisualizationSet object that represents the model.
    """
    # reset the coordinate system if requested
    if reset_coordinates:
        min_pt = model.min
        z_val = model.average_height - model.average_height_above_ground
        center = Point3D(min_pt.x, min_pt.y, z_val)
        model.reset_coordinate_system(center)
    # create the Honeybee Model from the Dragonfly one
    hb_model = model.to_honeybee(
        'District', use_multiplier=use_multiplier, exclude_plenums=exclude_plenums,
        solve_ceiling_adjacencies=solve_ceiling_adjacencies,
        enforce_adj=False, enforce_solid=True)[0]
    # convert the Honeybee Model to a VisualizationSet
    return hb_model_to_vis_set(
        hb_model, color_by, include_wireframe, use_mesh, hide_color_by,
        room_attrs, face_attrs, grid_display_mode, hide_grid)


def model_comparison_to_vis_set(
        base_model, incoming_model, use_multiplier=True, exclude_plenums=False,
        solve_ceiling_adjacencies=False, base_color=None, incoming_color=None,
        reset_coordinates=False):
    """Translate two Dragonfly Models to be compared to a VisualizationSet.

    Args:
        base_model: A Dragonfly Model object for the base model used in the
            comparison. Typically, this is the model with more data to be kept.
        incoming_model: A Dragonfly Model object for the incoming model used in the
            comparison. Typically, this is the model with new data to be
            evaluated against the base model.
        use_multiplier: If True, the multipliers on this Model's Stories will be
            passed along to the generated Honeybee Room objects, indicating the
            simulation will be run once for each unique room and then results
            will be multiplied. If False, full geometry objects will be written
            for each and every floor in the building that are represented through
            multipliers and all resulting multipliers will be 1. (Default: True).
        exclude_plenums: Boolean to indicate whether ceiling/floor plenum depths
            assigned to Room2Ds should be ignored during translation. This
            results in each Room2D translating to a single Honeybee Room at
            the full floor_to_ceiling_height instead of a base Room with (a)
            plenum Room(s). (Default: False).
        solve_ceiling_adjacencies: Boolean to note whether adjacencies should be
            solved between interior stories when Room2D floor and ceiling
            geometries are coplanar. This ensures that Surface boundary
            conditions are used instead of Adiabatic ones. Note that this input
            has no effect when the object_per_model is Story. (Default: False).
        base_color: An optional ladybug Color to set the color of the base model.
            If None, a default blue color will be used. (Default: None).
        incoming_color: An optional ladybug Color to set the color of the incoming model.
            If None, a default red color will be used. (Default: None).
            reset_coordinates: Boolean to note whether the coordinate system of the
            model should be reset in the resulting visualization set such that
            the model sits at the origin. This is useful when the resulting
            visualization platform is meant to orbit around the world
            origin. (Default: False).
    """
    # reset the coordinate system if requested
    if reset_coordinates:
        min_pt = base_model.min
        z_val = base_model.average_height - base_model.average_height_above_ground
        center = Point3D(min_pt.x, min_pt.y, z_val)
        base_model.reset_coordinate_system(center)
        incoming_model.reset_coordinate_system(center)
    # create the Honeybee Models from the Dragonfly ones
    base_model = base_model.to_honeybee(
        'District', use_multiplier=use_multiplier, exclude_plenums=exclude_plenums,
        solve_ceiling_adjacencies=solve_ceiling_adjacencies,
        enforce_adj=False, enforce_solid=True)[0]
    incoming_model = incoming_model.to_honeybee(
        'District', use_multiplier=use_multiplier, exclude_plenums=exclude_plenums,
        solve_ceiling_adjacencies=solve_ceiling_adjacencies,
        enforce_adj=False, enforce_solid=True)[0]
    # convert the Honeybee Model to a VisualizationSet
    return hb_model_comparison_to_vis_set(
        base_model, incoming_model, base_color, incoming_color)
