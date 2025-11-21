# coding=utf-8
"""OpenStudio ConstructionSet translator."""
from __future__ import division

from honeybee.typing import clean_ep_string
from honeybee_energy.construction.windowshade import WindowConstructionShade
from honeybee_energy.construction.dynamic import WindowConstructionDynamic
from honeybee_energy.constructionset import ConstructionSet

from honeybee_openstudio.openstudio import OSDefaultConstructionSet, \
    OSDefaultSurfaceConstructions, OSDefaultSubSurfaceConstructions
from honeybee_openstudio.construction import shade_construction_from_openstudio


def _assign_construction_to_subset(construction, os_constr_subset, face_type, os_model):
    """Assign a Honeybee construction object to an OpenStudio sub-set.

    Args:
        construction: The honeybee-energy construction object assigned to the
            ConstructionsSet (this can be None).
        os_constr_subset: The OpenStudio DefaultSurfaceConstructions object
            to which the construction will be added.
        face_type: Text for the type of Face to which the construction will be
            added. Must be either Wall, Floor or RoofCeiling.
        os_model: The OpenStudio Model.
    """
    if construction is not None:
        construction_ref = os_model.getConstructionByName(construction.identifier)
        if construction_ref.is_initialized():
            os_construction = construction_ref.get()
            if face_type == 'Wall':
                os_constr_subset.setWallConstruction(os_construction)
            elif face_type == 'Floor':
                os_constr_subset.setFloorConstruction(os_construction)
            else:
                os_constr_subset.setRoofCeilingConstruction(os_construction)


def _glazing_construction(construction, os_model):
    """Get an OpenStudio window construction with a check for dynamic constructions."""
    if construction is None:
        return None
    elif isinstance(construction, WindowConstructionShade):
        construction_id = construction.window_construction.identifier
    elif isinstance(construction, WindowConstructionDynamic):
        construction_id = '{}State0'.format(construction.constructions[0].identifier)
    else:
        construction_id = construction.identifier
    constr_ref = os_model.getConstructionByName(construction_id)
    if constr_ref.is_initialized():
        os_construction = constr_ref.get()
        return os_construction


def construction_set_to_openstudio(construction_set, os_model):
    """Convert Honeybee ConstructionSet to OpenStudio DefaultConstructionSet."""
    # create the construction set object
    os_constr_set = OSDefaultConstructionSet(os_model)
    os_constr_set.setName(construction_set.identifier)
    if construction_set._display_name is not None:
        os_constr_set.setDisplayName(construction_set.display_name)

    int_surf_const = OSDefaultSurfaceConstructions(os_model)
    ext_surf_const = OSDefaultSurfaceConstructions(os_model)
    grnd_surf_const = OSDefaultSurfaceConstructions(os_model)
    int_subsurf_const = OSDefaultSubSurfaceConstructions(os_model)
    ext_subsurf_const = OSDefaultSubSurfaceConstructions(os_model)

    os_constr_set.setDefaultInteriorSurfaceConstructions(int_surf_const)
    os_constr_set.setDefaultExteriorSurfaceConstructions(ext_surf_const)
    os_constr_set.setDefaultGroundContactSurfaceConstructions(grnd_surf_const)
    os_constr_set.setDefaultInteriorSubSurfaceConstructions(int_subsurf_const)
    os_constr_set.setDefaultExteriorSubSurfaceConstructions(ext_subsurf_const)

    # determine the frame type for measure tags
    frame_type = 'Metal Framing with Thermal Break' \
        if 'WoodFramed' in construction_set.identifier else 'Non-Metal Framing'

    # assign the constructions in the wall set
    int_con = construction_set.wall_set._interior_construction
    if int_con is not None:
        int_wall_ref = os_model.getConstructionByName(int_con.identifier)
        if int_wall_ref.is_initialized():
            interior_wall = int_wall_ref.get()
            int_surf_const.setWallConstruction(interior_wall)
            os_constr_set.setAdiabaticSurfaceConstruction(interior_wall)
    ext_con = construction_set.wall_set._exterior_construction
    _assign_construction_to_subset(ext_con, ext_surf_const, 'Wall', os_model)
    ground_con = construction_set.wall_set._ground_construction
    _assign_construction_to_subset(ground_con, grnd_surf_const, 'Wall', os_model)

    # assign the constructions in the floor set
    int_con = construction_set.floor_set._interior_construction
    _assign_construction_to_subset(int_con, int_surf_const, 'Floor', os_model)
    ext_con = construction_set.floor_set._exterior_construction
    _assign_construction_to_subset(ext_con, ext_surf_const, 'Floor', os_model)
    ground_con = construction_set.floor_set._ground_construction
    _assign_construction_to_subset(ground_con, grnd_surf_const, 'Floor', os_model)

    # assign the constructions in the roof ceiling set
    int_con = construction_set.roof_ceiling_set._interior_construction
    _assign_construction_to_subset(int_con, int_surf_const, 'RoofCeiling', os_model)
    ext_con = construction_set.roof_ceiling_set._exterior_construction
    _assign_construction_to_subset(ext_con, ext_surf_const, 'RoofCeiling', os_model)
    ground_con = construction_set.roof_ceiling_set._ground_construction
    _assign_construction_to_subset(ground_con, grnd_surf_const, 'RoofCeiling', os_model)

    # assign the constructions in the aperture set
    int_ap_con = construction_set.aperture_set._interior_construction
    int_ap_con = _glazing_construction(int_ap_con, os_model)
    if int_ap_con is not None:
        int_subsurf_const.setFixedWindowConstruction(int_ap_con)
        int_subsurf_const.setOperableWindowConstruction(int_ap_con)
        int_subsurf_const.setSkylightConstruction(int_ap_con)
    win_ap_con = construction_set.aperture_set._window_construction
    win_ap_con = _glazing_construction(win_ap_con, os_model)
    if win_ap_con is not None:
        ext_subsurf_const.setFixedWindowConstruction(win_ap_con)
        std_info = win_ap_con.standardsInformation()
        std_info.setFenestrationType('Fixed Window')
        std_info.setFenestrationFrameType(frame_type)
        std_info.setIntendedSurfaceType('ExteriorWindow')
    sky_ap_con = construction_set.aperture_set._skylight_construction
    sky_ap_con = _glazing_construction(sky_ap_con, os_model)
    if sky_ap_con is not None:
        ext_subsurf_const.setSkylightConstruction(sky_ap_con)
        std_info = sky_ap_con.standardsInformation()
        std_info.setFenestrationType('Fixed Window')
        std_info.setFenestrationFrameType(frame_type)
        if not std_info.intendedSurfaceType().is_initialized():
            std_info.setIntendedSurfaceType('Skylight')
    op_ap_con = construction_set.aperture_set._operable_construction
    op_ap_con = _glazing_construction(op_ap_con, os_model)
    if op_ap_con is not None:
        ext_subsurf_const.setOperableWindowConstruction(op_ap_con)
        std_info = op_ap_con.standardsInformation()
        std_info.setFenestrationFrameType(frame_type)
        std_info.setIntendedSurfaceType('ExteriorWindow')
        if not std_info.intendedSurfaceType().is_initialized():
            std_info.setFenestrationType('Operable Window')

    # assign the constructions in the door set
    int_dr_con = construction_set.door_set._interior_construction
    if int_dr_con is not None:
        int_door_ref = os_model.getConstructionByName(int_dr_con.identifier)
        if int_door_ref.is_initialized():
            interior_door = int_door_ref.get()
            int_subsurf_const.setDoorConstruction(interior_door)
            int_subsurf_const.setOverheadDoorConstruction(interior_door)
    ext_dr_con = construction_set.door_set._exterior_construction
    if ext_dr_con is not None:
        ext_door_ref = os_model.getConstructionByName(ext_dr_con.identifier)
        if ext_door_ref.is_initialized():
            exterior_door = ext_door_ref.get()
            ext_subsurf_const.setDoorConstruction(exterior_door)
            std_info = exterior_door.standardsInformation()
            if not std_info.intendedSurfaceType().is_initialized():
                std_info.setIntendedSurfaceType('ExteriorDoor')
    ov_dr_con = construction_set.door_set._overhead_construction
    if ov_dr_con is not None:
        overhead_door_ref = os_model.getConstructionByName(ov_dr_con.identifier)
        if overhead_door_ref.is_initialized():
            overhead_door = overhead_door_ref.get()
            ext_subsurf_const.setOverheadDoorConstruction(overhead_door)
            std_info = overhead_door.standardsInformation()
            if not std_info.intendedSurfaceType().is_initialized():
                std_info.setIntendedSurfaceType('OverheadDoor')
    ext_glz_for_con = construction_set.door_set._exterior_glass_construction
    ext_glz_for_con = _glazing_construction(ext_glz_for_con, os_model)
    if ext_glz_for_con is not None:
        ext_subsurf_const.setGlassDoorConstruction(ext_glz_for_con)
        std_info = ext_glz_for_con.standardsInformation()
        if not std_info.fenestrationType().is_initialized():
            std_info.setFenestrationType('Glazed Door')
        std_info.setFenestrationFrameType(frame_type)
        if not std_info.intendedSurfaceType().is_initialized():
            std_info.setIntendedSurfaceType('GlassDoor')
    int_glz_for_con = construction_set.door_set._interior_glass_construction
    int_glz_for_con = _glazing_construction(int_glz_for_con, os_model)
    if int_glz_for_con is not None:
        int_subsurf_const.setGlassDoorConstruction(int_glz_for_con)

    # assign the shading construction to construction set
    shade_con = construction_set._shade_construction
    if shade_con is not None:
        shade_ref = os_model.getConstructionByName(shade_con.identifier)
        if shade_ref.is_initialized():
            shade_construction = shade_ref.get()
            os_constr_set.setSpaceShadingConstruction(shade_construction)

    return os_constr_set


def construction_set_from_openstudio(os_construction_set, constructions):
    """Convert OpenStudio DefaultConstructionSet to Honeybee ConstructionSet."""
    con_set = ConstructionSet(clean_ep_string(os_construction_set.nameString()))

    # get interior surface constructions
    if os_construction_set.defaultInteriorSurfaceConstructions().is_initialized():
        os_int_set = os_construction_set.defaultInteriorSurfaceConstructions().get()
        if os_int_set.wallConstruction().is_initialized():
            int_wall_const = os_int_set.wallConstruction().get().nameString()
            try:
                int_wall_const = constructions[clean_ep_string(int_wall_const)]
                con_set.wall_set.interior_construction = int_wall_const
            except KeyError:
                pass  # construction that could not be re-serialized
        if os_int_set.floorConstruction().is_initialized():
            int_floor_const = os_int_set.floorConstruction().get().nameString()
            try:
                int_floor_const = constructions[clean_ep_string(int_floor_const)]
                con_set.floor_set.interior_construction = int_floor_const
            except KeyError:
                pass  # construction that could not be re-serialized
        if os_int_set.roofCeilingConstruction().is_initialized():
            int_roof_const = os_int_set.roofCeilingConstruction().get().nameString()
            try:
                int_roof_const = constructions[clean_ep_string(int_roof_const)]
                con_set.roof_ceiling_set.interior_construction = int_roof_const
            except KeyError:
                pass  # construction that could not be re-serialized

    # get interior subsurface constructions
    if os_construction_set.defaultInteriorSubSurfaceConstructions().is_initialized():
        int_subset = os_construction_set.defaultInteriorSubSurfaceConstructions().get()
        if int_subset.fixedWindowConstruction().is_initialized():
            int_wind_const = int_subset.fixedWindowConstruction().get().nameString()
            try:
                int_wind_const = constructions[clean_ep_string(int_wind_const)]
                con_set.aperture_set.interior_construction = int_wind_const
            except KeyError:
                pass  # construction that could not be re-serialized
        if int_subset.doorConstruction().is_initialized():
            int_door_const = int_subset.doorConstruction().get().nameString()
            try:
                int_door_const = constructions[clean_ep_string(int_door_const)]
                con_set.door_set.interior_construction = int_door_const
            except KeyError:
                pass  # construction that could not be re-serialized
        if int_subset.glassDoorConstruction().is_initialized():
            int_glass_door_con = int_subset.glassDoorConstruction().get().nameString()
            try:
                int_glass_door_con = constructions[clean_ep_string(int_glass_door_con)]
                con_set.door_set.interior_glass_construction = int_glass_door_con
            except KeyError:
                pass  # construction that could not be re-serialized

    # get exterior surface constructions
    if os_construction_set.defaultExteriorSurfaceConstructions().is_initialized():
        os_ext_set = os_construction_set.defaultExteriorSurfaceConstructions().get()
        if os_ext_set.wallConstruction().is_initialized():
            ext_wall_const = os_ext_set.wallConstruction().get().nameString()
            try:
                ext_wall_const = constructions[clean_ep_string(ext_wall_const)]
                con_set.wall_set.exterior_construction = ext_wall_const
            except KeyError:
                pass  # construction that could not be re-serialized
        if os_ext_set.floorConstruction().is_initialized():
            ext_floor_const = os_ext_set.floorConstruction().get().nameString()
            try:
                ext_floor_const = constructions[clean_ep_string(ext_floor_const)]
                con_set.floor_set.exterior_construction = ext_floor_const
            except KeyError:
                pass  # construction that could not be re-serialized
        if os_ext_set.roofCeilingConstruction().is_initialized():
            ext_roof_const = os_ext_set.roofCeilingConstruction().get().nameString()
            try:
                ext_roof_const = constructions[clean_ep_string(ext_roof_const)]
                con_set.roof_ceiling_set.exterior_construction = ext_roof_const
            except KeyError:
                pass  # construction that could not be re-serialized

    # get exterior subsurface construction
    if os_construction_set.defaultExteriorSubSurfaceConstructions().is_initialized():
        ext_subset = os_construction_set.defaultExteriorSubSurfaceConstructions().get()
        if ext_subset.fixedWindowConstruction().is_initialized():
            ext_wind_const = ext_subset.fixedWindowConstruction().get().nameString()
            try:
                ext_wind_const = constructions[clean_ep_string(ext_wind_const)]
                con_set.aperture_set.window_construction = ext_wind_const
            except KeyError:
                pass  # construction that could not be re-serialized
        if ext_subset.operableWindowConstruction().is_initialized():
            op_wind_const = ext_subset.operableWindowConstruction().get().nameString()
            try:
                op_wind_const = constructions[clean_ep_string(op_wind_const)]
                con_set.aperture_set.operable_construction = op_wind_const
            except KeyError:
                pass  # construction that could not be re-serialized
        if ext_subset.skylightConstruction().is_initialized():
            ext_skylight_const = ext_subset.skylightConstruction().get().nameString()
            try:
                ext_skylight_const = constructions[clean_ep_string(ext_skylight_const)]
                con_set.aperture_set.skylight_construction = ext_skylight_const
            except KeyError:
                pass  # construction that could not be re-serialized
        if ext_subset.doorConstruction().is_initialized():
            ext_door_const = ext_subset.doorConstruction().get().nameString()
            try:
                ext_door_const = constructions[clean_ep_string(ext_door_const)]
                con_set.door_set.exterior_construction = ext_door_const
            except KeyError:
                pass  # construction that could not be re-serialized
        if ext_subset.overheadDoorConstruction().is_initialized():
            ext_ovhd_door = ext_subset.overheadDoorConstruction().get().nameString()
            try:
                ext_ovhd_door = constructions[clean_ep_string(ext_ovhd_door)]
                con_set.door_set.overhead_construction = ext_ovhd_door
            except KeyError:
                pass  # construction that could not be re-serialized
        if ext_subset.glassDoorConstruction().is_initialized():
            ext_glz_door = ext_subset.glassDoorConstruction().get().nameString()
            try:
                ext_glz_door = constructions[clean_ep_string(ext_glz_door)]
                con_set.door_set.exterior_glass_construction = ext_glz_door
            except KeyError:
                pass  # construction that could not be re-serialized

    # assign the ground construction and other attributes
    if os_construction_set.defaultGroundContactSurfaceConstructions().is_initialized():
        os_gnd_set = os_construction_set.defaultGroundContactSurfaceConstructions().get()
        if os_gnd_set.wallConstruction().is_initialized():
            gnd_wall_const = os_gnd_set.wallConstruction().get().nameString()
            try:
                gnd_wall_const = constructions[clean_ep_string(gnd_wall_const)]
                con_set.wall_set.ground_construction = gnd_wall_const
            except KeyError:
                pass  # construction that could not be re-serialized
        if os_gnd_set.floorConstruction().is_initialized():
            gnd_floor_const = os_gnd_set.floorConstruction().get().nameString()
            try:
                gnd_floor_const = constructions[clean_ep_string(gnd_floor_const)]
                con_set.floor_set.ground_construction = gnd_floor_const
            except KeyError:
                pass  # construction that could not be re-serialized
        if os_gnd_set.roofCeilingConstruction().is_initialized():
            gnd_roof_const = os_gnd_set.roofCeilingConstruction().get().nameString()
            try:
                gnd_roof_const = constructions[clean_ep_string(gnd_roof_const)]
                con_set.roof_ceiling_set.ground_construction = gnd_roof_const
            except KeyError:
                pass  # construction that could not be re-serialized

    # assign shade and other optional attributes
    if os_construction_set.spaceShadingConstruction().is_initialized():
        shade_con = os_construction_set.spaceShadingConstruction().get()
        const_name = '{} Shade'.format(clean_ep_string(shade_con.nameString()))
        try:
            shade_con = constructions[const_name]
        except KeyError:
            const = shade_con.to_LayeredConstruction().get()
            shade_con = shade_construction_from_openstudio(const)
            constructions[const_name] = shade_con
        con_set.shade_construction = shade_con
    if os_construction_set.displayName().is_initialized():
        con_set.display_name = os_construction_set.displayName().get()
    return con_set
