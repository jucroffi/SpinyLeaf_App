
import sys, json, os
from pathlib import Path
import math
import numpy as np

import compute_rhino3d.Util
import clr


clr.AddReference(r"C:\Program Files\Rhino 8\System\RhinoCommon.dll")


compute_rhino3d.Util.url = "http://localhost:6500/"


import rhinoinside
rhinoinside.load()

import Rhino
import Rhino.FileIO
import Rhino.Geometry as rg
from Rhino.Geometry import Line

from ladybug.color import Colorset
from ladybug_rhino.config import tolerance, angle_tolerance
from ladybug_rhino.togeometry import to_polyface3d, to_face3d
from ladybug_rhino.fromgeometry import from_point3d, from_vector3d
from ladybug_rhino.intersect import join_geometry_to_mesh, intersect_mesh_rays, intersect_mesh_lines
from ladybug.graphic import GraphicContainer
from ladybug.viewsphere import view_sphere

from ladybug_rhino.config import tolerance, angle_tolerance

from ladybug_rhino.togeometry import to_polyface3d, to_face3d
from honeybee.typing import clean_and_id_string
from honeybee.model import Model
from honeybee.room import Room
from honeybee.aperture import Aperture

import Functions.App_Utils as slu
import Functions.App_Create_Model_Simulations_00 as spi
import Functions.App_Rhino_Geo as srg
import Functions.App_Satisfaction_T as sst



def prep_view_assets(payload):
    
    hb_view_path = Path(payload["hb_view_path"])
    out_dir = Path(payload["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    dist = float(payload.get("dist", 1.0))

    view_hb_model = Model.from_hbjson(hb_view_path)

    v_meshes, sensor_grid = spi.create_sensors(view_hb_model, dist)
    view_hb_model.properties.radiance.sensor_grids = sensor_grid
    view_hb_model.to_hbjson(name='view', folder=out_dir)

    v_centr = []
    for m in v_meshes:
        cents = []
        for p in getattr(m, "face_centroids", []):
            cents.append((float(p.x), float(p.y), float(p.z)))
        v_centr.append(cents)

    return {
        "ok": True,
        "view_hb_path": str((out_dir / "view.hbjson").resolve()),
        "v_centr": v_centr
    }


def save_room_scalar_results(payload):
    
    view_hb_path = Path(payload["view_hb_path"])
    out_folder = Path(payload["out_folder"]); out_folder.mkdir(parents=True, exist_ok=True)
    values = list(payload["values"])  
    dist = float(payload.get("sensor_dist", 1.0))

    view_hb_model = Model.from_hbjson(view_hb_path)
    v_meshes, _ = spi.create_sensors(view_hb_model, dist)

    v_centr = []
    for m in v_meshes:
        cents = []
        for p in getattr(m, "face_centroids", []):
            cents.append((float(p.x), float(p.y), float(p.z)))
        v_centr.append(cents)

    
    lists = [[v] * len(m) for v, m in zip(values, v_centr)]

    
    slu.save_res_files(lists, v_meshes, out_folder)

    return {"ok": True, "faces_per_room": [len(m.face_centroids) for m in v_meshes]}


def get_apertures(model, objs):
    windows_layer = 'WINDOWS'
    hb_apertures = [] 
    n=0
    for obj in objs:
        layer_index = obj.Attributes.LayerIndex
        layer = model.Layers[layer_index].Name

        if layer == windows_layer:
            geo = obj.Geometry
            n = n+1
            win_name = clean_and_id_string(windows_layer)
            lb_face = to_face3d(geo)
            hb_apt = Aperture(win_name, lb_face[0])
            hb_apertures.append(hb_apt)           

    return hb_apertures


def check_and_add_sub_face(face, sub_faces):
    
    for sf in sub_faces:
        if face.geometry.is_sub_face(sf.geometry, tolerance, angle_tolerance):
            face.add_aperture(sf)

def add_apertures_all(hb_obj, sub_faces):

    for obj in hb_obj:
        for face in obj.faces:
            check_and_add_sub_face(face, sub_faces)



def get_hbrooms(usage_geo, usage, objs):
    hbrooms = []
    polys = []
    centres = []

    for geo in usage_geo:
        if geo.IsSolid:
            if not isinstance(geo, rg.Brep):
                geo = geo.ToBrep()
            lb_polyf = to_polyface3d(geo)
            polys.append(lb_polyf)
            centres.append(lb_polyf.center.z)

    sorted_data = sorted(zip(centres, polys, usage_geo, objs), key=lambda x: x[0])
    sorted_polys = [p for _, p, _, _ in sorted_data]
    sorted_geos = [g for _, _, g, _ in sorted_data]
    sorted_objs = [o for _, _, _, o in sorted_data]

    for idx, poly in enumerate(sorted_polys):
        room_name = f'{usage}_{idx}'
        hb_room = Room.from_polyface3d(room_name, poly)
        hbrooms.append(hb_room)

    return hbrooms, sorted_geos, sorted_objs




USAGE_DICT = {
    0: "COMMERC", 1: "CONTEXT", 2: "RESID",
    3: "GREEN", 4: "BALCONIES", 5: "CORE",
    6: "SOCIAL_L1", 7: "SOCIAL_L2", 8: "SOCIAL_L3",
    9: "SOCIAL_L4_RESID", 10: "SOCIAL_L4_COMMERC",
    11: "SOCIAL_OUTDOOR_ALL", 12: "SOCIAL_OUTDOOR_RESID", 13: "SOCIAL_OUTDOOR_COMMERC"
}

def build_hb_models(payload):
    
    src_3dm = payload["src_3dm"]
    out_dir = Path(payload["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    window_identifier = payload["window_identifier"]
    wall_type = payload["wall_type"]
    wall_r = float(payload["wall_r"])
    roof_r = float(payload["roof_r"])
    ground_r = float(payload["ground_r"])
    operable = bool(payload["operable"])
    offices_occ_per_area = float(payload["offices_occ_per_area"])
    res_occ_per_bedroom = float(payload["res_occ_per_bedroom"])

    model = Rhino.FileIO.File3dm.Read(src_3dm)
    if model is None:
        return {"ok": False, "error": "Could not read 3DM"}

    objs = model.Objects

    # Context
    usage = 1
    context_breps, _ = srg.get_geo(model, objs, USAGE_DICT[usage])
    context_shades = srg.get_context_shades(context_breps, USAGE_DICT[usage], detached=True)

    # Green
    usage = 3
    green_breps, _ = srg.get_geo(model, objs, USAGE_DICT[usage])
    green_points = srg.get_green_points(green_breps, 3)
    green_shades = srg.get_context_shades(green_breps, USAGE_DICT[usage], detached=True)

    # Balconies
    usage = 4
    balc_breps, _ = srg.get_geo(model, objs, USAGE_DICT[usage])
    balc_shades = srg.get_context_shades(balc_breps, USAGE_DICT[usage], detached=True)

    # Out All 
    usage = 11
    soc_out_breps, _ = srg.get_geo(model, objs, USAGE_DICT[usage])
    soc_out_shades = srg.get_context_shades(soc_out_breps, USAGE_DICT[usage], detached=True)

    # Out Resid
    usage = 12
    soc_R_out_breps, _ = srg.get_geo(model, objs, USAGE_DICT[usage])
    soc_R_out_shades = srg.get_context_shades(soc_R_out_breps, USAGE_DICT[usage], detached=True)

    # Out Offices
    usage = 13
    soc_O_out_breps, _ = srg.get_geo(model, objs, USAGE_DICT[usage])
    soc_O_out_shades = srg.get_context_shades(soc_O_out_breps, USAGE_DICT[usage], detached=True)

    # Core
    usage = 5
    core_geo, c_objs = srg.get_geo(model, objs, USAGE_DICT[usage])
    core_rooms, core_geo, c_objs = get_hbrooms(core_geo, USAGE_DICT[usage], c_objs)
    c_n_beds = srg.get_n_beds(model, c_objs, USAGE_DICT[usage])
    view_rooms_core, view_core_geo, vc_objs = get_hbrooms(core_geo, USAGE_DICT[usage], c_objs)
    core_balc_areas = srg.get_balcony_area(core_geo, balc_breps)

    # Windows
    all_windows = get_apertures(model, objs)

    # Offices
    usage = 0
    office_geo, o_objs = srg.get_geo(model, objs, USAGE_DICT[usage])
    office_rooms, office_geo, o_objs = get_hbrooms(office_geo, USAGE_DICT[usage], o_objs)
    of_n_beds = srg.get_n_beds(model, o_objs, USAGE_DICT[usage])
    view_rooms_office, view_office_geo, vo_objs = get_hbrooms(office_geo, USAGE_DICT[usage], o_objs)
    office_balc_areas = srg.get_balcony_area(office_geo, balc_breps)

    # Constructions and programs
    n = clean_and_id_string('c_set')
    c_set = spi.construction_set_op(window_identifier, wall_type, wall_r, 'GRC_Insul_Plasterboard', roof_r, 'GRC_Insul_Plasterboard', ground_r, n)
    office_program = spi.create_program(0, offices_occ_per_area)
    office_vent_c = spi.vent_control(0, operable)
    add_apertures_all(office_rooms, all_windows)
    spi.apply_prop(office_rooms, office_vent_c, c_set, office_program, operable, window_identifier)

    # Apartments
    usage = 2
    resid_geo, r_objs = srg.get_geo(model, objs, USAGE_DICT[usage])
    resid_rooms, resid_geo, r_objs = get_hbrooms(resid_geo, USAGE_DICT[usage], r_objs)
    r_n_beds = srg.get_n_beds(model, r_objs, USAGE_DICT[usage])
    view_rooms_resid, view_resid_geo, vr_objs = get_hbrooms(resid_geo, USAGE_DICT[usage], r_objs)
    resid_balc_areas = srg.get_balcony_area(resid_geo, balc_breps)
    occupants_per_area = 0.023 if resid_rooms else 0.0

    resid_program = spi.create_program(2, occupants_per_area)
    resid_vent_c = spi.vent_control(2, True)
    add_apertures_all(resid_rooms, all_windows)
    spi.apply_prop(resid_rooms, resid_vent_c, c_set, resid_program, True, window_identifier)

    # Social
    usages = [6, 7, 8, 9, 10]
    social_rooms = []
    view_rooms_social = []
    social_balc_areas = []
    s_n_beds = []

    for usage in usages:
        social_geo, s_objs = srg.get_geo(model, objs, USAGE_DICT[usage])
        social_room, social_geo, s_objs = get_hbrooms(social_geo, USAGE_DICT[usage], s_objs)
        social_rooms.append(social_room)
        s_n_bed = srg.get_n_beds(model, s_objs, USAGE_DICT[usage])
        s_n_beds.append(s_n_bed)
        view_r_social, view_social_geo, vs_objs = get_hbrooms(social_geo, USAGE_DICT[usage], s_objs)
        view_rooms_social.append(view_r_social)
        social_balc_area = srg.get_balcony_area(social_geo, balc_breps)
        social_balc_areas.append(social_balc_area)

    social_rooms = [item for sublist in social_rooms for item in sublist]
    view_rooms_social = [item for sublist in view_rooms_social for item in sublist]
    social_balc_areas = [item for sublist in social_balc_areas for item in sublist]
    s_n_beds = [item for sublist in s_n_beds for item in sublist]

    social_program = spi.create_program(2, occupants_per_area)
    social_vent_c = spi.vent_control(2, True)
    add_apertures_all(social_rooms, all_windows)
    spi.apply_prop(social_rooms, social_vent_c, c_set, social_program, True, window_identifier)

    model_rooms = office_rooms + resid_rooms + social_rooms + core_rooms
    Room.stories_by_floor_height(model_rooms)
    balcon_areas = office_balc_areas + resid_balc_areas + social_balc_areas + core_balc_areas
    ids, av_orients, room_areas, storeys, f_heights = srg.get_rooms_info(model_rooms)
    n_beds = of_n_beds + r_n_beds + s_n_beds + c_n_beds
    balcon_areas = srg.sort_per_story(balcon_areas, storeys)
    n_beds = srg.sort_per_story(n_beds, storeys)
    model_rooms = srg.sort_per_story(model_rooms, storeys)
    ids, av_orients, room_areas, storeys, f_heights = srg.get_rooms_info(model_rooms)

    usages_list = ids[:]  

    Room.intersect_adjacency(model_rooms, tolerance=0.01)
    Room.solve_adjacency(model_rooms, 0.01)

    hb_model = Model(
        'hb_main',
        rooms=model_rooms,
        orphaned_shades=context_shades + green_shades + balc_shades + soc_out_shades + soc_R_out_shades + soc_O_out_shades
    )

    # View model
    v_winds, n = srg.get_geo(model, objs, 'WINDOWS')
    view_windows = srg.get_context_shades(v_winds, 'WINDOWS', detached = True)
    view_rooms = view_rooms_office + view_rooms_resid + view_rooms_social + view_rooms_core
    Room.stories_by_floor_height(view_rooms)
    id, av_orients, r_areas, storeys, f_heights = srg.get_rooms_info(view_rooms)
    view_rooms = srg.sort_per_story(view_rooms, storeys)
    id, av_orients, r_areas, storeys, f_heights = srg.get_rooms_info(view_rooms)

    view_hb_model = Model(
        'hb_view',
        rooms=view_rooms,
        orphaned_shades=context_shades + green_shades + balc_shades + view_windows + soc_out_shades + soc_R_out_shades + soc_O_out_shades
    )

    hb_main_path = str((out_dir / "hb_main.hbjson").resolve())
    hb_view_path = str((out_dir / "hb_view.hbjson").resolve())
    hb_model.to_hbjson(name='hb_main', folder=out_dir)
    view_hb_model.to_hbjson(name='hb_view', folder=out_dir)

    return {
        "ok": True,
        "hb_model_path": hb_main_path,
        "view_hb_model_path": hb_view_path,
        "ids": ids,
        "storeys": storeys,
        "av_orients": av_orients,
        "room_areas": room_areas,
        "balcon_areas": balcon_areas,
        "n_beds": n_beds,
        "usages_list": usages_list
    }



def get_rooms_db(payload):
    
    src_3dm = payload["src_3dm"]
    wall_red = float(payload["wall_red"])
    wind_red = float(payload["wind_red"])
    floor_a = float(payload["floor_a"])
    ceiling_a = float(payload["ceiling_a"])

    
    model = Rhino.FileIO.File3dm.Read(src_3dm)
    if model is None:
        return {"ok": False, "error": "Could not read 3DM"}

    objs = model.Objects

    
    busy_pts, local_pts = srg.get_street(model, objs)

    
    context_breps, _ = srg.get_geo(model, objs, USAGE_DICT[1])  
    office_geo, _ = srg.get_geo(model, objs, USAGE_DICT[0])  
    resid_geo, _  = srg.get_geo(model, objs, USAGE_DICT[2])    
    core_geo,  _  = srg.get_geo(model, objs, USAGE_DICT[5])    


    social_geo = []
    for u in [6, 7, 8, 9, 10]:
        g, _ = srg.get_geo(model, objs, USAGE_DICT[u])
        social_geo.extend(g)

    
    all_breps = context_breps + office_geo + resid_geo + social_geo + core_geo
    all_mesh = join_geometry_to_mesh(all_breps)

    
    def _rooms_for(usage_label, geos, objs_list):
        hb_rooms, _, _ = get_hbrooms(geos, usage_label, objs_list)
        return hb_rooms

    
    _, o_objs = srg.get_geo(model, objs, USAGE_DICT[0])
    _, r_objs = srg.get_geo(model, objs, USAGE_DICT[2])
    _, c_objs = srg.get_geo(model, objs, USAGE_DICT[5])
    s_objs_all = []
    for u in [6, 7, 8, 9, 10]:
        _, s_objs = srg.get_geo(model, objs, USAGE_DICT[u])
        s_objs_all.extend(s_objs)

    office_rooms = _rooms_for(USAGE_DICT[0], office_geo, o_objs)
    resid_rooms  = _rooms_for(USAGE_DICT[2], resid_geo,  r_objs)
    core_rooms   = _rooms_for(USAGE_DICT[5], core_geo,   c_objs)

    social_rooms = []
    for u in [6, 7, 8, 9, 10]:
        g, s_objs = srg.get_geo(model, objs, USAGE_DICT[u])
        s_rooms, _, _ = get_hbrooms(g, USAGE_DICT[u], s_objs)
        social_rooms.extend(s_rooms)

    
    all_windows = get_apertures(model, objs)
    add_apertures_all(office_rooms, all_windows)
    add_apertures_all(resid_rooms,  all_windows)
    add_apertures_all(social_rooms, all_windows)
    add_apertures_all(core_rooms,   all_windows)

    
    model_rooms = office_rooms + resid_rooms + social_rooms + core_rooms
    Room.stories_by_floor_height(model_rooms)
    
    ids_tmp, av_orients_tmp, room_areas_tmp, storeys_tmp, f_heights_tmp = srg.get_rooms_info(model_rooms)
    model_rooms = srg.sort_per_story(model_rooms, storeys_tmp)

    Room.intersect_adjacency(model_rooms, tolerance=0.01)
    Room.solve_adjacency(model_rooms, 0.01)

    
    rooms_sound_levels = []
    A_walls_window = 0.05  

    for room in model_rooms:
        room_L1s = []
        window_areas = []
        wall_areas = []
        absorption_total = 0.0

        floor_area = room.floor_area
        ceiling_area = room.floor_area
        absorption_total += floor_area * floor_a
        absorption_total += ceiling_area * ceiling_a

        for wall in room.walls:
            if wall.boundary_condition.name == "Surface":
                absorption_total += wall.area * A_walls_window
                continue

            if wall.boundary_condition.name == "Outdoors":
                w_centre = wall.center
                w_norm = wall.normal
                face_area = wall.area
                wind_area = wall.aperture_area
                solid_area = face_area - wind_area

                absorption_total += (wind_area + solid_area) * A_walls_window
                window_areas.append(wind_area)
                wall_areas.append(solid_area)

                moved_pt = w_centre.move(w_norm)
                rh_w_centre = [from_point3d(moved_pt)]

                L1_wall_sources = []

                # Busy street
                for busy_pt in busy_pts:
                    int_matrix = intersect_mesh_lines(all_mesh, rh_w_centre, [busy_pt], parallel=False)
                    is_clear = int_matrix[0][0]
                    if is_clear == 1:
                        f_line = Line(rh_w_centre[0], busy_pt)
                        f_curve = f_line.ToNurbsCurve()
                        dist = f_curve.GetLength()
                        L1 = 85 - 20 * math.log10(dist)
                        L1_wall_sources.append(L1)

                # Local street
                for local_pt in local_pts:
                    int_matrix = intersect_mesh_lines(all_mesh, rh_w_centre, [local_pt], parallel=False)
                    is_clear = int_matrix[0][0]
                    if is_clear == 1:
                        f_line = Line(rh_w_centre[0], local_pt)
                        f_curve = f_line.ToNurbsCurve()
                        dist = f_curve.GetLength()
                        L1 = 65 - 20 * math.log10(dist)
                        L1_wall_sources.append(L1)

                if L1_wall_sources:
                    E_sum = sum(10 ** (L / 10) for L in L1_wall_sources)
                    L1_combined = 10 * math.log10(E_sum)
                    room_L1s.append(L1_combined)
                else:
                    room_L1s.append(None)

        energies = []
        for i, L1 in enumerate(room_L1s):
            if L1 is None:
                continue
            w_area = window_areas[i]
            s_area = wall_areas[i]
            E_window = (w_area / absorption_total) * 10 ** ((L1 - wind_red) / 10) if w_area > 0 else 0.0
            E_wall   = (s_area / absorption_total) * 10 ** ((L1 - wall_red) / 10) if s_area > 0 else 0.0
            energies.append(E_window + E_wall)

        if energies:
            E_total = sum(energies)
            L2 = 10 * math.log10(E_total)
        else:
            L2 = None

        rooms_sound_levels.append(L2)

    rooms_sound_levels = [val if val is not None and val >= 0 else 0.0 for val in rooms_sound_levels]

    return {
        "ok": True,
        "rooms_db": rooms_sound_levels
    }




def _load_context_and_meshes(src_3dm, hb_model_path):
    model = Rhino.FileIO.File3dm.Read(src_3dm)
    objs = model.Objects
   
    context_breps, _ = srg.get_geo(model, objs, USAGE_DICT[1])
    office_geo, _ = srg.get_geo(model, objs, USAGE_DICT[0])
    resid_geo, _  = srg.get_geo(model, objs, USAGE_DICT[2])
    core_geo,  _  = srg.get_geo(model, objs, USAGE_DICT[5])
    social_geo = []
    for u in [6,7,8,9,10]:
        g, _ = srg.get_geo(model, objs, USAGE_DICT[u])
        social_geo.extend(g)
    all_breps = context_breps + office_geo + resid_geo + social_geo + core_geo

    shade_mesh = join_geometry_to_mesh(all_breps)

    hb_model = Model.from_hbjson(hb_model_path)
    building_meshes = srg.get_model_meshes(hb_model)  
    return model, objs, shade_mesh, building_meshes

def _make_v_meshes(view_hb_path, dist):
    view_hb_model = Model.from_hbjson(view_hb_path)
    v_meshes, sensor_grid = spi.create_sensors(view_hb_model, dist)
    return v_meshes




def run_horizontal_views(payload):
    src_3dm = payload["src_3dm"]
    hb_model_path = payload["hb_model_path"]
    view_hb_path = payload["view_hb_path"]
    hv_res_folder = Path(payload["hv_res_folder"]); hv_res_folder.mkdir(parents=True, exist_ok=True)
    hv_mean_folder = Path(payload["hv_mean_folder"]); hv_mean_folder.mkdir(parents=True, exist_ok=True)
    hv_satisf_folder = Path(payload["hv_satisf_folder"]); hv_satisf_folder.mkdir(parents=True, exist_ok=True)
    dist = float(payload.get("sensor_dist", 1.0))

    hb_model = Model.from_hbjson(hb_model_path)
    building_meshes = srg.get_model_meshes(hb_model)
    model = Rhino.FileIO.File3dm.Read(src_3dm)
    objs = model.Objects
    context_breps, _ = srg.get_geo(model, objs, USAGE_DICT[1])

    shade_mesh = join_geometry_to_mesh(context_breps + building_meshes)
    v_meshes = _make_v_meshes(view_hb_path, dist)


    room_matrix = []
    colored_meshes = []

    for study_mesh in v_meshes:

        lb_vecs = view_sphere.horizontal_radial_vectors(30 * 1)


        view_vecs = [from_vector3d(pt) for pt in lb_vecs]

        points = [from_point3d(pt.move(vec * 0)) for pt, vec in
                zip(study_mesh.face_centroids, study_mesh.face_normals)]
        
        int_matrix, angles = intersect_mesh_rays(shade_mesh, points, view_vecs, cpu_count=None, parallel=False)
        vec_count = len(view_vecs)
        results = [sum(int_list) * 100 / vec_count for int_list in int_matrix]
        room_matrix.append(results)

        legend_par_ = None
        graphic = GraphicContainer(results, study_mesh.min, study_mesh.max, legend_par_)
        graphic.legend_parameters.title = '%'
        if legend_par_ is None or legend_par_.are_colors_default:
            graphic.legend_parameters.colors = Colorset.view_study()

        study_mesh.colors = graphic.value_colors
        colored_meshes.append(study_mesh)

    # Save results
    slu.save_res_files(room_matrix, v_meshes, hv_res_folder)
    hv_mean = [[np.mean(sub)] * len(sub) for sub in room_matrix]
    slu.save_res_files(hv_mean, v_meshes, hv_mean_folder)
    hv_mean_room = [np.mean(sub) for sub in room_matrix]
    hv_satisf = sst.views_horiz_satisf(room_matrix)
    slu.save_res_files(hv_satisf, v_meshes, hv_satisf_folder)
    hv_satisf_room = [sub[0] for sub in hv_satisf]

    return {"hv_mean_room": hv_mean_room, "hv_satisf_room": hv_satisf_room}

def run_green_views(payload):
    src_3dm = payload["src_3dm"]
    hb_model_path = payload["hb_model_path"]
    view_hb_path = payload["view_hb_path"]
    gv_res_folder = Path(payload["gv_res_folder"]); gv_res_folder.mkdir(parents=True, exist_ok=True)
    gv_mean_folder = Path(payload["gv_mean_folder"]); gv_mean_folder.mkdir(parents=True, exist_ok=True)
    gv_satisf_folder = Path(payload["gv_satisf_folder"]); gv_satisf_folder.mkdir(parents=True, exist_ok=True)
    dist = float(payload.get("sensor_dist", 1.0))
    green_sampling = int(payload.get("green_sampling", 5))

    hb_model = Model.from_hbjson(hb_model_path)
    building_meshes = srg.get_model_meshes(hb_model)
    model = Rhino.FileIO.File3dm.Read(src_3dm)
    objs = model.Objects
    context_breps, _ = srg.get_geo(model, objs, USAGE_DICT[1])
    
    green_breps, _ = srg.get_geo(model, objs, USAGE_DICT[3])
    green_points = srg.get_green_points(green_breps, green_sampling)
    green_pts = [from_point3d(pt) for pt in green_points]

    v_meshes = _make_v_meshes(view_hb_path, dist)

    shade_mesh = join_geometry_to_mesh(context_breps + building_meshes)
    room_matrix = []
    colored_meshes = []
    green_pts = []

    for pt in green_points:
        gpt = from_point3d(pt)
        green_pts.append(gpt)

    for study_mesh in v_meshes:

        points = [from_point3d(pt.move(vec * 0)) for pt, vec in
                zip(study_mesh.face_centroids, study_mesh.face_normals)]

        int_matrix = intersect_mesh_lines(
            shade_mesh, points, green_pts, max_dist = None, cpu_count=None, parallel=False)
        vec_count = len(green_points)
        results = [sum(int_list) * 100 / vec_count for int_list in int_matrix]
        room_matrix.append(results)

        legend_par_ = None
        graphic = GraphicContainer(results, study_mesh.min, study_mesh.max, legend_par_)
        graphic.legend_parameters.title = '%'
        if legend_par_ is None or legend_par_.are_colors_default:
            graphic.legend_parameters.colors = Colorset.view_study()

        study_mesh.colors = graphic.value_colors
        colored_meshes.append(study_mesh)

    slu.save_res_files(room_matrix, v_meshes, gv_res_folder)
    gv_mean = [[np.mean(sub)] * len(sub) for sub in room_matrix]
    slu.save_res_files(gv_mean, v_meshes, gv_mean_folder)
    gv_mean_room = [np.mean(sub) for sub in room_matrix]
    gv_satisf = sst.views_green_satisf(room_matrix)
    slu.save_res_files(gv_satisf, v_meshes, gv_satisf_folder)
    gv_satisf_room = [sub[0] for sub in gv_satisf]

    return {"gv_mean_room": gv_mean_room, "gv_satisf_room": gv_satisf_room}


def run_sky_views(payload):
    src_3dm = payload["src_3dm"]
    hb_model_path = payload["hb_model_path"]
    view_hb_path = payload["view_hb_path"]
    sv_res_folder = Path(payload["sv_res_folder"]); sv_res_folder.mkdir(parents=True, exist_ok=True)
    sv_mean_folder = Path(payload["sv_mean_folder"]); sv_mean_folder.mkdir(parents=True, exist_ok=True)
    sv_satisf_folder = Path(payload["sv_satisf_folder"]); sv_satisf_folder.mkdir(parents=True, exist_ok=True)
    dist = float(payload.get("sensor_dist", 1.0))

    hb_model = Model.from_hbjson(hb_model_path)
    building_meshes = srg.get_model_meshes(hb_model)
    model = Rhino.FileIO.File3dm.Read(src_3dm)
    objs = model.Objects
    context_breps, _ = srg.get_geo(model, objs, USAGE_DICT[1])

    shade_mesh = join_geometry_to_mesh(context_breps + building_meshes)
    v_meshes = _make_v_meshes(view_hb_path, dist)


    room_matrix = []
    colored_meshes = []

    for study_mesh in v_meshes:

        patch_mesh, lb_vecs = view_sphere.dome_patches()


        view_vecs = [from_vector3d(pt) for pt in lb_vecs]

        points = [from_point3d(pt.move(vec * 0)) for pt, vec in
                zip(study_mesh.face_centroids, study_mesh.face_normals)]
        
        int_matrix, angles = intersect_mesh_rays(shade_mesh, points, view_vecs, cpu_count=None, parallel=False)
        vec_count = len(view_vecs)
        results = [sum(int_list) * 100 / vec_count for int_list in int_matrix]
        room_matrix.append(results)

        legend_par_ = None
        graphic = GraphicContainer(results, study_mesh.min, study_mesh.max, legend_par_)
        graphic.legend_parameters.title = '%'
        if legend_par_ is None or legend_par_.are_colors_default:
            graphic.legend_parameters.colors = Colorset.view_study()

        study_mesh.colors = graphic.value_colors
        colored_meshes.append(study_mesh)

    slu.save_res_files(room_matrix, v_meshes, sv_res_folder)
    sv_mean = [[np.mean(sub)] * len(sub) for sub in room_matrix]
    slu.save_res_files(sv_mean, v_meshes, sv_mean_folder)
    sv_mean_room = [np.mean(sub) for sub in room_matrix]
    sv_satisf = sst.views_sky_satisf(room_matrix)
    slu.save_res_files(sv_satisf, v_meshes, sv_satisf_folder)
    sv_satisf_room = [sub[0] for sub in sv_satisf]

    return {"sv_mean_room": sv_mean_room, "sv_satisf_room": sv_satisf_room}


def run_balcony_size_metrics(payload):
    view_hb_path = payload["view_hb_path"]
    as_area_folder = Path(payload["as_area_folder"]); as_area_folder.mkdir(parents=True, exist_ok=True)
    as_res_folder  = Path(payload["as_res_folder"]);  as_res_folder.mkdir(parents=True, exist_ok=True)
    as_satisf_folder = Path(payload["as_satisf_folder"]); as_satisf_folder.mkdir(parents=True, exist_ok=True)

    b_area_folder  = Path(payload["b_area_folder"]);  b_area_folder.mkdir(parents=True, exist_ok=True)
    b_res_folder   = Path(payload["b_res_folder"]);   b_res_folder.mkdir(parents=True, exist_ok=True)
    b_satisf_folder= Path(payload["b_satisf_folder"]);b_satisf_folder.mkdir(parents=True, exist_ok=True)

    room_areas = list(payload["room_areas"])
    balcon_areas = list(payload["balcon_areas"])
    ids = list(payload["ids"])
    n_beds = list(payload["n_beds"])
    occ_per_bedroom = float(payload["occupants_per_bedroom"])
    occ_per_area = float(payload["occupants_per_area"])
    dist = float(payload.get("sensor_dist", 1.0))

    v_meshes = _make_v_meshes(view_hb_path, dist)

    lists_balc_areas = [[area] * len(mesh.face_centroids) for area, mesh in zip(balcon_areas, v_meshes)]
    slu.save_res_files(lists_balc_areas, v_meshes, b_area_folder)

    balc_perc = [(b / r) * 100 if r > 0 else 0 for r, b in zip(room_areas, balcon_areas)]
    lists_balc_perc = [[p] * len(mesh.face_centroids) for p, mesh in zip(balc_perc, v_meshes)]
    slu.save_res_files(lists_balc_perc, v_meshes, b_res_folder)
    _balc_satisf = sst.balconies_satisf(lists_balc_perc)
    slu.save_res_files(_balc_satisf, v_meshes, b_satisf_folder)
    balc_satisf_room = [sub[0] for sub in _balc_satisf]


    lists_room_areas = [[a] * len(mesh.face_centroids) for a, mesh in zip(room_areas, v_meshes)]
    slu.save_res_files(lists_room_areas, v_meshes, as_area_folder)

    occupancy_rate = []
    for rid, beds, area in zip(ids, n_beds, room_areas):
        if rid.startswith("RESID"):
            occupants = beds * occ_per_bedroom
            rate = occupants / area if occupants > 0 and area > 0 else 0
        elif rid.startswith("COMMERC"):
            rate = occ_per_area
        else:
            rate = 0
        occupancy_rate.append(rate)

    lists_ap_occup = [[r] * len(mesh.face_centroids) for r, mesh in zip(occupancy_rate, v_meshes)]
    slu.save_res_files(lists_ap_occup, v_meshes, as_res_folder)

    size_sat = sst.size_satisf(occupancy_rate, ids)
    lists_size_satisf = [[s] * len(mesh.face_centroids) for s, mesh in zip(size_sat, v_meshes)]
    slu.save_res_files(lists_size_satisf, v_meshes, as_satisf_folder)

    return {
        "balcony_percentage": balc_perc,
        "balcony_satisf": balc_satisf_room,
        "occupancy": occupancy_rate,
        "space_size_satisf": size_sat
    }


def get_social_outdoor_areas(payload):
    """Return total shade areas for outdoor social groups."""
    src_3dm = payload["src_3dm"]
    model = Rhino.FileIO.File3dm.Read(src_3dm)
    if model is None:
        return {"ok": False, "error": "Could not read 3DM"}

    objs = model.Objects

    # SOCIAL_OUTDOOR_ALL
    soc_out_breps, _ = srg.get_geo(model, objs, USAGE_DICT[11])
    soc_out_shades = srg.get_context_shades(soc_out_breps, USAGE_DICT[11], detached=True)
    soc_out_area = srg.get_shade_areas(soc_out_shades)

    # SOCIAL_OUTDOOR_RESID
    soc_R_out_breps, _ = srg.get_geo(model, objs, USAGE_DICT[12])
    soc_R_out_shades = srg.get_context_shades(soc_R_out_breps, USAGE_DICT[12], detached=True)
    soc_R_out_area = srg.get_shade_areas(soc_R_out_shades)

    # SOCIAL_OUTDOOR_COMMERC
    soc_O_out_breps, _ = srg.get_geo(model, objs, USAGE_DICT[13])
    soc_O_out_shades = srg.get_context_shades(soc_O_out_breps, USAGE_DICT[13], detached=True)
    soc_O_out_area = srg.get_shade_areas(soc_O_out_shades)

    return {
        "ok": True,
        "soc_out_area": float(soc_out_area),
        "soc_R_out_area": float(soc_R_out_area),
        "soc_O_out_area": float(soc_O_out_area)
    }





def main():
    req = json.loads(sys.stdin.read())
    op = req.get("op")
    payload = req.get("payload", {})

    if op == "build_hb_models":
        res = build_hb_models(payload)
        print(json.dumps(res)); return

    elif op == "prep_view_assets":
        res = prep_view_assets(payload)
        print(json.dumps(res)); return
    
    elif op == "get_rooms_db":
        res = get_rooms_db(payload)
        print(json.dumps(res)); return
    
    elif op == "run_horizontal_views":
            res = run_horizontal_views(payload)
            print(json.dumps(res)); return
    
    elif op == "run_green_views":
        res = run_green_views(payload)
        print(json.dumps(res)); return
    
    elif op == "run_sky_views":
        res = run_sky_views(payload)
        print(json.dumps(res)); return
    
    elif op == "run_balcony_size_metrics":
        res = run_balcony_size_metrics(payload)
        print(json.dumps(res)); return
    
    elif op == "save_room_scalar_results":
        res = save_room_scalar_results(payload)
        print(json.dumps(res)); return
        
    elif op == "get_social_outdoor_areas":
        res = get_social_outdoor_areas(payload)
        print(json.dumps(res)); return



    print(json.dumps({"ok": False, "error": "unknown operation"}))
    sys.exit(2)


if __name__ == "__main__":
    main()
