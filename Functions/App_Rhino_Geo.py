
import numpy as np


import compute_rhino3d.Util
import clr

clr.AddReference(r"C:\Program Files\Rhino 8\System\RhinoCommon.dll")

compute_rhino3d.Util.url = "http://localhost:6500/"

import rhinoinside
rhinoinside.load()

import Rhino.Geometry as rg

from Rhino.Geometry.Intersect import Intersection 
from Rhino.Geometry import Curve
from Rhino.Geometry import Line, LineCurve
from ladybug.color import Colorset

from ladybug_rhino.togeometry import to_polyface3d, to_face3d
from ladybug_rhino.fromgeometry import from_face3ds_to_colored_mesh

from honeybee.shade import Shade
from honeybee.typing import clean_and_id_string

from honeybee.shade import Shade





def get_context_shades(context_breps, name, detached):
    shades = []

    for br in context_breps:
        lb_faces = to_face3d(br)

        for i, fc in enumerate(lb_faces):
            shade_name = clean_and_id_string(name)
            hb_shd = Shade(shade_name, fc, is_detached=detached) 
            shades.append(hb_shd)
  
    return shades



def get_geo(model, objs, usage):
    usage_geo = []
    selected_objs = []  

    for obj in objs:
        layer_index = obj.Attributes.LayerIndex
        layer = model.Layers[layer_index].Name

        if layer.startswith(usage):
            selected_objs.append(obj)
            geo = obj.Geometry
            if not isinstance(geo, rg.Brep):
                geo = geo.ToBrep()
            
            usage_geo.append(geo)

    return usage_geo, selected_objs


def get_n_beds(model, objs, usage):

    n_beds = []
    for obj in objs:
        layer_index = obj.Attributes.LayerIndex
        layer = model.Layers[layer_index].Name

        if layer.startswith(usage):

            if layer.startswith('COMMERC'):
                n_bed = 0
                n_beds.append(n_bed)

            elif layer.startswith('RESID'):
                n_bed = int(layer.split('_')[-1])
                n_beds.append(n_bed)

            elif layer.startswith('SOCIAL'):
                n_bed = 0
                n_beds.append(n_bed)

            elif layer.startswith('CORE'):
                n_bed = 0
                n_beds.append(n_bed)

    return n_beds




def get_street(model, objs):
    street_lines_busy = []
    street_lines_local = []
    street_pts_busy = []
    street_pts_local = []


    for obj in objs:
        layer_index = obj.Attributes.LayerIndex
        layer = model.Layers[layer_index].Name

        if layer.endswith('BUSY'):
            geo = obj.Geometry.Reparameterize()
            street_lines_busy.append(geo)
        elif layer.endswith('LOCAL'):
            geo = obj.Geometry.Reparameterize()
            street_lines_local.append(geo)

    for street in street_lines_busy:
        for i in np.arange(0, 1.1, 0.01):
            pt = street.PointAtNormalizedLength(i)
            street_pts_busy.append(pt)

    for street in street_lines_local:
        for i in np.arange(0, 1.1, 0.01):
            pt = street.PointAtNormalizedLength(i)
            street_pts_local.append(pt)

    return street_pts_busy, street_pts_local



def get_nbeds(model, objs):
    n_beds = []
    for obj in objs:
        layer_index = obj.Attributes.LayerIndex
        layer = model.Layers[layer_index].Name

        if layer.startswith('COMMERC'):
            n_bed = 0
            n_beds.append(n_bed)

        elif layer.startswith('RESID'):
            n_bed = int(layer.split('_')[-1])
            n_beds.append(n_bed)

        elif layer.startswith('SOCIAL'):
            n_bed = 0
            n_beds.append(n_bed)

    return n_beds


def get_usage_list(model, objs):
    usage_list = []
    for obj in objs:
        layer_index = obj.Attributes.LayerIndex
        layer_name = model.Layers[layer_index].Name
        layer_name = layer_name.split('_')[0]

        if layer_name == 'COMMERC' or layer_name == 'RESID' or layer_name == 'SOCIAL' or layer_name == 'CORE':
            usage_list.append(layer_name)

    return usage_list



def get_green_points(green_breps, dist):
    centroids = []

    for br in green_breps:
        lb_face = to_face3d(br)[0]
        lb_mesh_centroid = (lb_face.mesh_grid(x_dim = dist)).face_centroids
        centroids.append(lb_mesh_centroid)

    flattened = [item for sublist in centroids for item in sublist]

    return flattened


def get_model_meshes(model):
    palette = Colorset.openstudio_palette()
    brep_walls =[]
    brep_roofs =[]
    brep_floors = []
    context_mesh = []

    rooms = model.rooms
    for room in rooms:
        r_walls = room.walls
        r_roofs = room.roof_ceilings
        r_floors = room.floors
        
        for w in r_walls:
            brep_walls.append(w.punched_geometry)
        for r in r_roofs:
            brep_roofs.append(r.punched_geometry)
        for f in r_floors:
            brep_floors.append(f.punched_geometry)

        context_mesh.append(from_face3ds_to_colored_mesh(brep_walls, palette[1]))
        context_mesh.append(from_face3ds_to_colored_mesh(brep_roofs, palette[1]))
        context_mesh.append(from_face3ds_to_colored_mesh(brep_floors, palette[1]))
    
    return context_mesh


def get_balcony_area(rooms_geo, balconies_geo):
    b_areas = []

    for r_geo in rooms_geo:
        balcony_areas = []  

        for b_geo in balconies_geo:
            rc, out_curves, out_points = Intersection.BrepBrep(r_geo, b_geo, 0.001)

            if out_curves and len(out_curves) > 0:
                b_area = b_geo.GetArea()
                balcony_areas.append(b_area)

        if balcony_areas:
            total_area = sum(balcony_areas)
        else:
            total_area = 0

        b_areas.append(total_area)

    return b_areas
        
def get_room_areas(model_rooms):
    r_areas = []
    for room in model_rooms:
        area = room.floor_area
        r_areas.append(area)
    return r_areas



def get_centroids(v_meshes):
    v_mesh_centr = []
    for mesh in v_meshes:
        c_pts = list(mesh.face_centroids)
        v_mesh_centr.append(c_pts)

    return v_mesh_centr


def get_rooms_info(model_rooms):
    ids = []
    av_orients = []
    r_areas = []
    stories = []
    f_heights = []

    for room in model_rooms:
        avo = room.average_orientation()
        
        av_orients.append(avo)
        id = room.identifier
        ids.append(id)
        area = room.floor_area
        r_areas.append(area)
        floor_s = room.story
        stories.append(floor_s)
        floor_h = room.average_floor_height
        f_heights.append(floor_h)

    return ids, av_orients, r_areas, stories, f_heights


def sort_per_story(hb_rooms, storeys):

    sorted_rooms = [room for room, floor in sorted(zip(hb_rooms, storeys),
    key=lambda pair: int(pair[1].lstrip("Floor")))]

    return sorted_rooms


def get_shade_areas(sades_list):
    s_areas = []
    for shade in sades_list:
        s_area = shade.area
        s_areas.append(s_area)

    total_area = sum(s_areas)

    return total_area