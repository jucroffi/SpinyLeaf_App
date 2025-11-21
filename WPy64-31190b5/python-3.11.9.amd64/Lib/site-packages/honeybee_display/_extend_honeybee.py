# coding=utf-8
# import the core honeybee modules
from honeybee.model import Model
from honeybee.room import Room
from honeybee.face import Face
from honeybee.aperture import Aperture
from honeybee.door import Door
from honeybee.shade import Shade
from honeybee.shademesh import ShadeMesh
from honeybee.colorobj import ColorRoom, ColorFace

# import the extension functions
from .model import model_to_vis_set, model_to_vis_set_wireframe, \
    model_comparison_to_vis_set
from .room import room_to_vis_set, room_to_vis_set_wireframe
from .face import face_to_vis_set, face_to_vis_set_wireframe
from .aperture import aperture_to_vis_set, aperture_to_vis_set_wireframe
from .door import door_to_vis_set, door_to_vis_set_wireframe
from .shade import shade_to_vis_set, shade_to_vis_set_wireframe
from .shademesh import shade_mesh_to_vis_set, shade_mesh_to_vis_set_wireframe
from .colorobj import color_room_to_vis_set, color_face_to_vis_set

# inject the methods onto the classes
Model.to_vis_set = model_to_vis_set
Model.to_vis_set_wireframe = model_to_vis_set_wireframe
Model.to_vis_set_comparison = model_comparison_to_vis_set
Room.to_vis_set = room_to_vis_set
Room.to_vis_set_wireframe = room_to_vis_set_wireframe
Face.to_vis_set = face_to_vis_set
Face.to_vis_set_wireframe = face_to_vis_set_wireframe
Aperture.to_vis_set = aperture_to_vis_set
Aperture.to_vis_set_wireframe = aperture_to_vis_set_wireframe
Door.to_vis_set = door_to_vis_set
Door.to_vis_set_wireframe = door_to_vis_set_wireframe
Shade.to_vis_set = shade_to_vis_set
Shade.to_vis_set_wireframe = shade_to_vis_set_wireframe
ShadeMesh.to_vis_set = shade_mesh_to_vis_set
ShadeMesh.to_vis_set_wireframe = shade_mesh_to_vis_set_wireframe
ColorRoom.to_vis_set = color_room_to_vis_set
ColorFace.to_vis_set = color_face_to_vis_set

# attempt to extend honeybee-energy if it is installed
try:
    from honeybee_energy.result.colorobj import ColorRoom as EnergyColorRoom
    from honeybee_energy.result.colorobj import ColorFace as EnergyColorFace
    from .energy.colorobj import energy_color_room_to_vis_set, color_face_to_vis_set
    EnergyColorRoom.to_vis_set = energy_color_room_to_vis_set
    EnergyColorFace.to_vis_set = color_face_to_vis_set
except ImportError:  # honeybee-energy is not installed
    pass
