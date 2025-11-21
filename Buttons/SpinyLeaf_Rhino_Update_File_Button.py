import os
import shutil
import subprocess
import Rhino
import scriptcontext as sc

try:
    from pathlib import Path
except ImportError:
    import os
    class Path:
        @staticmethod
        def home():
            return os.path.expanduser("~")

def update_file():

    home_path = os.path.expanduser("~")
    main_f = os.path.join(home_path, "SpinyLeaf_App")
    model_f = os.path.join(main_f, "Rhino_Model")
    dest_path = os.path.join(model_f, "shared_model.3dm")

    if not os.path.exists(main_f):
        os.mkdir(main_f)
    if not os.path.exists(model_f):
        os.mkdir(model_f)

    
    Rhino.RhinoDoc.ActiveDoc.WriteFile(dest_path, Rhino.FileIO.FileWriteOptions())

    home_dir = os.path.expanduser("~")

    main_f = os.path.join(home_path, "SpinyLeaf_App")
    out_f = os.path.join(main_f, "Wellbeing_Fostered_by_Design")

    if not os.path.exists(main_f):
        os.makedirs(main_f)

    if not os.path.exists(out_f):
        os.makedirs(out_f)


def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def clear_and_create_out_folders(base_out_f):
    
    if os.path.exists(base_out_f):
        shutil.rmtree(base_out_f)

    
    os.makedirs(base_out_f)

    # Delight
    out_f_delight = os.path.join(base_out_f, "Delight_Dimension")
    make_dir(out_f_delight)

    for sub in [
        "Hor_Views", "Hor_Views_mean", "Hor_Views_satisf",
        "Green_Views", "Gree_Views_mean", "Green_Views_satisf",
        "Sky_Views", "Sky_Views_mean", "Sky_Views_satisf",
        "Balcony_Area", "Balcony_percentage", "Balcony_satisf",
        "Apartment_Area", "Apartment_Occupancy", "Apartment_Size_satisf",
        "_Delight_Satisfaction"
    ]:
        make_dir(os.path.join(out_f_delight, sub))

    # Comfort
    out_f_comfort = os.path.join(base_out_f, "Comfort_Dimension")
    make_dir(out_f_comfort)

    for sub in [
        "Extreme_Hot_Sim", "Extreme_Cold_Sim",
        "Extreme_Hot_Week_Temp", "Ext_Hot_Thermal_Sensation", "Ext_Hot_Thermal_Satisfaction",
        "Extreme_Cold_Week_Temp", "Ext_Cold_Thermal_Sensation", "Ext_Cold_Thermal_Satisfaction",
        "CO2_Levels", "Relative_Humidity", "Air_Quality_Satisfaction",
        "Sound_Levels", "Sound_Satisfaction",
        "Daylight_Simulation", "DA_Annual", "DA_Mean", "DA_Satisfaction",
        "_Comfort_Satisfaction"
    ]:
        make_dir(os.path.join(out_f_comfort, sub))

    # Social
    out_f_social = os.path.join(base_out_f, "Social_Dimension")
    make_dir(out_f_social)

    for sub in [
        "Social_Green_area", "Social_Green_Area_Occup", "Social_Green_Satisf",
        "Social_Total_Area", "Social_Area_Occup", "Social_Amount_Satisf",
        "Social_Levels_Available", "Social_WDSS", "Social_Distribution_Satisf",
        "_Social_Satisfaction"
    ]:
        make_dir(os.path.join(out_f_social, sub))

    # Wellbeing design summary
    make_dir(os.path.join(base_out_f, "Wellbeing_Design"))


home_dir = os.path.expanduser("~")
main_f = os.path.join(home_dir, "SpinyLeaf_App")
out_f = os.path.join(main_f, "Wellbeing_Fostered_by_Design")
clear_and_create_out_folders(out_f)

update_file()