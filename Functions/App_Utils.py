import streamlit as st
import os

import shutil
from matplotlib.path import Path


import numpy as np
import json
import re
from pathlib import Path




def read_res_file(file_path):
    with open(file_path, 'r') as f:
        return [float(line.strip()) for line in f if line.strip()]
    



def save_res_files(results, meshes, folder_path):

    for i, res in enumerate(results):
        res_path = folder_path / f"grid_{i}.res"

        with open(res_path, 'w') as file:
            for j, item in enumerate(res):
                line = f"{item}"
                if j < len(res) - 1:
                    line += '\n' 
                file.write(line)

    info = []
    for i, mesh in enumerate(meshes):
        count = len(mesh.face_centroids)

        info.append({
            "count": count,
            "name": f"grid_{i}",
            "identifier": f"grid_{i}",
            "group": "",
            "full_id": f"grid_{i}"
        })

    grids_info_path = folder_path / "grids_info.json"
    with open(grids_info_path, "w") as f:
        json.dump(info, f, indent=2)




def remove_last_blank_line_in_res(folder):
    folder = Path(folder)

    for file in folder.glob("*.res"):
        lines = file.read_text().splitlines()

        if lines and lines[-1].strip() == "":
            lines = lines[:-1]
        file.write_text("\n".join(lines) + "\n") 



def clean_data(folder_path):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            pass


def clean_grid_info_json(json_path):
    json_path = Path(json_path)

    with open(json_path, "r") as f:
        data = json.load(f)

    for entry in data:
        for key in ["name", "identifier", "full_id"]:
            if key in entry and isinstance(entry[key], str):
                entry[key] = re.sub(r"(_\w{8})$", "", entry[key])

    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)



def get_res_values(folder_path):
    result = []

    res_files = [f for f in os.listdir(folder_path) if f.endswith(".res")]
    res_files.sort(key=lambda x: int(re.search(r"grid_(\d+)", x).group(1)))

    for filename in res_files:
        file_path = os.path.join(folder_path, filename)

        with open(file_path, "r") as f:
            values = [float(line.strip()) for line in f if line.strip()]
            if values:
                result.append(values)

    return result




def max_op_temp(db_temp_hot, dtimes_hot):
    occupied = []

    for i, (temp, dt) in enumerate(zip(db_temp_hot, dtimes_hot)):
        if dt.weekday() < 5 and 9 <= dt.hour <= 17:  
            occupied.append((i, temp))

    if occupied:
        max_index, max_temp = max(occupied, key=lambda x: x[1])
        return max_temp, max_index
    else:
        return None, None
    

def max_occupied_co2_all_rooms(co2_values, dtimes_hot):
    max_val = None
    max_index = None

    for room_co2 in co2_values:
        for i, (co2, dt) in enumerate(zip(room_co2, dtimes_hot)):
            if dt.weekday() < 5 and 9 <= dt.hour <= 17:  
                if (max_val is None) or (co2 > max_val):
                    max_val = co2
                    max_index = i

    return max_val, max_index


def min_op_temp(db_temp_hot, dtimes_hot):
    occupied = []

    for i, (temp, dt) in enumerate(zip(db_temp_hot, dtimes_hot)):
        if dt.weekday() < 5 and 9 <= dt.hour <= 17: 
            occupied.append((i, temp))

    if occupied:
        max_index, max_temp = min(occupied, key=lambda x: x[1])
        return max_temp, max_index
    else:
        return None, None



def adjust_weights(changed_key, new_value):
    other_keys = [k for k in st.session_state.weights if k != changed_key]
    remaining = 1.0 - new_value
    current_sum = sum(st.session_state.weights[k] for k in other_keys)

    if current_sum == 0:
        for k in other_keys:
            st.session_state.weights[k] = remaining / 2
    else:
        for k in other_keys:
            prop = st.session_state.weights[k] / current_sum
            st.session_state.weights[k] = round(remaining * prop, 4)

    st.session_state.weights[changed_key] = round(new_value, 4)
    st.session_state.last_changed = changed_key



def slider_with_auto_adjust(label, key, changed_key):
    current_value = st.sidebar.slider(
        label, 0.0, 1.0,
        st.session_state.weights[key],
        step=0.01,
        key=key + "_slider"
    )

    if st.session_state.last_changed != changed_key and current_value != st.session_state.weights[key]:
        adjust_weights(changed_key, current_value)
        
    return dict(st.session_state.weights)


