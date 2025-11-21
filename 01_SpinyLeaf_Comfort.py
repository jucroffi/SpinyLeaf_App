
import streamlit as st
import sys
import os
import json
import zipfile
import shutil
import subprocess
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import shap
from PIL import Image


import pandas as pd
import joblib

from ladybug.stat import STAT
from ladybug.epw import EPW

from honeybee.typing import clean_and_id_string
from honeybee.model import Model
from honeybee.room import Room
from honeybee_energy.lib.constructions import window_construction_by_identifier



import Functions.App_Create_Model_Simulations_00 as spi
import Functions.App_Visualisation_lib as svi
import Functions.App_Satisfaction_T as sst
import Functions.App_Utils as slu


def predict_row(row):
    features_res = row[['t_sens', 't_satisf', 'daylight', 'dl_satisf', 'sound', 'sound_satisf', 'air_quali']].values.reshape(1, -1)
    features_office = row[['t_sens', 't_satisf', 'dl_satisf', 'sound', 'sound_satisf', 'air_quali']].values.reshape(1, -1)

    if row['room_ids'].startswith('COMMERC'):
        return office_model.predict(features_office)[0]
    elif row['room_ids'].startswith('RESID'):
        return home_model.predict(features_res)[0]
    else:
        return None




def estimate_occupants(row):
    room = row['room_ids'].lower()
    if room.startswith('commerc'):
        return row['floor_area'] * offices_occupants_per_area
    elif room.startswith('resid'):
        return row['ap_num_beds'] * occupants_per_bedroom
    elif room.startswith('social'):
        return 0
    else:
        return np.nan



# Page config

dim = 'Comfort'

st.set_page_config(page_title= f'SpinyLeaf - {dim}', layout='wide', page_icon="C:/SpinyLeaf/Media/Spl_LOGO_COMFORT.png")


script_dir=os.path.dirname(os.path.abspath(__file__))
logo_path=os.path.join(script_dir,'Media','Spl_LOGO_COMFORT.png')
st.set_page_config(page_title= f'SpinyLeaf - {dim}',layout='wide',page_icon=logo_path)
logo_img=Image.open(logo_path)
resized_logo=logo_img.resize((150,150))

st.markdown(
    """
    <style>
        [data-testid="stSidebar"] {
            min-width: 320px;
            max-width: 320px;
        }
    </style>
    """,
    unsafe_allow_html=True
)

        
st.sidebar.markdown(
    """
    <h2 style='text-align: center; font-size: 22px; font-family: "Verdana"; 
               font-weight: 600; margin-bottom: 10px; color: #2E2E2E;'>
        SpinyLeaf - Comfort
    </h2>
    """,
    unsafe_allow_html=True
)

col1_,col2_,col3_=st.sidebar.columns([1,4,1])
with col2_:st.image(resized_logo)

#----------------


main_f = Path.home() / "SpinyLeaf_App"
main_f.mkdir(exist_ok=True)
out_f = main_f / "Wellbeing_Fostered_by_Design"
out_f.mkdir(exist_ok=True)
epw_f = main_f / "EPWs"
epw_f.mkdir(exist_ok=True)


existing_zips = [f.name for f in epw_f.glob("*.zip")]
selected_zip = st.sidebar.selectbox("Select EPW:", existing_zips)
st.sidebar.write("OR")
uploaded_zip = st.sidebar.file_uploader("Upload a ZIP file with .epw, .ddy, and .stat", type="zip")

epw_file = ddy_file = stat_file = None

if uploaded_zip:
    zip_name = uploaded_zip.name
    save_path = epw_f / zip_name

    with open(save_path, "wb") as f:
        f.write(uploaded_zip.getbuffer())

    extract_folder = epw_f / zip_name.rsplit(".", 1)[0]
    extract_folder.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(save_path, "r") as zip_ref:
        zip_ref.extractall(extract_folder)

    for file in extract_folder.glob("*"):
        if file.suffix == ".epw":
            epw_file = file
        elif file.suffix == ".ddy":
            ddy_file = file
        elif file.suffix == ".stat":
            stat_file = file

    if epw_file and ddy_file and stat_file:
        st.sidebar.success("✅ All required files found in uploaded ZIP")
    else:
        st.sidebar.error("❌ ZIP must include one .epw, one .ddy, and one .stat file")

if selected_zip and not uploaded_zip:
    selected_folder = epw_f / selected_zip.rsplit(".", 1)[0]
    for file in selected_folder.glob("*"):
        if file.suffix == ".epw":
            epw_file = file
        elif file.suffix == ".ddy":
            ddy_file = file
        elif file.suffix == ".stat":
            stat_file = file

    if epw_file and ddy_file and stat_file:
        st.sidebar.success("✅ All required files found in the ZIP")
    else:
        st.sidebar.warning("❌ Some required files missing in selected ZIP")


epw_file = str(epw_file) if epw_file else None
ddy_file = str(ddy_file) if ddy_file else None
stat_file = str(stat_file) if stat_file else None


floor_options = {'Wood Tiles': 0.1, 'Carpet': 0.4}
ceiling_options = {'Plasterboard Ceiling': 0.1, 'Acoustic Tiles': 0.7}
glazing_options = {
    'U0.77_SHGC_0.77_SimpleGlazing_Window_02': 25,
    'U 0.56 SHGC 0.76 Dbl Clr 3mm/6mm Air': 32,
    'U 0.30 SHGC 0.40 Dbl LoE (e2-.1) Tint 6mm/13mm Air': 32,
    'U 0.24 SHGC 0.11 Dbl LoE Elec Abs Colored 6mm/13mm Arg': 38,
    'U 0.19 SHGC 0.20 Trp LoE Film (55) Bronze 6mm/13mm Air': 40,
}
wall_options = {'GRC_Insul_Plasterboard': 40, 'Metal_Insul_GRC': 40}

win_label = st.sidebar.selectbox("Select Glazing type:", list(glazing_options.keys()))
wind_red = glazing_options[win_label]
st.sidebar.write('Window Noise Reduction:', wind_red, 'dB')
window = win_label
win = window_construction_by_identifier(window)

wall_type = st.sidebar.selectbox("Select Wall type:", list(wall_options.keys()))
wall_r = st.sidebar.slider(label='Wall Insulation R', min_value=2, max_value=10, value=3, step=1, key='wall_r')
wall_red = wall_options[wall_type] + wall_r * 0.8
st.sidebar.write('Wall Noise Reduction:', wall_red, 'dB')

roof_r = st.sidebar.slider(label='Roof R Value', min_value=2, max_value=10, value=4, step=1, key='roof_r')
ground_r = st.sidebar.slider(label='Ground R Value', min_value=2, max_value=10, value=4, step=1, key='ground_r')

floor_m = st.sidebar.selectbox("Select Floor Finish:", list(floor_options.keys()))
ceiling_m = st.sidebar.selectbox("Select Ceiling Finish:", list(ceiling_options.keys()))
floor_a = floor_options[floor_m]
st.sidebar.write('Floor Absorptance: ', floor_a)
ceiling_a = ceiling_options[ceiling_m]
st.sidebar.write('Ceiling Absorptance: ', ceiling_a)

operable = st.sidebar.toggle(label='Commercial Windows are Operable?', value=False, key='operable')
operable_status = 'Yes' if operable else 'No'
st.sidebar.write(operable_status)
offices_occupants_per_area = st.sidebar.select_slider(label='Commercial - Occupants per m2',
                                                      options=[0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12],
                                                      value=0.08)
occupants_per_bedroom = st.sidebar.select_slider(label='Residential - Occupants per Bedroom',
                                                 options=[1, 1.5, 2], value=1)

usage_dict = {
    0: "COMMERC", 1: "CONTEXT", 2: "RESID",
    3: "GREEN", 4: "BALCONIES", 5: "CORE",
    6: "SOCIAL_L1", 7: "SOCIAL_L2", 8: "SOCIAL_L3",
    9: "SOCIAL_L4_RESID", 10: "SOCIAL_L4_COMMERC",
    11: "SOCIAL_OUTDOOR_ALL", 12: "SOCIAL_OUTDOOR_RESID", 13: "SOCIAL_OUTDOOR_COMMERC"
}


model_path = main_f / 'Rhino_Model' / 'shared_model.3dm'


if "model_loaded" not in st.session_state:
    st.session_state.model_loaded = False
if "model_done" not in st.session_state:
    st.session_state.model_done = False


def ri_call(op, payload, timeout=600):
    worker = Path(__file__).with_name("00_RI_worker.py")
    proc = subprocess.Popen(
        [sys.executable, str(worker)],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True
    )
    out, err = proc.communicate(json.dumps({"op": op, "payload": payload}), timeout=timeout)
    if proc.returncode != 0:
        raise RuntimeError(err or out or "ri_worker failed")
    return json.loads(out)



with st.form("Get/Update Model"):
    submitted = st.form_submit_button("Get/Update Model")

    if submitted:
        if not model_path.exists():
            st.error("Model not found. Please click the Rhino button to export the 3DM first.")
        else:
            with st.spinner("Building Honeybee models from Rhino geometry... this can take a moment"):
                
                payload = {
                    "src_3dm": str(model_path),
                    "out_dir": str(out_f),                
                    "window_identifier": window,
                    "wall_type": wall_type,
                    "wall_r": float(wall_r),
                    "roof_r": float(roof_r),
                    "ground_r": float(ground_r),
                    "operable": bool(operable),
                    "offices_occ_per_area": float(offices_occupants_per_area),
                    "res_occ_per_bedroom": float(occupants_per_bedroom)
                }
                res = ri_call("build_hb_models", payload)

                
                hb_model_path = Path(res["hb_model_path"])
                view_hb_model_path = Path(res["view_hb_model_path"])
                hb_model = Model.from_hbjson(hb_model_path)
                view_hb_model = Model.from_hbjson(view_hb_model_path)

                ids = res["ids"]
                storeys = res["storeys"]
                av_orients = res["av_orients"]
                room_areas = res["room_areas"]
                balcon_areas = res["balcon_areas"]
                n_beds = res["n_beds"]
                usages_list = res["usages_list"]

                dist = 1
                v_meshes, sensor_grid = spi.create_sensors(view_hb_model, dist)
        
                dist = 1
                res2 = ri_call("prep_view_assets", {
                    "hb_view_path": str(view_hb_model_path),
                    "out_dir": str(out_f),
                    "dist": float(dist)
                })
                view_hb_model = Model.from_hbjson(Path(res2["view_hb_path"]))
                v_centr = res2["v_centr"]

                hb_model.properties.radiance.sensor_grids = view_hb_model.properties.radiance.sensor_grids
                

                df = pd.DataFrame(data={
                    'room_ids': ids,
                    'floor_level': storeys,
                    'average_orient': av_orients,
                    'floor_area': room_areas,
                    'balcony_area': balcon_areas,
                    'ap_num_beds': n_beds
                })

                st.session_state.model_loaded = True
                st.session_state.model_done = True
                st.session_state.view_hb_model = view_hb_model
                st.session_state.hb_model = hb_model
                st.session_state.v_meshes = v_meshes
                st.session_state.v_centr = v_centr
                st.session_state.df = df
                st.session_state.usages_list = usages_list


if st.session_state.get("model_done"):
    hb_model = st.session_state.hb_model
    view_hb_model = st.session_state.view_hb_model
    v_meshes = st.session_state.v_meshes
    v_centr = st.session_state.v_centr
    df = st.session_state.df
    usages_list = st.session_state.usages_list

    col1, col2 = st.columns(2)
    with col2:
        name = 'test_0'
        with st.spinner("Generating 3D Model..."):
            v_height = 800
            svi.get_vtkjs(hb_model, out_f, name)
            svi.get_views(name, out_f, v_height)

    with col1:
        st.write(df)
        win_u = round(window_construction_by_identifier(window).u_factor, 2)
        shgc_c = round(window_construction_by_identifier(window).shgc, 2)
        s_trans = round(window_construction_by_identifier(window).solar_transmittance, 2)
        vis_trans = round(window_construction_by_identifier(window).visible_transmittance, 2)
        st.write('Windows U: ', win_u)
        st.write('SHGC: ', shgc_c)
        st.write('Sol_Transmittance: ', s_trans)
        st.write('Vis_Transmittance: ', vis_trans)
        st.write('Win Noise Reduction: ', wind_red, 'dB')
        st.write('Wall Noise Reduction: ', wall_red, 'dB')
        st.write('Floor Absorptance: ', floor_a)
        st.write('Ceiling Absorptance: ', ceiling_a)
        st.write('Wall R: ', wall_r)
        st.write('Roof R: ', roof_r)
        st.write('Ground R: ', ground_r)

    df_materials = pd.DataFrame(data={
        'Window_type': [win_label],
        'Windows_U': [win_u],
        'Win_reduction': [wind_red],
        'SHGC': [shgc_c],
        'sol_transmit': [s_trans],
        'vis_transmit': [vis_trans],
        'Wall_R': [wall_r],
        'Wall_reducrion': [wall_red],
        'Roof_R': [roof_r],
        'Ground_R': [ground_r]
    })

    st.success("Model successfully loaded!")


if "run_comfort" not in st.session_state:
    st.session_state.run_comfort = False
    st.session_state.comfort_done = False

if st.session_state.get("model_done"):
    with st.form("Comfort Dimension Study Form"):
        submitted = st.form_submit_button("Run Comfort Dimension Study")

        if submitted:
            out_f_comfort = out_f / "Comfort_Dimension"
            _hot_sim_folder = out_f_comfort / "Extreme_Hot_Sim"
            _cold_sim_folder = out_f_comfort / "Extreme_Cold_Sim"

            _hot_otemp_folder = out_f_comfort / "Extreme_Hot_Week_Temp"
            _hot_sens_folder = out_f_comfort / "Ext_Hot_Thermal_Sensation"
            _hot_satisf_folder = out_f_comfort / "Ext_Hot_Thermal_Satisfaction"

            _cold_otemp_folder = out_f_comfort / "Extreme_Cold_Week_Temp"
            _cold_sens_folder = out_f_comfort / "Ext_Cold_Thermal_Sensation"
            _cold_satisf_folder = out_f_comfort / "Ext_Cold_Thermal_Satisfaction"

            _arq_co2_folder = out_f_comfort / "CO2_Levels"
            _arq_rh_folder = out_f_comfort / "Relative_Humidity"
            _arq_satisf_folder = out_f_comfort / "Air_Quality_Satisfaction"

            _sound_levels_folder = out_f_comfort / "Sound_Levels"
            _sound_satisf_folder = out_f_comfort / "Sound_Satisfaction"

            _dl_sim_folder = out_f_comfort / "Daylight_Simulation"
            _udi_res_folder = out_f_comfort / "DA_Annual"
            _udi_mean_folder = out_f_comfort / "DA_Mean"
            _udi_satisf_folder = out_f_comfort / "DA_Satisfaction"

            _comfort_satisf_folder = out_f_comfort / "_Comfort_Satisfaction"

            df_folder = out_f_comfort / "Comfort.csv"
            materials = out_f_comfort / "Materials.csv"

            st.session_state.run_comfort = True

        if st.session_state.run_comfort:
            if df_folder.exists():
                os.remove(df_folder)
                if materials.exists():
                    os.remove(materials)

            comfort_df = df.copy()
            df_materials.to_csv(materials)

            with st.spinner("Running Thermal Comfort Simulation...", show_time=True):
                solar = 2
                method = 'PolygonClipping'

                epw_data = EPW(epw_file)
                db_temps = epw_data.dry_bulb_temperature
                out_rh = epw_data.relative_humidity

                hot_period = STAT(stat_file).extreme_hot_week
                cold_period = STAT(stat_file).extreme_cold_week

                db_temp_hot = (db_temps.filter_by_analysis_period(hot_period)).values
                db_temp_cold = (db_temps.filter_by_analysis_period(cold_period)).values
                out_rh_hot = (out_rh.filter_by_analysis_period(hot_period)).values
                out_rh_cold = (out_rh.filter_by_analysis_period(cold_period)).values
                dtimes_hot = hot_period.datetimes
                dtimes_cold = cold_period.datetimes
                max_temp, max_idx = slu.max_op_temp(db_temp_hot, dtimes_hot)
                min_temp, min_idx = slu.min_op_temp(db_temp_cold, dtimes_cold)
                max_rh, max_rh_idx = slu.max_op_temp(out_rh_hot, dtimes_hot)

                spi.run_sim_comfort(hb_model, _hot_sim_folder, epw_file, ddy_file, solar, method, hot_period)
                spi.run_sim_comfort(hb_model, _cold_sim_folder, epw_file, ddy_file, solar, method, cold_period)

                h_op_temps, rh_val, co2_values = spi.read_comf_results(_hot_sim_folder)
                co2_max_values = [(max(sublist), sublist.index(max(sublist))) for sublist in co2_values]
                rh_max_values = [(max(sublist), sublist.index(max(sublist))) for sublist in rh_val]
                hot_temp_hour = [op_t[max_idx] for op_t in h_op_temps]

                _rh_hour = [rh[max_rh_idx] for rh in rh_val]
                max_co2, max_co2_idx = slu.max_occupied_co2_all_rooms(co2_values, dtimes_hot)
                _co2_hour = [co2[max_co2_idx] for co2 in co2_values]

                hot_t_sens = sst.thermal_sensation(usages_list, hot_temp_hour)
                hot_t_satisf = sst.thermal_satisfaction(usages_list, hot_temp_hour)

                c_op_temps, rh_val, co2_values = spi.read_comf_results(_cold_sim_folder)
                min_values = [(min(sublist), sublist.index(min(sublist))) for sublist in c_op_temps]

                cold_temp_hour = [op_t[min_idx] for op_t in c_op_temps]
                cold_rh_hour = [rh[min_idx] for rh in rh_val]
                cold_t_sens = sst.thermal_sensation(usages_list, cold_temp_hour)
                cold_t_satisf = sst.thermal_satisfaction(usages_list, cold_temp_hour)

                lists_hot_temps = [[temps] * len(faces) for temps, faces in zip(hot_temp_hour, st.session_state.v_centr)]
                slu.save_res_files(lists_hot_temps, v_meshes, _hot_otemp_folder)
                lists_hot_sens = [[temps] * len(faces) for temps, faces in zip(hot_t_sens, st.session_state.v_centr)]
                slu.save_res_files(lists_hot_sens, v_meshes, _hot_sens_folder)
                lists_hot_satis = [[temps] * len(faces) for temps, faces in zip(hot_t_satisf, st.session_state.v_centr)]
                slu.save_res_files(lists_hot_satis, v_meshes, _hot_satisf_folder)

                lists_cold_temps = [[temps] * len(faces) for temps, faces in zip(cold_temp_hour, st.session_state.v_centr)]
                slu.save_res_files(lists_cold_temps, v_meshes, _cold_otemp_folder)
                lists_cold_sens = [[temps] * len(faces) for temps, faces in zip(cold_t_sens, st.session_state.v_centr)]
                slu.save_res_files(lists_cold_sens, v_meshes, _cold_sens_folder)
                lists_cold_satis = [[temps] * len(faces) for temps, faces in zip(cold_t_satisf, st.session_state.v_centr)]
                slu.save_res_files(lists_cold_satis, v_meshes, _cold_satisf_folder)

                lists_co2_levels = [[co2s] * len(faces) for co2s, faces in zip(_co2_hour, st.session_state.v_centr)]
                slu.save_res_files(lists_co2_levels, v_meshes, _arq_co2_folder)
                lists_rh_levels = [[rhs] * len(faces) for rhs, faces in zip(_rh_hour, st.session_state.v_centr)]
                slu.save_res_files(lists_rh_levels, v_meshes, _arq_rh_folder)

                _air_quali_perc = sst.predict_air_quality(lists_co2_levels, lists_rh_levels)
                slu.save_res_files(_air_quali_perc, v_meshes, _arq_satisf_folder)
                air_quali = [np.mean(sublist) for sublist in _air_quali_perc]

                comfort_df['extreme_hot_temp'] = hot_temp_hour
                comfort_df['extreme_hot_sens'] = hot_t_sens
                comfort_df['extreme_hot_satisf'] = hot_t_satisf
                comfort_df['extreme_cold_temp'] = cold_temp_hour
                comfort_df['extreme_cold_sens'] = cold_t_sens
                comfort_df['extreme_cold_satisf'] = cold_t_satisf
                comfort_df['extreme_RH'] = _rh_hour
                comfort_df['extreme_CO2'] = _co2_hour
                comfort_df['air_quali'] = air_quali

                study_names = ["Extreme_Hot_Week_Temp", "Ext_Hot_Thermal_Sensation", "Ext_Hot_Thermal_Satisfaction"]
                labels = [f"Extreme Hot Temp - {dtimes_hot[max_idx]}, {db_temp_hot[max_idx]} C", "Thermal Sensation", "Thermal Satisfaction"]
                study_dics = {
                    "Extreme_Hot_Week_Temp": _hot_otemp_folder,
                    "Ext_Hot_Thermal_Sensation": _hot_sens_folder,
                    "Ext_Hot_Thermal_Satisfaction": _hot_satisf_folder
                }
                st.write(comfort_df[['room_ids', 'extreme_hot_temp', 'extreme_hot_sens', 'extreme_hot_satisf']])
                svi.view_study(view_hb_model, study_dics, study_names, labels, v_height)

                study_names = ["Extreme_Cold_Week_Temp", "Ext_Cold_Thermal_Sensation", "Ext_Cold_Thermal_Satisfaction"]
                labels = [f"Extreme Cold Temp - {dtimes_cold[min_idx]}, {db_temp_cold[min_idx]} C", "Thermal Sensation", "Thermal Satisfaction"]
                study_dics = {
                    "Extreme_Cold_Week_Temp": _cold_otemp_folder,
                    "Ext_Cold_Thermal_Sensation": _cold_sens_folder,
                    "Ext_Cold_Thermal_Satisfaction": _cold_satisf_folder
                }
                st.write(comfort_df[['room_ids', 'extreme_cold_temp', 'extreme_cold_sens', 'extreme_cold_satisf']])
                svi.view_study(view_hb_model, study_dics, study_names, labels, v_height)

            with st.spinner("Running Air Quality Study..."):
                study_names = ["CO2_Levels", "Relative_Humidity", "Air_Quality_Satisfaction"]
                study_dics = {
                    "CO2_Levels": _arq_co2_folder,
                    "Relative_Humidity": _arq_rh_folder,
                    "Air_Quality_Satisfaction": _arq_satisf_folder
                }
                st.write(comfort_df[['room_ids', 'extreme_CO2', 'extreme_RH', 'air_quali']])
                svi.view_study(view_hb_model, study_dics, study_names, study_names, v_height)

                st.session_state.thermal_done = True

            if st.session_state.thermal_done:
                with st.spinner("Running Daylight Simulation...", show_time=True):
                    spi.run_udi(hb_model, epw_file, _dl_sim_folder)

                    udi_folder = _dl_sim_folder / "annual_daylight" / "metrics" / "da"
                    grid_info = _dl_sim_folder / "annual_daylight" / "results" / "grids_info.json"
                    files = os.listdir(udi_folder)
                    spi.convert_da_to_res(udi_folder, _udi_res_folder)
                    shutil.copy2(grid_info, _udi_res_folder)
                    new_grid_path = _udi_res_folder / "grids_info.json"
                    slu.clean_grid_info_json(new_grid_path)

                    _udi_values = slu.get_res_values(_udi_res_folder)
                    _udi_means = [[np.mean(sublist)] * len(sublist) for sublist in _udi_values]
                    slu.save_res_files(_udi_means, v_meshes, _udi_mean_folder)
                    udi_mean_room = [np.mean(sublist) for sublist in _udi_means]

                    _udi_satisf = sst.udi_satisf(_udi_values)
                    slu.save_res_files(_udi_satisf, v_meshes, _udi_satisf_folder)
                    udi_satisf_room = [np.mean(sublist) for sublist in _udi_satisf]
                    _dl_percep = sst.dl_percep(_udi_values)
                    dl_percep_room = [np.mean(sublist) for sublist in _dl_percep]

                    comfort_df['DA_mean'] = udi_mean_room
                    comfort_df['daylight_percep'] = dl_percep_room
                    comfort_df['daylight_satisf'] = udi_satisf_room

                    study_names = ["Daylight_Autonomy", "DA_mean", "Daylight_Satisfaction"]
                    study_dics = {
                        "Daylight_Autonomy": _udi_res_folder,
                        "DA_mean": _udi_mean_folder,
                        "Daylight_Satisfaction": _udi_satisf_folder
                    }
                    svi.view_study(view_hb_model, study_dics, study_names, study_names, v_height)

                    st.session_state.daylight_done = True

            if st.session_state.thermal_done:
                with st.spinner("Running Acoustic Comfort Simulation...", show_time=True):
                    
                    acous_res = ri_call("get_rooms_db", {"src_3dm": str(model_path), "wall_red": float(wall_red), "wind_red": float(wind_red), "floor_a": float(floor_a), "ceiling_a": float(ceiling_a)})
                    rooms_db = acous_res["rooms_db"]

                    lists_sound_levels = [[sound] * len(faces) for sound, faces in zip(rooms_db, st.session_state.v_centr)]
                    slu.save_res_files(lists_sound_levels, v_meshes, _sound_levels_folder)

                    list_sound_satisf = sst.sound_satisf(lists_sound_levels)
                    slu.save_res_files(list_sound_satisf, v_meshes, _sound_satisf_folder)
                    _sound_satisf = [np.mean(sublist) for sublist in list_sound_satisf]
                    list_sound_percep = sst.sound_satisf(lists_sound_levels)
                    _sound_percep = [np.mean(sublist) for sublist in list_sound_percep]

                    study_names = ["Sound_Levels", "Sound_Level_Satisfaction"]
                    study_dics = {
                        "Sound_Levels": _sound_levels_folder,
                        "Sound_Level_Satisfaction": _sound_satisf_folder
                    }

                    comfort_df['sound_level'] = rooms_db
                    comfort_df['sound_percep'] = _sound_percep
                    comfort_df['sound_satisf'] = _sound_satisf

                    svi.view_study(view_hb_model, study_dics, study_names, study_names, v_height)

                    st.session_state.acoustic_done = True

                with st.spinner("Comfort Dimension Score...", show_time=True):
                    comfort_df['t_satisf'] = comfort_df[['extreme_hot_satisf', 'extreme_cold_satisf']].min(axis=1)
                    comfort_df['t_sens'] = np.where(
                        comfort_df['extreme_hot_satisf'] < comfort_df['extreme_cold_satisf'],
                        comfort_df['extreme_hot_sens'],
                        comfort_df['extreme_cold_sens']
                    )

                    ML_df = comfort_df[['room_ids', 'floor_area', 't_sens', 't_satisf',
                                        'daylight_percep', 'daylight_satisf',
                                        'sound_percep', 'sound_satisf', 'air_quali']]

                    ML_df = ML_df.rename(columns={'daylight_percep': 'daylight',
                                                    'daylight_satisf': 'dl_satisf',
                                                    'sound_percep': 'sound'})
                    ML_df = ML_df[['room_ids', 'floor_area', 't_sens', 't_satisf',
                                    'daylight', 'dl_satisf', 'sound', 'sound_satisf', 'air_quali']]

                    ML_df_folder = out_f_comfort / "ML_df.csv"
                    ML_df.to_csv(ML_df_folder)

                    office_model = joblib.load("C:/SpinyLeaf/ML_Models/office_ml_model.joblib")
                    home_model = joblib.load("C:/SpinyLeaf/ML_Models/home_ml_model.joblib")


                    ML_df['comfort_satisfaction'] = ML_df.apply(predict_row, axis=1)
                    comfort_df['comfort_satisfaction'] = ML_df['comfort_satisfaction']

                    mask = comfort_df['room_ids'].str.startswith(('SOCIAL', 'CORE'))
                    satisfaction_cols = ['extreme_hot_sens', 'extreme_hot_satisf', 'daylight_satisf', 'sound_satisf', 'air_quali']
                    comfort_df.loc[mask, satisfaction_cols] = None

                    comfort_df.to_csv(df_folder)
                    st.write(comfort_df[['room_ids', 'extreme_hot_sens', 'extreme_hot_satisf', 'extreme_cold_sens', 'extreme_cold_satisf',
                                            'air_quali', 'daylight_satisf', 'sound_satisf', 'comfort_satisfaction']])

                    satisf_list = comfort_df['comfort_satisfaction'].tolist()
                    lists_comfort = [[satis] * len(faces) for satis, faces in zip(satisf_list, st.session_state.v_centr)]
                    slu.save_res_files(lists_comfort, v_meshes, _comfort_satisf_folder)

                    study_names = ["Comfort_Satisfaction"]
                    study_dics = {"Comfort_Satisfaction": _comfort_satisf_folder}

                    col1, col2, col3 = st.columns(3)
                    svi.color_vtkjs_from_results(view_hb_model, _comfort_satisf_folder, study_name="Comfort_Satisfaction")

                    v_height = 1200
                    with col1:
                        st.success('Comfort Dimension Satisfaction')
                        svi.get_views("Comfort_Satisfaction", _comfort_satisf_folder, v_height)

                    with col2:
                        variables = ['t_satisf', 'dl_satisf', 'sound_satisf', 'air_quali']
                        title = 'Comfort Factors Satisfaction'
                        fig = svi.get_spider_matplotlib(ML_df, variables, title, out_f_comfort)
                        fig2 = svi.plot_sbar_satisfaction(ML_df, variables, title, out_f_comfort)
                        st.success("Response Distribution per Factor (%)")
                        with st.container(height=int(v_height * 3 / 5)):
                            st.pyplot(fig)
                        with st.container(height=int(v_height * 2 / 5)):
                            st.pyplot(fig2)

                    with col3:
                        title = 'Comfort Dimension Satisfaction'
                        fig = svi.get_violing(comfort_df, 'comfort_satisfaction', title, out_f_comfort)
                        level_column = 'comfort_satisfaction'
                        fig2 = svi.plot_satisf_bar(comfort_df, level_column, title, out_f_comfort)
                        st.success("Satisfaction Score Distribution")
                        with st.container(height=int(v_height * 3 / 5)):
                            st.pyplot(fig)
                        with st.container(height=int(v_height * 2 / 5)):
                            st.pyplot(fig2)
