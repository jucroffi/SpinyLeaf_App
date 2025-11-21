
import os, sys, json, subprocess
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image

from honeybee.model import Model
from honeybee_energy.lib.constructions import window_construction_by_identifier

import Functions.App_Create_Model_Simulations_00 as spi
import Functions.App_Visualisation_lib as svi
import Functions.App_Satisfaction_T as sst
import Functions.App_Utils as slu



def ri_call(op, payload, timeout=600):
    worker = Path(__file__).with_name("00_RI_worker.py")
    proc = subprocess.Popen(
        [sys.executable, str(worker)],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True
    )
    out, err = proc.communicate(json.dumps({"op": op, "payload": payload}), timeout=timeout)
    if proc.returncode != 0:
        raise RuntimeError(err or out or "RI worker failed")
    res = json.loads(out)
    if not res.get("ok", True):
        raise RuntimeError(res.get("error", f"RI worker error on {op}"))
    return res



# Page config

dim = 'Acoustic'

st.set_page_config(page_title= f'SpinyLeaf - {dim}', layout='wide', page_icon="C:/SpinyLeaf/Media/Spl_sound_button_01.png")


script_dir=os.path.dirname(os.path.abspath(__file__))
logo_path=os.path.join(script_dir,'Media','Spl_sound_button_01.png')
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
        SpinyLeaf - Acoustic
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
ac_folder = out_f / "Acoustic_Comfort"; ac_folder.mkdir(exist_ok=True)


floor_options = {'Wood Tiles': 0.1, 'Carpet': 0.4}
ceiling_options = {'Plasterboard Ceiling': 0.1, 'Acoustic Tiles': 0.7}
glazing_options = {
    'U0.77_SHGC_0.77_SimpleGlazing_Window_02': 29,
    'U 0.56 SHGC 0.76 Dbl Clr 3mm/6mm Air': 34,
    'U 0.30 SHGC 0.40 Dbl LoE (e2-.1) Tint 6mm/13mm Air': 34,
    'U 0.24 SHGC 0.11 Dbl LoE Elec Abs Colored 6mm/13mm Arg': 38,
    'U 0.19 SHGC 0.20 Trp LoE Film (55) Bronze 6mm/13mm Air': 40,
}


window = st.sidebar.selectbox("Select Glazing type:", list(glazing_options.keys()))
wind_red = glazing_options[window]  
_ = window_construction_by_identifier(window)  

st.sidebar.write('Window Noise Reduction:', wind_red, 'dB')

wall_options = {'GRC_Insul_Plasterboard': 40, 'Metal_Insul_GRC': 40}
wall_type = st.sidebar.selectbox("Select Wall type:", list(wall_options.keys()))
wall_r = st.sidebar.slider('Wall Insulation R', 3, 10, 5, 1)
wall_red = wall_options[wall_type] + wall_r * 0.8 

st.sidebar.write('Wall Noise Reduction:', wall_red, 'dB')

floor_m = st.sidebar.selectbox("Select Floor Finish:", list(floor_options.keys()))
floor_a = floor_options[floor_m]
st.sidebar.write('Floor Absorptance:', floor_a)

ceiling_m = st.sidebar.selectbox("Select Ceiling Finish:", list(ceiling_options.keys()))
ceiling_a = ceiling_options[ceiling_m]
st.sidebar.write('Ceiling Absorptance:', ceiling_a)


roof_r = 5
ground_r = 5
operable = False
offices_occupants_per_area = 0.08
occupants_per_bedroom = 1


for k, v in {
    "hb_main_path": None,
    "view_hb_model_path": None,
    "model_done": False,
    "df": None,
    "v_meshes": None,
}.items():
    if k not in st.session_state:
        st.session_state[k] = v


model_path = main_f / 'Rhino_Model' / 'shared_model.3dm'


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
        w = window_construction_by_identifier(window)
        st.write('Windows U:', round(w.u_factor, 2))
        st.write('SHGC:', round(w.shgc, 2))
        st.write('Sol_Transmittance:', round(w.solar_transmittance, 2))
        st.write('Vis_Transmittance:', round(w.visible_transmittance, 2))
        st.write('Win Noise Reduction:', wind_red, 'dB')
        st.write('Wall Noise Reduction:', wall_red, 'dB')
        st.write('Floor Absorptance:', floor_a)
        st.write('Ceiling Absorptance:', ceiling_a)



if st.session_state.get("model_done"):
    with st.form("Acoustic Comfort Study Form"):
        submitted = st.form_submit_button("Run Acoustic Comfort Study")
        if submitted:
            _sound_levels_folder  = ac_folder / "Sound_Levels"
            _sound_satisf_folder  = ac_folder / "Sound_Satisfaction"
            for p in [_sound_levels_folder, _sound_satisf_folder]:
                p.mkdir(parents=True, exist_ok=True)

            with st.spinner("Running Acoustic Comfort...", show_time=True):
                
                rooms_db_res = ri_call("get_rooms_db", {
                    "src_3dm": str(model_path),
                    "wall_red": float(wall_red),
                    "wind_red": float(wind_red),
                    "floor_a": float(floor_a),
                    "ceiling_a": float(ceiling_a),
                })
                rooms_db = rooms_db_res["rooms_db"]  

                
                v_meshes = st.session_state.v_meshes
                lists_sound_levels = [
                    [val] * len(getattr(mesh, "face_centroids", []))
                    for val, mesh in zip(rooms_db, v_meshes)
                ]
                slu.save_res_files(lists_sound_levels, v_meshes, _sound_levels_folder)

                
                list_sound_satisf = sst.sound_satisf(lists_sound_levels)
                slu.save_res_files(list_sound_satisf, v_meshes, _sound_satisf_folder)
                _sound_satisf = [np.mean(sub) for sub in list_sound_satisf]

                
                df = st.session_state.df.copy()
                df["sound_level_dBA"] = rooms_db
                df["sound_satisf"] = _sound_satisf
                st.dataframe(df)

            
            with st.spinner("Preparing visuals...", show_time=True):
                v_height = 1200
                study_names = ["Sound_Levels", "Sound_Level_Satisfaction"]
                study_dics = {
                    "Sound_Levels": _sound_levels_folder,
                    "Sound_Level_Satisfaction": _sound_satisf_folder,
                }
                svi.view_study(view_hb_model, study_dics, study_names, study_names, v_height)
                st.success("Acoustic Comfort study complete.")
