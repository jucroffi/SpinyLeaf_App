
import os, sys, json, subprocess, zipfile
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image

from ladybug.epw import EPW  

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

dim = 'Daylight'

st.set_page_config(page_title= f'SpinyLeaf - {dim}', layout='wide', page_icon="C:/SpinyLeaf/Media/Spl_daylight_button_00.png")


script_dir=os.path.dirname(os.path.abspath(__file__))
logo_path=os.path.join(script_dir,'Media','Spl_daylight_button_00.png')
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
        SpinyLeaf - Daylight
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


daylight_folder = out_f / "Daylight_Study"


epw_f = main_f / "EPWs"; epw_f.mkdir(exist_ok=True)
existing_zips = [f.name for f in epw_f.glob("*.zip")]
selected_zip = st.sidebar.selectbox("Select EPW ZIP:", existing_zips)
st.sidebar.write("OR")
uploaded_zip = st.sidebar.file_uploader("Upload a ZIP with .epw, .ddy, .stat", type="zip")

def _read_epw_zip(zip_path: Path):
    extract_folder = epw_f / zip_path.stem
    extract_folder.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_folder)
    epw_file = ddy_file = stat_file = None
    for file in extract_folder.glob("*"):
        if file.suffix.lower() == ".epw":  epw_file  = file
        if file.suffix.lower() == ".ddy":  ddy_file  = file
        if file.suffix.lower() == ".stat": stat_file = file
    return epw_file, ddy_file, stat_file

epw_file = ddy_file = stat_file = None
if uploaded_zip:
    save_path = epw_f / uploaded_zip.name
    with open(save_path, "wb") as f: f.write(uploaded_zip.getbuffer())
    epw_file, ddy_file, stat_file = _read_epw_zip(save_path)
    if epw_file and ddy_file and stat_file:
        st.sidebar.success("✅ All required files found in uploaded ZIP")
    else:
        st.sidebar.error("❌ ZIP must include one .epw, one .ddy, and one .stat file")

if selected_zip and not uploaded_zip:
    epw_file, ddy_file, stat_file = _read_epw_zip(epw_f / selected_zip)
    if epw_file and ddy_file and stat_file:
        st.sidebar.success("✅ All required files found in the ZIP")
    else:
        st.sidebar.warning("❌ Some required files are missing in selected ZIP")


glazing_options = {
    'U0.77_SHGC_0.77_SimpleGlazing_Window_02': 29,
    'U 0.56 SHGC 0.76 Dbl Clr 3mm/6mm Air' : 34,
    'U 0.30 SHGC 0.40 Dbl LoE (e2-.1) Tint 6mm/13mm Air': 34,
    'U 0.24 SHGC 0.11 Dbl LoE Elec Abs Colored 6mm/13mm Arg': 38,
    'U 0.19 SHGC 0.20 Trp LoE Film (55) Bronze 6mm/13mm Air': 40,
}
wall_options = {'GRC_Insul_Plasterboard': 40, 'Metal_Insul_GRC': 40}

window = st.sidebar.selectbox("Select Glazing type:", list(glazing_options.keys()))
_ = glazing_options[window]                
_ = window_construction_by_identifier(window)  

wall_type = st.sidebar.selectbox("Select Wall type:", list(wall_options.keys()))
wall_r = 5
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

                st.session_state.hb_main_path = str(hb_model_path)

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

            st.success("Model successfully loaded!")


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

    st.success("Model successfully loaded!")



if st.session_state.get("model_done") and epw_file and ddy_file and stat_file:
    with st.form("Daylight Study Form"):
        submitted = st.form_submit_button("Run Daylight Study")
        if submitted:
            
            _dl_sim_folder   = daylight_folder / "Daylight_Simulation"
            _da_res_folder   = daylight_folder / "DA_Annual"
            _da_mean_folder  = daylight_folder / "DA_Mean"
            _da_satisf_folder= daylight_folder / "DA_Satisfaction"
            _udi_res_folder  = daylight_folder / "UDI_Annual"
            _udi_mean_folder = daylight_folder / "UDI_Mean"
            _glare_sim_folder= daylight_folder / "Glare_Simulation"
            _glare_res_folder= daylight_folder / "Glare_Results"
            _glare_mean_folder= daylight_folder / "Glare_Mean"
            for p in [daylight_folder, _dl_sim_folder, _da_res_folder, _da_mean_folder, _da_satisf_folder,
                      _udi_res_folder, _udi_mean_folder, _glare_sim_folder, _glare_res_folder, _glare_mean_folder]:
                p.mkdir(parents=True, exist_ok=True)

            with st.spinner("Running Daylight Simulation...", show_time=True):

                
                spi.run_udi(hb_model, str(epw_file), _dl_sim_folder)

                
                da_folder = _dl_sim_folder / "annual_daylight" / "metrics" / "da"
                grid_info = _dl_sim_folder / "annual_daylight" / "results" / "grids_info.json"
                spi.convert_da_to_res(da_folder, _da_res_folder)
                
                if grid_info.exists():
                    ( _da_res_folder / "grids_info.json").write_bytes(grid_info.read_bytes())
                    slu.clean_grid_info_json(_da_res_folder / "grids_info.json")

                _da_values = slu.get_res_values(_da_res_folder)  
                _da_means = [[float(np.mean(sublist))] * len(sublist) for sublist in _da_values]
                slu.save_res_files(_da_means, v_meshes, _da_mean_folder)

                _da_satisf = sst.udi_satisf(_da_values)  
                slu.save_res_files(_da_satisf, v_meshes, _da_satisf_folder)

                st.subheader("Daylight Autonomy")
                v_height = 1200
                study_names = ["Daylight_Autonomy", "DA_mean", "Daylight_Satisfaction"]
                study_dics = {
                    "Daylight_Autonomy": _da_res_folder,
                    "DA_mean": _da_mean_folder,
                    "Daylight_Satisfaction": _da_satisf_folder
                }
                svi.view_study(view_hb_model, study_dics, study_names, study_names, v_height)

                
                udi_folder = _dl_sim_folder / "annual_daylight" / "metrics" / "udi"
                grid_info = _dl_sim_folder / "annual_daylight" / "results" / "grids_info.json"
                spi.convert_udi_to_res(udi_folder, _udi_res_folder)
                if grid_info.exists():
                    (_udi_res_folder / "grids_info.json").write_bytes(grid_info.read_bytes())
                    slu.clean_grid_info_json(_udi_res_folder / "grids_info.json")

                _udi_values = slu.get_res_values(_udi_res_folder)
                _udi_means = [[float(np.mean(sublist))] * len(sublist) for sublist in _udi_values]
                slu.save_res_files(_udi_means, v_meshes, _udi_mean_folder)

                st.subheader("Useful Daylight Illuminance")
                study_names = ["Useful_Daylight_Illuminance", "UDI_mean"]
                study_dics = {
                    "Useful_Daylight_Illuminance": _udi_res_folder,
                    "UDI_mean": _udi_mean_folder
                }
                svi.view_study(view_hb_model, study_dics, study_names, study_names, v_height)

            with st.spinner("Running Imageless Annual Glare Simulation...", show_time=True):

                st.subheader("Glare (Imageless Annual Glare)")
                spi.run_glare(hb_model, str(epw_file), _glare_sim_folder)
                glare_folder = _glare_sim_folder / "imageless_annual_glare" / "metrics" / "ga"
                grid_info = _glare_sim_folder / "imageless_annual_glare" / "results" / "grids_info.json"
                spi.convert_ga_to_res(glare_folder, _glare_res_folder)
                if grid_info.exists():
                    (_glare_res_folder / "grids_info.json").write_bytes(grid_info.read_bytes())
                    slu.clean_grid_info_json(_glare_res_folder / "grids_info.json")

                _ga_values = slu.get_res_values(_glare_res_folder)
                _ga_means = [[float(np.mean(sublist))] * len(sublist) for sublist in _ga_values]
                slu.save_res_files(_ga_means, v_meshes, _glare_mean_folder)

                study_names = ["Glare_Autonomy", "GA_mean"]
                study_dics = {
                    "Glare_Autonomy": _glare_res_folder,
                    "GA_mean": _glare_mean_folder
                }
                svi.view_study(view_hb_model, study_dics, study_names, study_names, v_height)

                st.success("Daylight study complete.")
