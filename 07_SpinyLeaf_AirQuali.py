# 07_SpinyLeaf_AirQuali.py
import os, sys, json, subprocess, zipfile
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

from ladybug.stat import STAT
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



def plot_indoor_vs_outdoor_rh(rh_val, out_rh_hot, usages_list, dtimes_hot, path_out):
    n_rooms = len(rh_val)
    if len(usages_list) != n_rooms:
        st.error("❌ Length of usages_list must match rh_val"); return
    if len(out_rh_hot) != len(dtimes_hot):
        st.error("❌ out_rh_hot and dtimes_hot must have the same length"); return

    path_out = Path(path_out); path_out.mkdir(parents=True, exist_ok=True)
    cols = st.columns(4)
    sns.set_theme()
    for i, room_rh in enumerate(rh_val):
        if len(room_rh) != len(dtimes_hot):
            st.error(f"❌ Room '{usages_list[i]}' has a mismatched RH list length."); return
        fig, ax = plt.subplots(figsize=(4.5, 3))
        ax.plot(dtimes_hot, out_rh_hot, label='Outdoor RH')
        ax.plot(dtimes_hot, room_rh, label='Indoor RH')
        ax.set_title(usages_list[i], fontsize=9)
        ax.set_ylabel("% RH"); ax.set_xlabel("")
        ax.xaxis.set_major_locator(mdates.DayLocator())
        ticks = ax.get_xticks(); labels = []
        for j, t in enumerate(ticks):
            label = ''
            if j in [0, 3, 6]:
                try:
                    label = mdates.num2date(t).strftime('%a\n%d %b')
                except Exception:
                    label = ''
            labels.append(label)
        ax.set_xticklabels(labels)
        ax.set_xlim([dtimes_hot[0], dtimes_hot[-1]])
        ax.grid(True); ax.legend(fontsize=8, loc="upper right")
        fig.tight_layout()
        fig.savefig(path_out / f"{usages_list[i].replace(' ', '_')}_rh_plot.png", dpi=150)
        cols[i % 4].pyplot(fig)
        plt.close(fig)


def plot_indoor_co2(co2_values, usages_list, dtimes, path_out):
    n_rooms = len(co2_values)
    if len(usages_list) != n_rooms:
        st.error("❌ Length of usages_list must match co2_values"); return
    if not co2_values or len(dtimes) != len(co2_values[0]):
        st.error("❌ dtimes must match length of each room's CO₂ data"); return

    path_out = Path(path_out); path_out.mkdir(parents=True, exist_ok=True)
    cols = st.columns(4)
    sns.set_theme()
    for i, room_co2 in enumerate(co2_values):
        if len(room_co2) != len(dtimes):
            st.error(f"❌ Room '{usages_list[i]}' has a mismatched CO₂ list length."); return
        fig, ax = plt.subplots(figsize=(4.5, 3))
        ax.plot(dtimes, room_co2, label='Indoor CO₂')
        ax.axhline(y=900,  color='gold',     linestyle='--', linewidth=1, label='900 ppm threshold')
        ax.axhline(y=1000, color='indianred', linestyle='--', linewidth=1, label='1000 ppm limit')
        ax.set_title(usages_list[i], fontsize=9)
        ax.set_ylabel("CO₂ (ppm)"); ax.set_xlabel("")
        plt.yticks(np.arange(350, 1250, 150), fontsize=12)
        ax.xaxis.set_major_locator(mdates.DayLocator())
        ticks = ax.get_xticks(); labels = []
        for j, t in enumerate(ticks):
            label = ''
            if j in [0, 3, 6]:
                try:
                    label = mdates.num2date(t).strftime('%a\n%d %b')
                except Exception:
                    label = ''
            labels.append(label)
        ax.set_xticklabels(labels)
        ax.set_xlim([dtimes[0], dtimes[-1]])
        ax.grid(True); ax.legend(fontsize=8, loc="upper right")
        fig.tight_layout()
        fig.savefig(path_out / f"{usages_list[i].replace(' ', '_')}_co2_plot.png", dpi=150)
        cols[i % 4].pyplot(fig)
        plt.close(fig)



# Page config

dim = 'Air Quality'

st.set_page_config(page_title= f'SpinyLeaf - {dim}', layout='wide', page_icon="C:/SpinyLeaf/Media/Spl_airquality_button_00.png")


script_dir=os.path.dirname(os.path.abspath(__file__))
logo_path=os.path.join(script_dir,'Media','Spl_airquality_button_00.png')
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
        SpinyLeaf - Air Quality
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
airq_folder = out_f / "Air_Quality"; airq_folder.mkdir(exist_ok=True)


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

# Materials/options (consistent with other pages)
glazing_options = {
    'U0.77_SHGC_0.77_SimpleGlazing_Window_02': 29,
    'U 0.56 SHGC 0.76 Dbl Clr 3mm/6mm Air' : 34,
    'U 0.30 SHGC 0.40 Dbl LoE (e2-.1) Tint 6mm/13mm Air': 34,
    'U 0.24 SHGC 0.11 Dbl LoE Elec Abs Colored 6mm/13mm Arg': 38,
    'U 0.19 SHGC 0.20 Trp LoE Film (55) Bronze 6mm/13mm Air': 40,
}

wall_options = {'GRC_Insul_Plasterboard': 40, 'Metal_Insul_GRC': 40}

window = st.sidebar.selectbox("Select Glazing type:", list(glazing_options.keys()))
_ = glazing_options[window]                 # NR not used directly here
_ = window_construction_by_identifier(window)  # validate identifier

wall_type = st.sidebar.selectbox("Select Wall type:", list(wall_options.keys()))
wall_r = st.sidebar.slider('Wall Insulation R', 3, 10, 3, 1)
roof_r = st.sidebar.slider('Roof R Value', 3, 10, 4, 1)
ground_r = st.sidebar.slider('Ground R Value', 3, 10, 4, 1)
operable = st.sidebar.toggle('Commercial Windows are Operable?', value=False)
offices_occupants_per_area = st.sidebar.select_slider(
    'Commercial - Occupants per m²',
    options=[0.06,0.07,0.08,0.09,0.10,0.11,0.12], value=0.08
)
occupants_per_bedroom = st.sidebar.select_slider(
    'Residential - Occupants per Bedroom',
    options=[1,1.5,2], value=1
)

# Session keys
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
        win_u = round(window_construction_by_identifier(window).u_factor, 2)
        shgc_c = round(window_construction_by_identifier(window).shgc, 2)
        s_trans = round(window_construction_by_identifier(window).solar_transmittance, 2)
        vis_trans = round(window_construction_by_identifier(window).visible_transmittance, 2)
        st.write('Windows U: ', win_u)
        st.write('SHGC: ', shgc_c)
        st.write('Sol_Transmittance: ', s_trans)
        st.write('Vis_Transmittance: ', vis_trans)
        st.write('Wall R: ', wall_r)
        st.write('Roof R: ', roof_r)
        st.write('Ground R: ', ground_r)


    st.success("Model successfully loaded!")



if st.session_state.get("model_done") and epw_file and ddy_file and stat_file:
    with st.form("Air Quality Study Form"):
        submitted = st.form_submit_button("Run Air Quality Study")
        if submitted:
            # Folders
            _hot_sim_folder   = airq_folder / "Extreme_Hot_Sim"
            _arq_co2_folder   = airq_folder / "CO2_Levels"
            _arq_rh_folder    = airq_folder / "Relative_Humidity"
            _arq_satisf_folder= airq_folder / "Air_Quality_Satisfaction"
            _rh_plot_folder   = airq_folder / "Relative_Humidity_Plots"
            _co2_plot_folder  = airq_folder / "CO2_Levels_Plots"
            for p in [_hot_sim_folder, _arq_co2_folder, _arq_rh_folder, _arq_satisf_folder,
                      _rh_plot_folder, _co2_plot_folder]:
                p.mkdir(parents=True, exist_ok=True)

            (airq_folder / "Materials.csv").write_text(st.session_state.get("df", pd.DataFrame()).to_csv(index=False))

            with st.spinner("Running Air Quality (thermal engine for CO₂/RH)...", show_time=True):


                epw = EPW(str(epw_file))
                db_temps = epw.dry_bulb_temperature
                out_rh = epw.relative_humidity
                stat = STAT(str(stat_file))
                hot_period = stat.extreme_hot_week

                db_temp_hot = list(db_temps.filter_by_analysis_period(hot_period).values)
                out_rh_hot  = list(out_rh.filter_by_analysis_period(hot_period).values)
                dtimes_hot  = hot_period.datetimes
                max_rh_idx  = int(np.argmax(out_rh_hot)) if out_rh_hot else 0

                solar, method = 2, "PolygonClipping"
                spi.run_sim_comfort(hb_model, _hot_sim_folder, str(epw_file), str(ddy_file), solar, method, hot_period)

                h_op_temps, rh_val, co2_values = spi.read_comf_results(_hot_sim_folder)

                _rh_hour  = [series[max_rh_idx] for series in rh_val]
                _, max_co2_idx = slu.max_occupied_co2_all_rooms(co2_values, dtimes_hot)
                _co2_hour = [series[max_co2_idx] for series in co2_values]

                lists_rh_levels  = [[val]  * len(faces) for val,  faces in zip(_rh_hour,  v_meshes)]
                lists_co2_levels = [[val]  * len(faces) for val,  faces in zip(_co2_hour, v_meshes)]

                slu.save_res_files(lists_co2_levels, v_meshes, _arq_co2_folder)
                slu.save_res_files(lists_rh_levels,  v_meshes, _arq_rh_folder)

                _air_quali_perc = sst.predict_air_quality(lists_co2_levels, lists_rh_levels)
                slu.save_res_files(_air_quali_perc, v_meshes, _arq_satisf_folder)

            with st.spinner("Preparing visuals...", show_time=True):
                v_height = 1200
                study_names = ["CO2_Levels", "Relative_Humidity", "Air_Quality_Satisfaction"]
                study_dics = {
                    "CO2_Levels": _arq_co2_folder,
                    "Relative_Humidity": _arq_rh_folder,
                    "Air_Quality_Satisfaction": _arq_satisf_folder
                }
                svi.view_study(view_hb_model, study_dics, study_names, study_names, v_height)

                st.success("Relative Humidity (week traces)")
                plot_indoor_vs_outdoor_rh(rh_val, out_rh_hot, usages_list, dtimes_hot, _rh_plot_folder)

                st.success("CO₂ Levels (week traces)")
                plot_indoor_co2(co2_values, usages_list, dtimes_hot, _co2_plot_folder)

                st.success("Air Quality study complete.")
