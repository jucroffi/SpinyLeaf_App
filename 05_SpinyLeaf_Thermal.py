
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


def plot_indoor_vs_outdoor(h_or_c_op_temps, db_temp, usages_list, dtimes, path_out):
    n_rooms = len(h_or_c_op_temps)
    if len(usages_list) != n_rooms:
        st.error("❌ Length of usages_list must match indoor series"); return
    if len(db_temp) != len(dtimes):
        st.error("❌ Outdoor series and datetimes length mismatch"); return

    path_out = Path(path_out); path_out.mkdir(parents=True, exist_ok=True)
    cols = st.columns(4)
    sns.set_theme()
    for i, room_series in enumerate(h_or_c_op_temps):
        if len(room_series) != len(db_temp):
            st.error(f"❌ Room '{usages_list[i]}' indoor/outdoor series mismatch"); return
        fig, ax = plt.subplots(figsize=(4.5, 3))
        ax.plot(dtimes, db_temp, label='Outdoor Temp')
        ax.plot(dtimes, room_series, label='Indoor Temp')
        ax.set_title(usages_list[i], fontsize=9)
        ax.set_ylabel("°C"); ax.set_xlabel("")
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
        fig.savefig(path_out / f"{usages_list[i].replace(' ', '_')}_temp_plot.png", dpi=150)
        cols[i % 4].pyplot(fig)
        plt.close(fig)



# Page config

dim = 'Thermal Comfort'

st.set_page_config(page_title= f'SpinyLeaf - {dim}', layout='wide', page_icon="C:/SpinyLeaf/Media/Spl_thermal_button_00.png")


script_dir=os.path.dirname(os.path.abspath(__file__))
logo_path=os.path.join(script_dir,'Media','Spl_thermal_button_00.png')
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
        SpinyLeaf - Thermal Comfort
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
tc_folder = out_f / "Thermal_Comfort"; tc_folder.mkdir(exist_ok=True)


hot_sim_folder     = tc_folder / "Extreme_Hot_Sim"
cold_sim_folder    = tc_folder / "Extreme_Cold_Sim"
hot_temp_folder    = tc_folder / "Extreme_Hot_Week_Temp"
hot_sens_folder    = tc_folder / "Ext_Hot_Thermal_Sensation"
hot_satisf_folder  = tc_folder / "Ext_Hot_Thermal_Satisfaction"
hot_plot_folder    = tc_folder / "Ext_Hot_Temp_Plots"
cold_temp_folder   = tc_folder / "Extreme_Cold_Week_Temp"
cold_sens_folder   = tc_folder / "Ext_Cold_Thermal_Sensation"
cold_satisf_folder = tc_folder / "Ext_Cold_Thermal_Satisfaction"
cold_plot_folder   = tc_folder / "Ext_Cold_Temp_Plots"

# EPW / STAT / DDY selection
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
window_identifier = window
_ = window_construction_by_identifier(window_identifier)  

wall_type = st.sidebar.selectbox("Select Wall type:", list(wall_options.keys()))
wall_r = st.sidebar.slider('Wall Insulation R', 2, 10, 3, 1)
roof_r = st.sidebar.slider('Roof R Value', 2, 10, 4, 1)
ground_r = st.sidebar.slider('Ground R Value', 2, 10, 4, 1)
operable = st.sidebar.toggle('Commercial Windows are Operable?', value=False)
offices_occupants_per_area = st.sidebar.select_slider(
    'Commercial - Occupants per m²',
    options=[0.06,0.07,0.08,0.09,0.10,0.11,0.12], value=0.08
)
occupants_per_bedroom = st.sidebar.select_slider(
    'Residential - Occupants per Bedroom',
    options=[1,1.5,2], value=1
)


if "hb_main_path" not in st.session_state: st.session_state.hb_main_path = None
if "view_hb_model_path" not in st.session_state: st.session_state.view_hb_model_path = None
if "ids" not in st.session_state: st.session_state.ids = None
if "model_done" not in st.session_state: st.session_state.model_done = False


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
    with st.form("Thermal Comfort Study Form"):
        submitted = st.form_submit_button("Run Thermal Comfort Study")
        if submitted:

            for p in [tc_folder, hot_sim_folder, cold_sim_folder, hot_temp_folder, hot_sens_folder,
                      hot_satisf_folder, hot_plot_folder, cold_temp_folder, cold_sens_folder,
                      cold_satisf_folder, cold_plot_folder]:
                p.mkdir(parents=True, exist_ok=True)

            with st.spinner("Running Thermal Comfort Simulation...", show_time=True):


                epw = EPW(str(epw_file))
                stat = STAT(str(stat_file))
                hot_period = stat.extreme_hot_week
                cold_period = stat.extreme_cold_week

                db_temps = epw.dry_bulb_temperature
                db_temp_hot  = list(db_temps.filter_by_analysis_period(hot_period).values)
                db_temp_cold = list(db_temps.filter_by_analysis_period(cold_period).values)
                dtimes_hot   = hot_period.datetimes
                dtimes_cold  = cold_period.datetimes

        
                solar, method = 2, "PolygonClipping"
                spi.run_sim_comfort(hb_model, hot_sim_folder,  str(epw_file), str(ddy_file), solar, method, hot_period)
                spi.run_sim_comfort(hb_model, cold_sim_folder, str(epw_file), str(ddy_file), solar, method, cold_period)

                h_op_temps, h_rh, h_co2 = spi.read_comf_results(hot_sim_folder)
                c_op_temps, c_rh, c_co2 = spi.read_comf_results(cold_sim_folder)

                max_idx = int(np.argmax(db_temp_hot))  if db_temp_hot  else 0
                min_idx = int(np.argmin(db_temp_cold)) if db_temp_cold else 0

                hot_temp_hour  = [series[max_idx] for series in h_op_temps]
                cold_temp_hour = [series[min_idx] for series in c_op_temps]

                hot_t_sens   = sst.thermal_sensation(usages_list, hot_temp_hour)
                hot_t_satisf = sst.thermal_satisfaction(usages_list, hot_temp_hour)

                lists_hot_temps = [[temps] * len(faces) for temps, faces in zip(hot_temp_hour, st.session_state.v_centr)]
                slu.save_res_files(lists_hot_temps, v_meshes, hot_temp_folder)
                lists_hot_sens = [[temps] * len(faces) for temps, faces in zip(hot_t_sens, st.session_state.v_centr)]
                slu.save_res_files(lists_hot_sens, v_meshes, hot_sens_folder)
                lists_hot_satis = [[temps] * len(faces) for temps, faces in zip(hot_t_satisf, st.session_state.v_centr)]
                slu.save_res_files(lists_hot_satis, v_meshes, hot_satisf_folder)

                cold_t_sens   = sst.thermal_sensation(usages_list, cold_temp_hour)
                cold_t_satisf = sst.thermal_satisfaction(usages_list, cold_temp_hour)

                lists_cold_temps = [[temps] * len(faces) for temps, faces in zip(cold_temp_hour, st.session_state.v_centr)]
                slu.save_res_files(lists_cold_temps, v_meshes, cold_temp_folder)
                lists_cold_sens = [[temps] * len(faces) for temps, faces in zip(cold_t_sens, st.session_state.v_centr)]
                slu.save_res_files(lists_cold_sens, v_meshes, cold_sens_folder)
                lists_cold_satis = [[temps] * len(faces) for temps, faces in zip(cold_t_satisf, st.session_state.v_centr)]
                slu.save_res_files(lists_cold_satis, v_meshes, cold_satisf_folder)
                
                comfort_df = pd.DataFrame({
                    "room_ids": usages_list,
                    "extreme_hot_temp": hot_temp_hour,
                    "extreme_hot_sens": hot_t_sens,
                    "extreme_hot_satisf": hot_t_satisf,
                    "extreme_cold_temp": cold_temp_hour,
                    "extreme_cold_sens": cold_t_sens,
                    "extreme_cold_satisf": cold_t_satisf
                })

                st.dataframe(comfort_df)

                v_height = 1200
                
                st.subheader("Extreme Hot Week")
                study_names = ["Extreme_Hot_Week_Temp", "Ext_Hot_Thermal_Sensation", "Ext_Hot_Thermal_Satisfaction"]
                labels = [f"Extreme Hot Temp - {dtimes_hot[max_idx]} , {db_temp_hot[max_idx]} °C",
                          "Thermal Sensation", "Thermal Satisfaction"]
                study_dics = {
                    "Extreme_Hot_Week_Temp": hot_temp_folder,
                    "Ext_Hot_Thermal_Sensation": hot_sens_folder,
                    "Ext_Hot_Thermal_Satisfaction": hot_satisf_folder
                }
                svi.view_study(view_hb_model, study_dics, study_names, labels, v_height)
                plot_indoor_vs_outdoor(h_op_temps, db_temp_hot, usages_list, dtimes_hot, hot_plot_folder)

                
                st.subheader("Extreme Cold Week")
                study_names = ["Extreme_Cold_Week_Temp", "Ext_Cold_Thermal_Sensation", "Ext_Cold_Thermal_Satisfaction"]
                labels = [f"Extreme Cold Temp - {dtimes_cold[min_idx]} , {db_temp_cold[min_idx]} °C",
                          "Thermal Sensation", "Thermal Satisfaction"]
                study_dics = {
                    "Extreme_Cold_Week_Temp": cold_temp_folder,
                    "Ext_Cold_Thermal_Sensation": cold_sens_folder,
                    "Ext_Cold_Thermal_Satisfaction": cold_satisf_folder
                }
                svi.view_study(view_hb_model, study_dics, study_names, labels, v_height)
                plot_indoor_vs_outdoor(c_op_temps, db_temp_cold, usages_list, dtimes_cold, cold_plot_folder)

                st.success("Thermal Comfort study complete.")

