
import os, sys, json, subprocess
from pathlib import Path

import streamlit as st
from PIL import Image
import pandas as pd
import numpy as np

from honeybee.model import Model
from honeybee.room import Room
from honeybee.typing import clean_and_id_string
from honeybee_energy.lib.constructions import window_construction_by_identifier


import Functions.App_Create_Model_Simulations_00 as spi
import Functions.App_Visualisation_lib as svi
import Functions.App_Satisfaction_T as sst
import Functions.App_Utils as slu


def ri_call(op, payload, timeout=600):
    worker = Path(__file__).with_name("00_RI_worker.py")
    proc = subprocess.Popen([sys.executable, str(worker)],
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True)
    out, err = proc.communicate(json.dumps({"op": op, "payload": payload}), timeout=timeout)
    if proc.returncode != 0:
        raise RuntimeError(err or out or "Delight worker failed")
    return json.loads(out)


def estimate_occupants(row, comm_occup_per_area, occupants_per_bedroom):
    room = row['room_ids'].lower()
    if room.startswith('commerc'):
        return row['floor_area'] * comm_occup_per_area
    elif room.startswith('resid'):
        return row['ap_num_beds'] * occupants_per_bedroom
    elif room.startswith('social'):
        return 0
    else:
        return np.nan

def compute_scalar_wdss(social_1_area, social_2_area, social_3_area,
                        social_4_resid_area, social_4_office_area,
                        resid_n_occup, office_n_occup, alpha=0.5):
    total_n_occup = resid_n_occup + office_n_occup
    if total_n_occup == 0:
        return 0.0, 0.0

    social_1_occup = social_1_area / total_n_occup
    social_2_occup = social_2_area / total_n_occup
    social_3_occup = social_3_area / total_n_occup

    social_4_resid_occup  = social_4_resid_area  / resid_n_occup  if resid_n_occup  > 0 else 0.0
    social_4_office_occup = social_4_office_area / office_n_occup if office_n_occup > 0 else 0.0

    resid_asr  = social_1_occup + social_2_occup + social_3_occup + social_4_resid_occup
    office_asr = social_1_occup + social_2_occup + social_3_occup + social_4_office_occup

    P_resid  = 1.0 if (social_3_area > 0 or social_4_resid_area  > 0) else alpha
    P_office = 1.0 if (social_3_area > 0 or social_4_office_area > 0) else alpha

    w1, w2, w3, w4 = 0.5, 0.5, 1.0, 1.0

    resid_num  = w1*social_1_occup + w2*social_2_occup + w3*social_3_occup + w4*social_4_resid_occup
    office_num = w1*social_1_occup + w2*social_2_occup + w3*social_3_occup + w4*social_4_office_occup

    resid_wdss  = P_resid  * (resid_num  / resid_asr)  if resid_asr  > 0 else 0.0
    office_wdss = P_office * (office_num / office_asr) if office_asr > 0 else 0.0
    return office_wdss, resid_wdss


# Page config

dim = 'Social'

st.set_page_config(page_title= f'SpinyLeaf - {dim}', layout='wide', page_icon="C:/SpinyLeaf/Media/Spl_LOGO_SOCIAL.png")


script_dir=os.path.dirname(os.path.abspath(__file__))
logo_path=os.path.join(script_dir,'Media','Spl_LOGO_SOCIAL.png')
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
        SpinyLeaf - Social
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
out_f_social = out_f / "Social_Dimension"
out_f_social.mkdir(exist_ok=True)


offices_occupants_per_area   = st.sidebar.select_slider('Commercial - Occupants per m²', options=[0.06, 0.07, 0.08, 0.09, 0.11, 0.12], value=0.08)
occupants_per_bedroom = st.sidebar.select_slider('Residential - Occupants per Bedroom', options=[1, 1.5, 2], value=1)

if "weights" not in st.session_state:
    st.session_state.weights = {"amount": 0.33, "dist": 0.33, "green": 0.34}
if "last_changed" not in st.session_state:
    st.session_state.last_changed = None

st.sidebar.write("Adjust Social Dimension Factor Weights:")
new_amount = slu.slider_with_auto_adjust("Amount", "amount", "amount")['amount']
new_dist   = slu.slider_with_auto_adjust("Distribution / Diversity", "dist", "dist")['dist']
new_green  = slu.slider_with_auto_adjust("Outdoors / Green", "green", "green")['green']


window = 'U 0.67 SHGC 0.77 Sgl LoE (e2-.2) Clr 3mm'
win = window_construction_by_identifier(window)
wall_type = 'GRC_Insul_Plasterboard'
wall_r = 6; roof_r = 6; ground_r = 6
operable = False


model_path = main_f / 'Rhino_Model' / 'shared_model.3dm'


if "hb_main_path" not in st.session_state:
    st.session_state.hb_main_path = None
if "hb_view_path" not in st.session_state:
    st.session_state.hb_view_path = None
if "base_df" not in st.session_state:
    st.session_state.base_df = None


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
                res_views = ri_call("prep_view_assets", {
                    "hb_view_path": str(view_hb_model_path),
                    "out_dir": str(out_f),
                    "dist": float(dist)
                })
                view_hb_model = Model.from_hbjson(Path(res_views["view_hb_path"]))
                st.session_state.view_hb_model_path = str(Path(res_views["view_hb_path"]))   
                st.session_state.hb_view_path = st.session_state.view_hb_model_path  

                hb_model.properties.radiance.sensor_grids = view_hb_model.properties.radiance.sensor_grids

                df = pd.DataFrame({
                    "room_ids": ids,
                    "floor_level": storeys,
                    "average_orient": av_orients,
                    "floor_area": room_areas,
                    "balcony_area": balcon_areas,
                    "ap_num_beds": n_beds
                })

            
                
                st.session_state.model_loaded = True
                st.session_state.model_done = True
                st.session_state.hb_model = hb_model
                st.session_state.view_hb_model = view_hb_model
                st.session_state.ids = ids
                st.session_state.storeys = storeys
                st.session_state.room_areas = room_areas
                st.session_state.balcon_areas = balcon_areas
                st.session_state.n_beds = n_beds
                st.session_state.hb_model_path = str(hb_model_path)
            
            

            out_areas = ri_call("get_social_outdoor_areas", {"src_3dm": str(model_path)})
            st.session_state.soc_out_area   = out_areas["soc_out_area"]
            st.session_state.soc_R_out_area = out_areas["soc_R_out_area"]
            st.session_state.soc_O_out_area = out_areas["soc_O_out_area"]

            st.session_state.base_df = df

if st.session_state.get("model_done"):
    hb_model = st.session_state.hb_model
    view_hb_model = st.session_state.view_hb_model
    ids = st.session_state.ids
    storeys = st.session_state.storeys
    room_areas = st.session_state.room_areas
    balcon_areas = st.session_state.balcon_areas
    n_beds = st.session_state.n_beds

    col1, col2 = st.columns(2)
    with col2:
        name = 'social_preview'
        with st.spinner("Generating 3D Model..."):
            v_height = 600
            svi.get_vtkjs(hb_model, out_f, name)
            svi.get_views(name, out_f, v_height)

    with col1:
        df = pd.DataFrame({
            "room_ids": ids,
            "floor_level": storeys,
            "floor_area": room_areas,
            "balcony_area": balcon_areas,
            "ap_num_beds": n_beds
        })

        st.write(df)
        win = window_construction_by_identifier(window)
        st.write('Windows U:', round(win.u_factor, 2))
        st.write('SHGC:', round(win.shgc, 2))
        st.write('Sol_Transmittance:', round(win.solar_transmittance, 2))
        st.write('Vis_Transmittance:', round(win.visible_transmittance, 2))
        st.write('Amount of Social Spaces weight:', round(new_amount, 2))
        st.write('Distrib/ Diversity weight:', round(new_dist, 2))
        st.write('Outdoors/ Green weight:', round(new_green, 2))

    st.success("Model successfully loaded!")


if st.session_state.base_df is not None:
    with st.form("Run Social"):
        go = st.form_submit_button("Run Social Dimension Study")
        if go:

            with st.spinner("Running Social Dimension Study..."):
                hb_view_path = st.session_state.view_hb_model_path
                view_hb_model = st.session_state.view_hb_model

                folders = {
                    "green_area_folder":        out_f_social / "Social_Green_area",
                    "green_occup_folder":       out_f_social / "Social_Green_Area_Occup",
                    "green_satisf_folder":      out_f_social / "Social_Green_Satisf",
                    "total_area_folder":        out_f_social / "Social_Total_Area",
                    "total_area_occup_folder":  out_f_social / "Social_Area_Occup",
                    "amount_satisf_folder":     out_f_social / "Social_Amount_Satisf",
                    "levels_available_folder":  out_f_social / "Social_Levels_Available",
                    "wdss_folder":              out_f_social / "Social_WDSS",
                    "dist_satisf_folder":       out_f_social / "Social_Distribution_Satisf",
                    "social_satisf_folder":     out_f_social / "_Social_Satisfaction",
                }
                for p in folders.values():
                    p.mkdir(parents=True, exist_ok=True)

                def save_scalar(values, folder):
                    ri_call("save_room_scalar_results", {
                        "view_hb_path": hb_view_path,             
                        "out_folder": str(folder),
                        "values": list(values),
                        "sensor_dist": 1.0
                    })

                df = st.session_state.base_df.copy()

               
                df['n_occup'] = df.apply(
                    lambda r: estimate_occupants(r, offices_occupants_per_area, occupants_per_bedroom),
                    axis=1
                )
                office_n_occup = df[df['room_ids'].str.lower().str.startswith('commerc')]['n_occup'].sum()
                resid_n_occup  = df[df['room_ids'].str.lower().str.startswith('resid')]['n_occup'].sum()

                
                s1  = df[df['room_ids'].str.startswith('SOCIAL_L1')]['floor_area'].sum()
                s2  = df[df['room_ids'].str.startswith('SOCIAL_L2')]['floor_area'].sum()
                s3  = df[df['room_ids'].str.startswith('SOCIAL_L3')]['floor_area'].sum()
                s4r = df[df['room_ids'].str.startswith('SOCIAL_L4_RESID')]['floor_area'].sum()
                s4o = df[df['room_ids'].str.startswith('SOCIAL_L4_COMMERC')]['floor_area'].sum()

                
                soc_out_area   = st.session_state.soc_out_area
                soc_R_out_area = st.session_state.soc_R_out_area
                soc_O_out_area = st.session_state.soc_O_out_area

                resid_social_areas = s1 + s2 + s3 + s4r + soc_out_area + soc_R_out_area
                office_social_areas = s1 + s2 + s3 + s4o + soc_out_area + soc_O_out_area

                resid_soc_green_occup  = (soc_out_area + soc_R_out_area) / resid_n_occup  if resid_n_occup  > 0 else 0.0
                office_soc_green_occup = (soc_out_area + soc_O_out_area) / office_n_occup if office_n_occup > 0 else 0.0

                
                df.loc[df['room_ids'].str.startswith('COMMERC'), 'social_green_area'] = (soc_out_area + soc_O_out_area)
                df.loc[df['room_ids'].str.startswith('RESID'),   'social_green_area'] = (soc_out_area + soc_R_out_area)
                df.loc[df['room_ids'].str.startswith('COMMERC'), 'social_green_area/occup'] = office_soc_green_occup
                df.loc[df['room_ids'].str.startswith('RESID'),   'social_green_area/occup'] = resid_soc_green_occup

                resid_social_occup  = resid_social_areas  / resid_n_occup  if resid_n_occup  > 0 else 0.0
                office_social_occup = office_social_areas / office_n_occup if office_n_occup > 0 else 0.0
                df.loc[df['room_ids'].str.startswith('COMMERC'), 'social_total_area'] = office_social_areas
                df.loc[df['room_ids'].str.startswith('RESID'),   'social_total_area'] = resid_social_areas
                df.loc[df['room_ids'].str.startswith('COMMERC'), 'social_total_area/occup'] = office_social_occup
                df.loc[df['room_ids'].str.startswith('RESID'),   'social_total_area/occup'] = resid_social_occup

                
                office_wdss, resid_wdss = compute_scalar_wdss(s1, s2, s3, s4r, s4o, resid_n_occup, office_n_occup, alpha=0.5)
                df.loc[df['room_ids'].str.startswith('RESID'),   'wdss_factor'] = resid_wdss
                df.loc[df['room_ids'].str.startswith('COMMERC'), 'wdss_factor'] = office_wdss

                shared_levels = set()
                if not df[df['room_ids'].str.startswith('SOCIAL_L1')].empty: shared_levels.add('SOCIAL_L1')
                if not df[df['room_ids'].str.startswith('SOCIAL_L2')].empty: shared_levels.add('SOCIAL_L2')
                if not df[df['room_ids'].str.startswith('SOCIAL_L3')].empty: shared_levels.add('SOCIAL_L3')

                resid_levels_count  = len(shared_levels) + (0 if df[df['room_ids'].str.startswith('SOCIAL_L4_RESID')].empty   else 1)
                office_levels_count = len(shared_levels) + (0 if df[df['room_ids'].str.startswith('SOCIAL_L4_COMMERC')].empty else 1)
                df.loc[df['room_ids'].str.startswith('RESID'),   'social_levels_available'] = resid_levels_count
                df.loc[df['room_ids'].str.startswith('COMMERC'), 'social_levels_available'] = office_levels_count


                v_height = 600

            with st.spinner("Running Social Amount study..."):
                samount_in   = df['social_total_area/occup'].tolist()
                samount_s    = sst.samount_satisf(samount_in)
                df['social_amount_satisf'] = samount_s

                save_scalar(df['social_total_area'].tolist(),          folders["total_area_folder"])
                save_scalar(df['social_total_area/occup'].tolist(),    folders["total_area_occup_folder"])
                save_scalar(df['social_amount_satisf'].tolist(),       folders["amount_satisf_folder"])

                study_names = ["Social_Total_Areas", "Social_Area_Occupants", "Social_Amount_Satisfaction"]
                study_dics = {
                    "Social_Total_Areas": folders["total_area_folder"],
                    "Social_Area_Occupants": folders["total_area_occup_folder"],
                    "Social_Amount_Satisfaction": folders["amount_satisf_folder"]}

                svi.view_study(view_hb_model, study_dics, study_names, study_names, v_height)

            with st.spinner("Running Social Distribution study..."):

                sdist_in   = df['wdss_factor'].tolist()
                sdist_s    = sst.sdist_satisf(sdist_in)
                df['social_distribution_satisf'] = sdist_s

                save_scalar(df['social_levels_available'].tolist(),    folders["levels_available_folder"])
                save_scalar(df['wdss_factor'].tolist(),                folders["wdss_folder"])
                save_scalar(df['social_distribution_satisf'].tolist(), folders["dist_satisf_folder"])

                study_names = ["Social_Levels_Available", "Weighted_Distribution_Social_Spaces", "Social_Distribution_Satisfaction"]
                study_dics = {
                    "Social_Levels_Available": folders["levels_available_folder"],
                    "Weighted_Distribution_Social_Spaces": folders["wdss_folder"],
                    "Social_Distribution_Satisfaction" : folders["dist_satisf_folder"]}

                svi.view_study(view_hb_model, study_dics, study_names, study_names, v_height)


            with st.spinner("Running Social Green Areas study..."):

                sgreen   = df['social_green_area/occup'].tolist()
                sgreen_s = sst.sgreen_satisf(sgreen)
                df['social_green_satisf'] = sgreen_s

                save_scalar(df['social_green_area'].tolist(),          folders["green_area_folder"])
                save_scalar(df['social_green_area/occup'].tolist(),    folders["green_occup_folder"])
                save_scalar(df['social_green_satisf'].tolist(),        folders["green_satisf_folder"])

                study_names = ["Social_Green_Areas", "Social_Green_Area_Occupants", "Social_green_satisfaction"]
                study_dics = {
                    "Social_Green_Areas": folders["green_area_folder"],
                    "Social_Green_Area_Occupants": folders["green_occup_folder"],
                    "Social_green_satisfaction" : folders["green_satisf_folder"]}

                svi.view_study(view_hb_model, study_dics, study_names, study_names, v_height)


            with st.spinner("Social Dimension Score..."):

                df['social_satisfaction'] = (
                    df['social_green_satisf'] * new_green +
                    df['social_amount_satisf'] * new_amount +
                    df['social_distribution_satisf'] * new_dist
                )
                

                mask = df['room_ids'].str.startswith(('SOCIAL', 'CORE'))
                satisfaction_cols = ['social_green_satisf','social_amount_satisf','social_distribution_satisf','social_satisfaction']
                df.loc[mask, satisfaction_cols] = None

                save_scalar(df['social_satisfaction'].tolist(),        folders["social_satisf_folder"])

                st.write(df[['room_ids', 'social_green_satisf','social_amount_satisf','social_distribution_satisf','social_satisfaction']])

                
                df_path = out_f_social / "Social.csv"
                df.to_csv(df_path, index=False)

                
                view_model = Model.from_hbjson(hb_view_path)
                svi.color_vtkjs_from_results(view_model, folders["social_satisf_folder"], study_name="Social_Satisfaction")

                v_height = 1200
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.success('Social Dimension Satisfaction')
                    svi.get_views("Social_Satisfaction", folders["social_satisf_folder"], v_height)

                with col2:
                    variables = ['social_green_satisf', 'social_amount_satisf', 'social_distribution_satisf']
                    title = 'Social Factors Satisfaction'
                    fig = svi.get_spider_matplotlib(df, variables, title, out_f_social)
                    fig2 = svi.plot_sbar_satisfaction(df, variables, title, out_f_social)
                    st.success("Response Distribution per Factor (%)")
                    with st.container(height=int(v_height * 3 / 5)):
                        st.pyplot(fig)
                    with st.container(height=int(v_height * 2/ 5)):
                        st.pyplot(fig2)

                with col3:
                    title = 'Social Dimension Satisfaction'
                    fig = svi.get_violing(df, 'social_satisfaction', title, out_f_social)
                    fig2 = svi.plot_satisf_bar(df, 'social_satisfaction', title, out_f_social)
                    st.success("Satisfaction Score Distribution")
                    with st.container(height=int(v_height * 3 / 5)):
                        st.pyplot(fig)
                    with st.container(height=int(v_height * 2 / 5)):
                        st.pyplot(fig2)

                st.success("Social Dimension study complete.")
