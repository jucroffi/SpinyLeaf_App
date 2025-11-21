
import streamlit as st
import os, sys, json, subprocess, shutil
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image

from honeybee.model import Model
from honeybee_energy.lib.constructions import window_construction_by_identifier

import Functions.App_Create_Model_Simulations_00 as spi
import Functions.App_Visualisation_lib as svi
import Functions.App_Satisfaction_T as sst
import Functions.App_Utils as slu



# Page config

dim = 'Delight'

st.set_page_config(page_title= f'SpinyLeaf - {dim}', layout='wide', page_icon="C:/SpinyLeaf/Media/Spl_LOGO_DELIGHT.png")


script_dir=os.path.dirname(os.path.abspath(__file__))
logo_path=os.path.join(script_dir,'Media','Spl_LOGO_DELIGHT.png')
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
        SpinyLeaf - Delight
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


USAGE_DICT = {
    0: "COMMERC", 1: "CONTEXT", 2: "RESID",
    3: "GREEN", 4: "BALCONIES", 5: "CORE",
    6: "SOCIAL_L1", 7: "SOCIAL_L2", 8: "SOCIAL_L3",
    9: "SOCIAL_L4_RESID", 10: "SOCIAL_L4_COMMERC",
    11: "SOCIAL_OUTDOOR_ALL", 12: "SOCIAL_OUTDOOR_RESID", 13: "SOCIAL_OUTDOOR_COMMERC"
}


window = 'U 0.67 SHGC 0.77 Sgl LoE (e2-.2) Clr 3mm'
wall_type = 'GRC_Insul_Plasterboard'
wall_r = 6
roof_r = 6
ground_r = 6
operable = False


offices_occupants_per_area = st.sidebar.select_slider('Commercial - Occupants per m2',
                                              options=[0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12],
                                              value=0.08)
occupants_per_bedroom = st.sidebar.select_slider('Residential - Occupants per Bedroom',
                                                 options=[1, 1.5, 2], value=1)


if "weights" not in st.session_state:
    st.session_state.weights = {"views": 0.33, "green": 0.33, "size": 0.34}
if "last_changed" not in st.session_state:
    st.session_state.last_changed = None

st.sidebar.write("Adjust Delight Dimension Factors Weights:")
new_views = slu.slider_with_auto_adjust("Views", "views", "views")['views']
new_green = slu.slider_with_auto_adjust("Access to Green", "green", "green")['green']
new_size = slu.slider_with_auto_adjust("Space Size", "size", "size")['size']

model_path = main_f / 'Rhino_Model' / 'shared_model.3dm'


def ri_call(op, payload, timeout=600):
    worker = Path(__file__).with_name("00_RI_worker.py")
    proc = subprocess.Popen([sys.executable, str(worker)],
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True)
    out, err = proc.communicate(json.dumps({"op": op, "payload": payload}), timeout=timeout)
    if proc.returncode != 0:
        raise RuntimeError(err or out or "Delight worker failed")
    return json.loads(out)


if "model_loaded" not in st.session_state:
    st.session_state.model_loaded = False
if "model_done" not in st.session_state:
    st.session_state.model_done = False


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
                st.session_state.view_hb_model = view_hb_model                          

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
                st.session_state.view_hb_model_path = str(view_hb_model_path)

            

            
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
        name = 'delight_preview'
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
        st.write('Views weight:', round(new_views, 2))
        st.write('Access to Green weight:', round(new_green, 2))
        st.write('Space Size weight:', round(new_size, 2))

    
    st.success("Model successfully loaded!")


if "run_delight" not in st.session_state:
    st.session_state.run_delight = False
    st.session_state.delight_done = False

if st.session_state.get("model_done"):
    with st.form("Delight Dimension Study Form"):
        submitted = st.form_submit_button("Run Delight Dimension Study")
        if submitted:
            out_f_delight = out_f / "Delight_Dimension"
            hv_res_folder = out_f_delight / "Hor_Views"
            hv_mean_folder = out_f_delight / "Hor_Views_mean"
            hv_satisf_folder = out_f_delight / "Hor_Views_satisf"

            gv_res_folder = out_f_delight / "Green_Views"
            gv_mean_folder = out_f_delight / "Green_Views_mean"
            gv_satisf_folder = out_f_delight / "Green_Views_satisf"

            sv_res_folder = out_f_delight / "Sky_Views"
            sv_mean_folder = out_f_delight / "Sky_Views_mean"
            sv_satisf_folder = out_f_delight / "Sky_Views_satisf"

            b_area_folder = out_f_delight / "Balcony_Area"
            b_res_folder = out_f_delight / "Balcony_Percentage"
            b_satisf_folder = out_f_delight / "Balcony_Satisf"

            as_area_folder = out_f_delight / "Apartment_Area"
            as_res_folder = out_f_delight / "Apartment_Occupancy"
            as_satisf_folder = out_f_delight / "Apartment_Size_Satisf"

            delight_df = pd.DataFrame({
                "room_ids": st.session_state.ids,
                "floor_level" : st.session_state.storeys,
                "floor_area": st.session_state.room_areas,
                "balcony_area": st.session_state.balcon_areas,
                "ap_num_beds": st.session_state.n_beds
            })


            v_height = 600
            
            with st.spinner("Running horizontal views study..."):
                res_h = ri_call("run_horizontal_views", {
                    "src_3dm": str(model_path),
                    "hb_model_path": st.session_state.hb_model_path,
                    "view_hb_path": st.session_state.view_hb_model_path,
                    "hv_res_folder": str(hv_res_folder),
                    "hv_mean_folder": str(hv_mean_folder),
                    "hv_satisf_folder": str(hv_satisf_folder),
                    "sensor_dist": 1.0
                })
                delight_df["hor_views_mean"] = res_h["hv_mean_room"]
                delight_df["hor_views_satisf"] = res_h["hv_satisf_room"]

                study_names = ["Horizontal_Views", "Horizontal_Mean", "Outdoors_Views_Satisfaction"]
                study_dics = {
                    "Horizontal_Views": hv_res_folder,
                    "Horizontal_Mean": hv_mean_folder,
                    "Outdoors_Views_Satisfaction": hv_satisf_folder
                }
                view_hb_model = st.session_state.view_hb_model
                svi.view_study(view_hb_model, study_dics, study_names, study_names, v_height)

            
            with st.spinner("Running views of green study..."):
                res_g = ri_call("run_green_views", {
                    "src_3dm": str(model_path),
                    "hb_model_path": st.session_state.hb_model_path,
                    "view_hb_path": st.session_state.view_hb_model_path,
                    "gv_res_folder": str(gv_res_folder),
                    "gv_mean_folder": str(gv_mean_folder),
                    "gv_satisf_folder": str(gv_satisf_folder),
                    "sensor_dist": 1.0,
                    "green_sampling": 5
                })
                delight_df["green_views_mean"] = res_g["gv_mean_room"]
                delight_df["green_views_satisf"] = res_g["gv_satisf_room"]

                study_names = ["Green_Views", "Green_Mean", "Green_Views_Satisfaction"]
                study_dics = {
                    "Green_Views": gv_res_folder,
                    "Green_Mean": gv_mean_folder,
                    "Green_Views_Satisfaction": gv_satisf_folder
                }
                svi.view_study(view_hb_model, study_dics, study_names, study_names, v_height)

            
            with st.spinner("Running sky views study..."):
                res_s = ri_call("run_sky_views", {
                    "src_3dm": str(model_path),
                    "hb_model_path": st.session_state.hb_model_path,
                    "view_hb_path": st.session_state.view_hb_model_path,
                    "sv_res_folder": str(sv_res_folder),
                    "sv_mean_folder": str(sv_mean_folder),
                    "sv_satisf_folder": str(sv_satisf_folder),
                    "sensor_dist": 1.0
                })
                delight_df["sky_views_mean"] = res_s["sv_mean_room"]
                delight_df["sky_views_satisf"] = res_s["sv_satisf_room"]

                study_names = ["Sky_Views", "Sky_Mean", "Sky_Views_Satisfaction"]
                study_dics = {
                    "Sky_Views": sv_res_folder,
                    "Sky_Mean": sv_mean_folder,
                    "Sky_Views_Satisfaction": sv_satisf_folder
                }
                svi.view_study(view_hb_model, study_dics, study_names, study_names, v_height)

            
            with st.spinner("Running balcony and size studies..."):
                res_b = ri_call("run_balcony_size_metrics", {
                    "view_hb_path": st.session_state.view_hb_model_path,
                    "as_area_folder": str(as_area_folder),
                    "as_res_folder": str(as_res_folder),
                    "as_satisf_folder": str(as_satisf_folder),
                    "b_area_folder": str(b_area_folder),
                    "b_res_folder": str(b_res_folder),
                    "b_satisf_folder": str(b_satisf_folder),
                    "room_areas": st.session_state.room_areas,
                    "balcon_areas": st.session_state.balcon_areas,
                    "ids": st.session_state.ids,
                    "n_beds": st.session_state.n_beds,
                    "occupants_per_bedroom": float(occupants_per_bedroom),
                    "occupants_per_area": float(offices_occupants_per_area),
                    "sensor_dist": 1.0
                })
                delight_df["balcony_percentage"] = res_b["balcony_percentage"]
                delight_df["balcony_satisf"] = res_b["balcony_satisf"]
                delight_df["occupancy"] = res_b["occupancy"]
                delight_df["space_size_satisf"] = res_b["space_size_satisf"]

                study_names = ["Balcony_Areas", "Balcony_Percentage", "Access_to_Green_Satisfaction"]
                study_dics = {
                    "Balcony_Areas": b_area_folder,
                    "Balcony_Percentage": b_res_folder,
                    "Access_to_Green_Satisfaction": b_satisf_folder
                }
                svi.view_study(view_hb_model, study_dics, study_names, study_names, 600)

                study_names = ["Areas", "Occupancy_Rate", "Space_Size_Satisfaction"]
                study_dics = {
                    "Areas": as_area_folder,
                    "Occupancy_Rate": as_res_folder,
                    "Space_Size_Satisfaction": as_satisf_folder
                }
                svi.view_study(view_hb_model, study_dics, study_names, study_names, 600)

            
            with st.spinner("Delight Dimension Score..."):
                delight_df['views_overall_satisf'] = delight_df[['hor_views_satisf',
                                                                 'green_views_satisf',
                                                                 'sky_views_satisf']].mean(axis=1)
                delight_df['delight_satisfaction'] = (delight_df['views_overall_satisf'] * new_views +
                                                      delight_df['balcony_satisf'] * new_green +
                                                      delight_df['space_size_satisf'] * new_size)


                
                mask = delight_df['room_ids'].str.startswith(('SOCIAL', 'CORE'))
                to_null = ['hor_views_satisf', 'green_views_satisf', 'sky_views_satisf',
                           'balcony_satisf', 'space_size_satisf', 'delight_satisfaction']
                delight_df.loc[mask, to_null] = None

                st.write(delight_df[['room_ids', 'hor_views_satisf', 'green_views_satisf',
                                     'sky_views_satisf', 'balcony_satisf', 'space_size_satisf',
                                     'delight_satisfaction']])

                df_folder = (out_f / "Delight_Dimension" / "Delight.csv")
                delight_df.to_csv(df_folder)

                col1, col2, col3 = st.columns(3)
                
                _delight_satisf_folder = (out_f / "Delight_Dimension" / "_Delight_Satisfaction")

                weighted_satisf = delight_df['delight_satisfaction'].tolist()

                ri_call("save_room_scalar_results", {
                    "view_hb_path": st.session_state.view_hb_model_path, 
                    "out_folder": str(_delight_satisf_folder),
                    "values": weighted_satisf,
                    "sensor_dist": 1.0
                })

                
                svi.color_vtkjs_from_results(st.session_state.view_hb_model,
                                            _delight_satisf_folder,
                                            study_name="Delight_Satisfaction")

                v_height = 1200

                with col1:
                    st.success('Delight Dimension Satisfaction')
                    svi.get_views("Delight_Satisfaction", _delight_satisf_folder, v_height)

                with col2:
                    variables = ['hor_views_satisf', 'green_views_satisf', 'sky_views_satisf', 'balcony_satisf', 'space_size_satisf']
                    title = 'Delight Factors Satisfaction'
                    fig = svi.get_spider_matplotlib(delight_df, variables, title, out_f_delight)
                    fig2 = svi.plot_sbar_satisfaction(delight_df, variables, title, out_f_delight)

                    st.success("Response Distribution per Factor (%)")
                    with st.container(height=int(v_height * 3 / 5)):
                        st.pyplot(fig)
                    with st.container(height=int(v_height * 2/ 5)):
                        st.pyplot(fig2)

                with col3:
                    title = 'Delight Dimension Satisfaction'
                    fig = svi.get_violing(delight_df, 'delight_satisfaction', title, out_f_delight)
                    fig2 = svi.plot_satisf_bar(delight_df, 'delight_satisfaction', title, out_f_delight)

                    
                    st.success("Satisfaction Score Distribution")
                    with st.container(height=int(v_height * 3 / 5)):
                        st.pyplot(fig)
                    with st.container(height=int(v_height * 2/ 5)):
                        st.pyplot(fig2)

            st.session_state.delight_done = True
