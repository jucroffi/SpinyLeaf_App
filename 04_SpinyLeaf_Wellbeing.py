
import os, sys, json, subprocess
from pathlib import Path

import streamlit as st
from PIL import Image
import pandas as pd
import numpy as np

from honeybee.model import Model
from honeybee_energy.lib.constructions import window_construction_by_identifier

import Functions.App_Visualisation_lib as svi
import Functions.App_Utils as slu


def ri_call(op, payload, timeout=600):
    worker = Path(__file__).with_name("00_RI_worker.py")
    proc = subprocess.Popen([sys.executable, str(worker)],
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True)
    out, err = proc.communicate(json.dumps({"op": op, "payload": payload}), timeout=timeout)
    if proc.returncode != 0:
        raise RuntimeError(err or out or "RI worker failed")
    res = json.loads(out)
    if not res.get("ok", True):
        raise RuntimeError(res.get("error", f"RI worker error on {op}"))
    return res



# Page config

dim = 'Wellbeing'

st.set_page_config(page_title= f'SpinyLeaf - {dim}', layout='wide', page_icon="C:/SpinyLeaf/Media/Spl_LOGO_WELLBEING.png")


script_dir=os.path.dirname(os.path.abspath(__file__))
logo_path=os.path.join(script_dir,'Media','Spl_LOGO_WELLBEING.png')
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
        SpinyLeaf - Wellbeing
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
out_f_comfort = out_f / "Comfort_Dimension"
out_f_delight = out_f / "Delight_Dimension"
out_f_social  = out_f / "Social_Dimension"
wb_folder     = out_f / "Wellbeing_Design"

comfort_csv = out_f_comfort / "Comfort.csv"
delight_csv = out_f_delight / "Delight.csv"
social_csv  = out_f_social / "Social.csv"

comfort_satisf_folder = out_f_comfort / "_Comfort_Satisfaction"
delight_satisf_folder = out_f_delight / "_Delight_Satisfaction"
social_satisf_folder  = out_f_social  / "_Social_Satisfaction"

wb_folder.mkdir(parents=True, exist_ok=True)


model_path = main_f / 'Rhino_Model' / 'shared_model.3dm'


generate_report = st.sidebar.toggle("Generate Wellbeing Report", value=False)
st.sidebar.write("Yes" if generate_report else "No")

if "openai_key" not in st.session_state:
    st.session_state.openai_key = ""

st.sidebar.header("🔑 OpenAI API Key ")
input_key = st.sidebar.text_input(
    label="Enter your OpenAI API key",
    type="password",
    placeholder="sk-...",
    key="api_key_input"
)
if st.sidebar.button("Save Key"):
    if input_key.startswith("sk-"):
        st.session_state.openai_key = input_key
        st.sidebar.success("API key saved securely.")
    else:
        st.sidebar.error("Invalid API key format.")


window = 'U 0.67 SHGC 0.77 Sgl LoE (e2-.2) Clr 3mm'
wall_type = 'GRC_Insul_Plasterboard'
wall_r = 6; roof_r = 6; ground_r = 6
operable = False
offices_occupants_per_area = 0.08
occupants_per_bedroom = 1.0


if "view_hb_model_path" not in st.session_state:
    st.session_state.view_hb_model_path = None
if "hb_main_path" not in st.session_state:
    st.session_state.hb_main_path = None
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
        name = 'wellbeing_preview'
        with st.spinner("Generating 3D Model..."):
            v_height = 800
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

    st.success("Model successfully loaded!")
    st.session_state.model_done = True


if "run_wellbeing" not in st.session_state:
    st.session_state.run_wellbeing = False
    st.session_state.wellbeing_done = False

if st.session_state.model_done:

    with st.form("Run Wellbeing"):
        go = st.form_submit_button("Run Wellbeing Fostered by Design Study")
        if go:

            with st.spinner("Running Wellbeing Fostered by Design Study...", show_time=True):

                missing = []
                if not comfort_csv.exists(): missing.append("Comfort")
                if not delight_csv.exists(): missing.append("Delight")
                if not social_csv.exists():  missing.append("Social")

                if missing:
                    st.error(f"❌ The following studies are missing: {', '.join(missing)}")
                else:
                    
                    comfort_df = pd.read_csv(comfort_csv)
                    delight_df = pd.read_csv(delight_csv)
                    social_df  = pd.read_csv(social_csv)

                    
                    wellbeing_df = pd.DataFrame()
                    wellbeing_df['room_ids'] = social_df['room_ids']
                    wellbeing_df['floor_area'] = social_df['floor_area']
                    wellbeing_df['social_satisfaction'] = social_df['social_satisfaction']
                    wellbeing_df['delight_satisfaction'] = delight_df['delight_satisfaction']
                    wellbeing_df['comfort_satisfaction'] = comfort_df['comfort_satisfaction']

                    wellbeing_df['wellbeing_satisfaction'] = (
                        wellbeing_df['comfort_satisfaction'] +
                        wellbeing_df['social_satisfaction'] +
                        wellbeing_df['delight_satisfaction']
                    )

                    
                    mask = wellbeing_df['room_ids'].str.startswith(('SOCIAL', 'CORE'))
                    cols = ['social_satisfaction', 'delight_satisfaction',
                            'comfort_satisfaction', 'wellbeing_satisfaction']
                    wellbeing_df.loc[mask, cols] = None

                    wb_csv = out_f / "Wellbeing.csv"
                    wellbeing_df.to_csv(wb_csv, index=False)
                    st.dataframe(wellbeing_df)

                    
                    if not st.session_state.view_hb_model_path:
                        st.error("HB view model path is missing. Click 'Get/Update Model Assets' first.")
                    else:
                        values = wellbeing_df['wellbeing_satisfaction'].tolist()
                        ri_call("save_room_scalar_results", {
                            "view_hb_path": st.session_state.view_hb_model_path,
                            "out_folder": str(wb_folder),
                            "values": values,
                            "sensor_dist": 1.0
                        })

                        
                        view_model = Model.from_hbjson(st.session_state.view_hb_model_path)
                        v_height = 1200

                        study_names = ["Comfort_Satisfaction", "Delight_Satisfaction", "Social_Satisfaction"]
                        study_dics = {
                            "Comfort_Satisfaction": comfort_satisf_folder,
                            "Delight_Satisfaction": delight_satisf_folder,
                            "Social_Satisfaction": social_satisf_folder
                        }
                        svi.view_study(view_model, study_dics, study_names, study_names, v_height)

                        
                        svi.color_vtkjs_from_results(view_model, wb_folder, study_name="Wellbeing_Fostered_by_Design")

                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.success('Wellbeing Fostered by Design')
                            svi.get_views("Wellbeing_Fostered_by_Design", wb_folder, v_height)

                        with col2:
                            variables = ['comfort_satisfaction', 'social_satisfaction', 'delight_satisfaction']
                            title = 'Wellbeing Factors Satisfaction'
                            fig = svi.get_spider_matplotlib(wellbeing_df, variables, title, out_f)
                            fig2 = svi.plot_sbar_satisfaction(wellbeing_df, variables, title, out_f)
                            st.success("Response Distribution per Factor (%)")
                            with st.container(height=int(v_height * 3 / 5)):
                                st.pyplot(fig)
                            with st.container(height=int(v_height * 2/ 5)):
                                st.pyplot(fig2)

                        with col3:
                            title = 'Wellbeing Fostered by Design'
                            fig = svi.get_violin_wellbeing(wellbeing_df, 'wellbeing_satisfaction', title, out_f)
                            fig2 = svi.plot_wellbeing_bar(wellbeing_df, 'wellbeing_satisfaction', title, out_f)
                            st.success("Satisfaction Score Distribution")
                            with st.container(height=int(v_height * 3 / 5)):
                                st.pyplot(fig)
                            with st.container(height=int(v_height * 2 / 5)):
                                st.pyplot(fig2)

                        st.success("Wellbeing aggregation complete.")

                        
                        if generate_report:
                            with st.spinner("Generating Report..."):
                                try:
                                    if input_key:
                                        result = subprocess.run(
                                            [sys.executable, "C:/SpinyLeaf/Functions/App_create_report_OpenAI.py", input_key],
                                            check=True, capture_output=True, text=True
                                        )
                                    else:
                                        result = subprocess.run(
                                            [sys.executable, "C:/SpinyLeaf/Functions/App_create_report_Llama.py"],
                                            check=True, capture_output=True, text=True
                                        )
                                    st.success("📄 Report generated successfully!")
                                    st.code(result.stdout)
                                except subprocess.CalledProcessError as e:
                                    st.error("❌ Report generation failed.")
                                    st.code(e.stderr)
