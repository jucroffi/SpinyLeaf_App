import streamlit as st
import sys

import numpy as np
import os

import matplotlib.pyplot as plt

import matplotlib
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Polygon

import seaborn as sns




import pandas as pd
import numpy as np

import json
import pathlib


from ladybug.color import Color
import honeybee_vtk
from honeybee_vtk.scene import Scene
from honeybee_vtk.camera import Camera
from honeybee_vtk.model import DisplayMode
from honeybee_vtk.model import (HBModel,
                                Model as VTKModel,
                                SensorGridOptions,
                                DisplayMode)
from honeybee_vtk.legend_parameter import ColorSets

from honeybee_radiance.view import View

from streamlit_vtkjs import st_vtkjs



def get_config(study_name, results_folder, unit, d_range, cs):

    cfg = {
        "data": [
            {
                "identifier": study_name,
                "object_type": "grid",
                "unit": unit[study_name],
                "path": results_folder.as_posix(),
                "hide": False,
                "legend_parameters": {
                        "hide_legend": False,
                        "min": d_range[study_name][0],
                        "max": d_range[study_name][1],
                        "color_set": cs[study_name],
                        "label_parameters": {
                            "color": [0, 0, 0],
                            "size": 0,
                            "bold": True
                        }
                }
            }
        ]
    }

    
    config_file = results_folder.joinpath("config.json")
    with open(config_file, "w") as f:
        json.dump(cfg, f, indent=2)

    return config_file




def color_vtkjs_from_results(model, results_folder, study_name):
    
    vtk_model = VTKModel(model, SensorGridOptions.Mesh)

    cs = {'Horizontal_Views': 'view_study',
          'Horizontal_Mean' : 'heat_sensation',
          'Outdoors_Views_Satisfaction' : 'shade_benefit_harm',
          'Sky_Views': 'view_study',
          'Sky_Mean' : 'cold_sensation',
          'Sky_Views_Satisfaction' : 'shade_benefit_harm',
          'Green_Views': 'view_study',
          'Green_Mean' : 'peak_load_balance',
          'Green_Views_Satisfaction' : 'shade_benefit_harm',
          "Balcony_Areas": 'shadow_study',
          "Balcony_Percentage": 'shadow_study',
          "Access_to_Green_Satisfaction": 'shade_benefit_harm',
          "Areas": 'annual_comfort',
          "Occupancy_Rate": 'energy_balance',
          "Space_Size_Satisfaction" : 'shade_benefit_harm',
          "Extreme_Hot_Week_Temp": "nuanced",
          "Ext_Hot_Thermal_Sensation": "thermal_comfort",
          "Ext_Hot_Thermal_Satisfaction" : 'shade_benefit_harm',
          "Extreme_Cold_Week_Temp": 'nuanced',
          "Ext_Cold_Thermal_Sensation": 'thermal_comfort',
          "Ext_Cold_Thermal_Satisfaction": 'shade_benefit_harm',
          "Daylight_Autonomy": "ecotect",
          "DA_mean": 'ecotect',
          "Daylight_Satisfaction": 'shade_benefit_harm',
          "Useful_Daylight_Illuminance": "ecotect",
          "UDI_mean": "ecotect",
          "Glare_Autonomy": "glare_study",
          "GA_mean": "glare_study",
          "CO2_Levels": 'black_to_white',
          "Relative_Humidity": 'cloud_cover',
          "Air_Quality_Satisfaction": 'shade_benefit_harm',
          "Delight_Satisfaction": 'benefit_harm',
          "Sound_Levels": 'blue_green_red',
          "Sound_Level_Satisfaction": 'shade_benefit_harm',
          "Comfort_Satisfaction": 'benefit_harm',
          "Social_Green_Areas": 'peak_load_balance',
          "Social_Green_Area_Occupants": 'peak_load_balance',
          "Social_green_satisfaction":'shade_benefit_harm',
          "Social_Total_Areas": 'shadow_study',
          "Social_Area_Occupants": 'shadow_study',
          "Social_Amount_Satisfaction": 'shade_benefit_harm',
          "Social_Levels_Available": 'annual_comfort',
          "Weighted_Distribution_Social_Spaces": 'annual_comfort',
          "Social_Distribution_Satisfaction": 'shade_benefit_harm',
          "Social_Satisfaction": 'benefit_harm',
          "Wellbeing_Fostered_by_Design": 'benefit_harm'}

    
    d_range = {'Horizontal_Views': [0, 50],
               'Horizontal_Mean' : [0, 30],
               'Outdoors_Views_Satisfaction' : [0, 2],
               'Sky_Views': [0, 50],
               'Sky_Mean' : [0, 30],
               'Sky_Views_Satisfaction' : [0, 2],
               'Green_Views': [0, 50],
               'Green_Mean' : [0, 20],
               'Green_Views_Satisfaction' : [0, 2],
                "Balcony_Areas": [0,15],
                "Balcony_Percentage": [0,15],
                "Access_to_Green_Satisfaction":[0, 2],
                "Areas": [30,120],
                "Occupancy_Rate": [0.016,0.026],
                "Space_Size_Satisfaction":[0, 2],
                "Extreme_Hot_Week_Temp": [24, 30],
                "Ext_Hot_Thermal_Sensation": [-2,2],
                "Ext_Hot_Thermal_Satisfaction" : [0, 2],
                "Extreme_Cold_Week_Temp": [15, 24],
                "Ext_Cold_Thermal_Sensation": [-2,2],
                "Ext_Cold_Thermal_Satisfaction": [0, 2],
                "Daylight_Autonomy": [0,100],
                "DA_mean": [40,80],
                "Daylight_Satisfaction": [0, 1],
                "Useful_Daylight_Illuminance": [0,100],
                "UDI_mean":[40,80],
                "Glare_Autonomy": [0,100],
                "GA_mean": [40,80],
                "CO2_Levels": [400,1000],
                "Relative_Humidity": [40,70],
                "Air_Quality_Satisfaction":[0, 2],
                "Delight_Satisfaction":[0,2],
                "Sound_Levels": [30,50],
                "Sound_Level_Satisfaction": [0,1],
                "Comfort_Satisfaction":[0,2],
                "Social_Green_Areas": [100,200],
                "Social_Green_Area_Occupants": [0,20],
                "Social_green_satisfaction":[0,2],
                "Social_Total_Areas": [50,500],
                "Social_Area_Occupants": [0,20],
                "Social_Amount_Satisfaction":[0,2],
                "Social_Levels_Available": [0,4],
                "Weighted_Distribution_Social_Spaces": [0,1],
                "Social_Distribution_Satisfaction":[0,2],
                "Social_Satisfaction": [0,2],
                "Wellbeing_Fostered_by_Design": [0,6]}
    
    
    unit = {'Horizontal_Views': "%",
            'Horizontal_Mean' : "% mean",
            'Outdoors_Views_Satisfaction' : "Satisfaction", 
            'Sky_Views': "%",
            'Sky_Mean' : "% mean",
            'Sky_Views_Satisfaction' : "Satisfaction", 
            'Green_Views': "%",
            'Green_Mean' : "% mean",
            'Green_Views_Satisfaction' : "Satisfaction",
            "Balcony_Areas": 'Area m2',
            "Balcony_Percentage": '%',
            "Access_to_Green_Satisfaction": "Satisfaction",
            "Areas": 'Area m2',
            "Occupancy_Rate": 'ppl/m2',
            "Space_Size_Satisfaction" :"Satisfaction",
            "Extreme_Hot_Week_Temp": "C",
            "Ext_Hot_Thermal_Sensation": "Cold / Warm",
            "Ext_Hot_Thermal_Satisfaction" : 'Satisfaction',
            "Extreme_Cold_Week_Temp": "C",
            "Ext_Cold_Thermal_Sensation": "Cold / Warm",
            "Ext_Cold_Thermal_Satisfaction": 'Satisfaction',
            "Daylight_Autonomy": "DA %",
            "DA_mean": "% mean",
            "Daylight_Satisfaction": 'Satisfaction',
            "Useful_Daylight_Illuminance": "UDI %",
            "UDI_mean": "% mean",
            "Glare_Autonomy": "GA %",
            "GA_mean": "% mean",
            "CO2_Levels": 'ppm',
            "Relative_Humidity": '%',
            "Air_Quality_Satisfaction":'Satisfaction',
            "Delight_Satisfaction": 'Satisfaction',
            "Sound_Levels": 'dB',
            "Sound_Level_Satisfaction":'Satisfaction',
            "Comfort_Satisfaction":'Satisfaction',
            "Social_Green_Areas": 'Area m2',
            "Social_Green_Area_Occupants": 'm2/occupant',
            "Social_green_satisfaction":'Satisfaction',
            "Social_Total_Areas": 'm2/occupant',
            "Social_Area_Occupants": 'm2/occupant',
            "Social_Amount_Satisfaction":'Satisfaction',
            "Social_Levels_Available": 'Number of Social Levels Available',
            "Weighted_Distribution_Social_Spaces": 'wdss',
            "Social_Distribution_Satisfaction": 'Satisfaction',
            "Social_Satisfaction": 'Satisfaction',
            "Wellbeing_Fostered_by_Design": 'Satisfaction'}
    
    
    vtk_model.sensor_grids.display_mode = DisplayMode.SurfaceWithEdges
    vtk_model.shades.display_mode = DisplayMode.Surface
    vtk_model.shades.color = Color(130, 130, 130, 200)
    vtk_model.walls.display_mode = DisplayMode.Wireframe
    vtk_model.walls.color = Color(0, 0, 0, 255)
    vtk_model.floors.display_mode = DisplayMode.Wireframe
    vtk_model.floors.color = Color(0, 0, 0, 0)
    vtk_model.roof_ceilings.display_mode = DisplayMode.Wireframe
    vtk_model.roof_ceilings.color = Color(0, 0, 0, 0)


    config_file = get_config(study_name, results_folder, unit, d_range, cs)

    vtk_model.to_vtkjs(folder=results_folder,name=study_name, 
                       config=config_file.as_posix(),
                       model_display_mode=DisplayMode.Wireframe)
    

    
def get_vtkjs(model, output_folder, name):

    view = View(identifier='view_test', position=(0, 0, 100), direction=(5,20,2), up_vector=None, type='v', h_size=40, v_size=40)
    model.properties.radiance.add_view(view)
    vtk_model = VTKModel(model, SensorGridOptions.Sensors)
    vtk_model.shades.display_mode = DisplayMode.SurfaceWithEdges
    vtk_model.shades.color = Color(130, 130, 130, 180)
    vtk_model.walls.display_mode = DisplayMode.Surface
    vtk_model.walls.color = Color(250, 250, 250, 255)
    vtk_model.roof_ceilings.display_mode = DisplayMode.Surface
    vtk_model.roof_ceilings.color = Color(109, 148, 105, 255)


    vtk_model.to_vtkjs(folder=output_folder, name = name,
                       model_display_mode=DisplayMode.Surface)
    model.to_hbjson(folder=output_folder, name = name)


def get_views(name, output_folder, height):

    key = f'{name}_{0}'
    read_folder = pathlib.Path('data', output_folder, f'{name}.vtkjs').read_bytes()
    height = f'{height}px'
    views = st_vtkjs(content=read_folder, key=key,style={'height': height}, subscribe=False)
    return views


def view_study(view_hb_model, study_dics, study_names, labels, v_height):

    for n, column in enumerate(st.columns(len(study_names))):
        color_vtkjs_from_results(view_hb_model, study_dics[study_names[n]], study_name=study_names[n])
        with column:
            st.success(labels[n])
            get_views(study_names[n], study_dics[study_names[n]], v_height)



def calc_response_percentages_by_area(df, variables, target_level):
    """
    Calculates the percentage of floor area associated with a satisfaction level.
    
    Levels:
        0: Dissatisfied (0 to 0.66)
        1: Neutral (0.66 to 1.33)
        2: Satisfied (> 1.33)
    """
    total_area = df['floor_area'].sum()
    percentages = []

    for var in variables:
        valid = df[df[var].notna()]
        
        if target_level == 0:
            condition = (valid[var] >= 0) & (valid[var] <= 0.66)
        elif target_level == 1:
            condition = (valid[var] > 0.66) & (valid[var] <= 1.33)
        elif target_level == 2:
            condition = (valid[var] > 1.33)
        else:
            raise ValueError("target_level must be 0 (dissatisfied), 1 (neutral), or 2 (satisfied)")

        target_area = valid[condition]['floor_area'].sum()
        percent = 100 * target_area / total_area if total_area > 0 else 0
        percentages.append(percent)

    return percentages



def get_spider_matplotlib(df, variables, title, path_images):
    label_dict = {
        't_sens': 'Thermal\nSensation', 
        't_satisf': 'Thermal\nSatisf', 
        'daylight': 'Daylight\nPercep', 
        'dl_satisf': 'Daylight\nSatisf', 
        'sound': 'Sound\nPercep',
        'sound_satisf': 'Acoustic\nSatisf', 
        'air_quali': 'Air Quality\nSatisf',
        'hor_views_satisf': 'Views\nOutdoors', 
        'green_views_satisf': 'Views\nNature', 
        'sky_views_satisf': 'Views\nSky', 
        'balcony_satisf': 'Access to Green\n(balconies)', 
        'space_size_satisf': 'Space Size\nSatisf',
        'social_green_satisf': 'Social\nOutdoors/Green', 
        'social_amount_satisf': 'Amount of\nSocial Spaces', 
        'social_distribution_satisf': 'Distribution of\nSocial Spaces',
        'comfort_satisfaction': 'Comfort\nDimension', 
        'social_satisfaction': 'Social\nDimension', 
        'delight_satisfaction': 'Delight\nDimension'
    }

    
    for col in ['dl_satisf', 'sound_satisf']:
        if col in df.columns:
            df[col] = df[col].replace({1: 2})  

    satisfied = calc_response_percentages_by_area(df, variables, 2)
    neutral = calc_response_percentages_by_area(df, variables, 1)
    dissatisfied = calc_response_percentages_by_area(df, variables, 0)

    satisfied += [satisfied[0]]
    neutral += [neutral[0]]
    dissatisfied += [dissatisfied[0]]

    angles = np.linspace(0, 2 * np.pi, len(variables), endpoint=False).tolist()
    angles += angles[:1]

    labels = [label_dict.get(var, var) for var in variables]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    

    ax.plot(angles, satisfied, label='Satisfied', color="#77b986", linewidth=2)
    ax.fill(angles, satisfied, color="lightgreen", alpha=0.25)

    ax.plot(angles, neutral, label='Neutral', color="#f3cd5d", linewidth=2)
    ax.fill(angles, neutral, color="yellow", alpha=0.25)

    ax.plot(angles, dissatisfied, label='Dissatisfied', color="#cf7175", linewidth=2)
    ax.fill(angles, dissatisfied, color="salmon", alpha=0.25)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_ylim(0, 100)
    ax.legend(loc='upper right', bbox_to_anchor=(1.5, 1.1))

    plt.title(f'{title}\n', fontdict={'fontweight': 'bold', 'fontsize': 16}, color='darkgreen')

    fig_name = ((title.split(" "))[0]).lower() + "_factors"
    fig.savefig(f"{path_images}\\{fig_name}.png", dpi=300)

    return fig


def get_spider_wellbeing(df, variables, title, path_images):
    
    label_dict = {
        'comfort_satisfaction': 'Comfort\nDimension', 
        'social_satisfaction': 'Social\nDimension', 
        'delight_satisfaction': 'Delight\nDimension'
    }


    satisfied = calc_response_percentages_by_area(df, variables, 2)
    neutral = calc_response_percentages_by_area(df, variables, 1)
    dissatisfied = calc_response_percentages_by_area(df, variables, 0)


    satisfied += [satisfied[0]]
    neutral += [neutral[0]]
    dissatisfied += [dissatisfied[0]]


    angles = np.linspace(0, 2 * np.pi, len(variables), endpoint=False).tolist()
    angles += angles[:1]

    labels = [label_dict.get(var, var) for var in variables]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

    ax.plot(angles, satisfied, label='Satisfied', color="#77b986", linewidth=2)
    ax.fill(angles, satisfied, color="lightgreen", alpha=0.25)

    ax.plot(angles, neutral, label='Neutral', color="#f3cd5d", linewidth=2)
    ax.fill(angles, neutral, color="yellow", alpha=0.25)

    ax.plot(angles, dissatisfied, label='Dissatisfied', color="#cf7175", linewidth=2)
    ax.fill(angles, dissatisfied, color="salmon", alpha=0.25)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_ylim(0, 100)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.title(f'{title}\n', fontdict={'fontweight': 'bold', 'fontsize': 14}, color='darkgreen')
    plt.tight_layout()

    fig_name = ((title.split(" "))[0]).lower() + "_factors"
    fig.savefig(f"{path_images}\\{fig_name}.png", dpi=300)

    return fig


def get_violing(df, var, title, path_images):
    df['distribution'] = 'Satisfaction'

    fig, ax = plt.subplots(figsize=(6, 6))

    sns.violinplot(
        x='distribution',
        y=var,
        data=df,
        ax=ax,
        fill=False,
        linewidth=0.1,
        width=0.4
        )

    custom_colors = [
        (0.00, (0.788, 0.212, 0.133)),   # -1
        (0.25, (0.788, 0.212, 0.133)),       # 0
        (0.45, (0.859, 0.820, 0.267)),
        (0.55, (0.859, 0.820, 0.267)),    # 1
        (0.75, (0.396, 0.612, 0.298)),     # 2
        (1.00, (0.396, 0.612, 0.298))  # 3
    ]
    cmap = LinearSegmentedColormap.from_list("custom_satisfaction", custom_colors)
    norm = Normalize(vmin=-1, vmax=3)

    ax.set_ylim(-1.5, 3.5)
    ax.set_yticks([-1, 0, 1, 2, 3])
    ax.set_yticklabels(['', 'Dissatisfied', 'Neutral', 'Satisfied', ''], fontsize=12)
    ax.set_ylabel("")
    ax.set_xlim(-0.5, 0.5)

    plt.title(f'{title}\n', fontdict = {'fontweight': 'bold', 'fontsize': 14}, color = 'darkgreen')

    for child in ax.get_children():
        if isinstance(child, matplotlib.collections.PolyCollection):
            paths = child.get_paths()
            for path in paths:
                verts = path.vertices
                min_y = np.min(verts[:, 1])
                max_y = np.max(verts[:, 1])

                y_vals = np.linspace(min_y, max_y, 1000)
                y_normed = norm(y_vals)
                colors = cmap(y_normed)
                img_array = np.tile(colors[:, np.newaxis, :], (1, 2, 1)) 

                extent = [path.get_extents().xmin, path.get_extents().xmax, min_y, max_y]

                polygon = Polygon(verts, facecolor='none', edgecolor='none')
                ax.add_patch(polygon)

                im = ax.imshow(
                    img_array,
                    extent=extent,
                    origin='lower',
                    aspect='auto'
                )
                im.set_clip_path(polygon)

    # Colorbar
    ax_divider = make_axes_locatable(ax)
    cax = ax_divider.append_axes("right", size="5%", pad="2%")
    cb = matplotlib.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm, orientation='vertical')
    cb.set_ticks([0, 1, 2])
    cb.set_ticklabels(['Dissatisfied', 'Neutral', 'Satisfied'])

    pa = (f'{path_images}\\{var}.png')
    plt.savefig(pa, bbox_inches='tight', dpi = 300, transparent=True)

    return fig


def get_violin_wellbeing(df, var, title, path_images):
    df['distribution'] = 'Satisfaction'

    fig, ax = plt.subplots(figsize=(6, 6))

    sns.violinplot(
        x='distribution',
        y=var,
        data=df,
        ax=ax,
        fill=False,
        linewidth=0.1,
        width=0.4
        )

    custom_colors = [
        (0.00, (0.788, 0.212, 0.133)),   # -1
        (0.25, (0.788, 0.212, 0.133)),       # 0
        (0.45, (0.859, 0.820, 0.267)),
        (0.55, (0.859, 0.820, 0.267)),    # 1
        (0.75, (0.396, 0.612, 0.298)),     # 2
        (1.00, (0.396, 0.612, 0.298))  # 3
    ]
    cmap = LinearSegmentedColormap.from_list("custom_satisfaction", custom_colors)
    norm = Normalize(vmin=-1, vmax=7)

    ax.set_ylim(-1.5, 7.5)
    ax.set_yticks([-1, 0, 1, 2, 3, 4, 5, 6, 7])
    ax.set_yticklabels(['','Dissatisfied', '', '', 'Neutral', '', '', 'Satisfied', ''], fontsize=12)
    ax.set_ylabel("")
    ax.set_xlim(-0.5, 0.5)
    plt.title(f'{title}\n', fontdict = {'fontweight': 'bold', 'fontsize': 14}, color = 'darkgreen')

    for child in ax.get_children():
        if isinstance(child, matplotlib.collections.PolyCollection):
            paths = child.get_paths()
            for path in paths:
                verts = path.vertices
                min_y = np.min(verts[:, 1])
                max_y = np.max(verts[:, 1])

                y_vals = np.linspace(min_y, max_y, 1000)
                y_normed = norm(y_vals)
                colors = cmap(y_normed)
                img_array = np.tile(colors[:, np.newaxis, :], (1, 2, 1)) 

                extent = [path.get_extents().xmin, path.get_extents().xmax, min_y, max_y]

                polygon = Polygon(verts, facecolor='none', edgecolor='none')
                ax.add_patch(polygon)

                im = ax.imshow(
                    img_array,
                    extent=extent,
                    origin='lower',
                    aspect='auto'
                )
                im.set_clip_path(polygon)

    
    ax_divider = make_axes_locatable(ax)
    cax = ax_divider.append_axes("right", size="5%", pad="2%")
    cb = matplotlib.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm, orientation='vertical')
    cb.set_ticks([0, 3, 6])
    cb.set_ticklabels(['Dissatisfied', 'Neutral', 'Satisfied'])

    pa = (f'{path_images}\\{var}.png')
    plt.savefig(pa, bbox_inches='tight', dpi = 300, transparent=True)

    return fig



def plot_sbar_satisfaction(df, variables, title, output_folder):
    label_dict = {
        't_sens': 'Thermal\nSensation', 
        't_satisf': 'Thermal\nSatisf', 
        'daylight': 'Daylight\nPercep', 
        'dl_satisf': 'Daylight\nSatisf', 
        'sound': 'Sound\nPercep',
        'sound_satisf': 'Acoustic\nSatisf', 
        'air_quali': 'Air Quality\nSatisf',
        'hor_views_satisf': 'Views\nOutdoors', 
        'green_views_satisf': 'Views\nNature', 
        'sky_views_satisf': 'Views\nSky', 
        'balcony_satisf': 'Access Green\n(balconies)', 
        'space_size_satisf': 'Space\nSize',
        'social_green_satisf': 'Social\nOutdoors/Green', 
        'social_amount_satisf': 'Amount of\nSocial Spaces', 
        'social_distribution_satisf': 'Distribution of\nSocial Spaces',
        'comfort_satisfaction': 'Comfort\nDimension', 
        'social_satisfaction': 'Social\nDimension', 
        'delight_satisfaction': 'Delight\nDimension'
    }

    satisfaction_labels = ['Dissatisfied', 'Neutral', 'Satisfied']
    satisfaction_colors = {0: '#cf7175', 1: '#f3cd5d', 2: '#77b986'}
    
    percentages = {label: [] for label in satisfaction_labels}
    labels = [label_dict.get(var, var) for var in variables]

    for var in variables:
        df['mapped_satisf'] = df[var].apply(map_to_satisfaction)
        total_area = df['floor_area'].sum()
        for level in range(3):
            area_level = df[df['mapped_satisf'] == level]['floor_area'].sum()
            percentages[satisfaction_labels[level]].append((area_level / total_area) * 100)

    x = range(len(variables))
    fig, ax = plt.subplots(figsize=(8, 5))
    bottom = [0] * len(variables)

    for level, label in enumerate(satisfaction_labels):
        current_values = percentages[label]
        ax.bar(x, current_values, bottom=bottom, color=satisfaction_colors[level], label=label, width=0.6)
        for i, val in enumerate(current_values):
            if val > 0:
                ax.text(i, bottom[i] + val / 2, f'{val:.0f}%', ha='center', va='center', fontsize=12, color='black')
        bottom = [i + j for i, j in zip(bottom, current_values)]

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=14)
    ax.set_ylabel('%', fontsize=16)
    ax.set_title(f'{title}\n', fontdict={'fontweight': 'bold', 'fontsize': 18}, color='darkgreen')
    ax.legend(title='Satisfaction Level', bbox_to_anchor=(1.0, 0), loc='lower left', fontsize=10)
    ax.set_ylim(0, 100)

    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, title.replace("Factor_satisf_bar", "_") + ".png")
    plt.tight_layout()
    plt.savefig(output_path)

    return fig



def map_to_satisfaction(value):
    if pd.isna(value): 
        return None  

    if isinstance(value, (float, int)):
        if value < 0.66:
            return 0  # Dissatisfied
        elif value > 1.33:
            return 2  # Satisfied
        else:
            return 1  # Neutral

    return None  


def map_to_wellbeing(value):
    if pd.isna(value): 
        return None  

    if isinstance(value, (float, int)):
        if value < 2:
            return 0  # Dissatisfied
        elif value > 4:
            return 2  # Satisfied
        else:
            return 1  # Neutral

    return None  


def plot_satisf_bar(df, level_column, title, output_folder):
    area_column = 'floor_area'
    levels = ['Dissatisfied', 'Neutral', 'Satisfied']
    level_map = {0: 'Dissatisfied', 1: 'Neutral', 2: 'Satisfied'}
    colors = ['#cf7175', '#f3cd5d', '#77b986']

    df['satisf_mapped'] = df[level_column].apply(map_to_satisfaction)
    df['level_label'] = df['satisf_mapped'].map(level_map)

    area_sum = df.groupby('level_label')[area_column].sum()
    total_area = area_sum.sum()
    area_percent = (area_sum / total_area * 100).reindex(levels).fillna(0)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(levels, area_percent, color=colors, width=0.6)
    ax.set_xticks(range(len(levels)))
    ax.set_xticklabels(levels, fontsize=14)
    ax.set_ylabel('%', fontsize=16)
    ax.set_title(f'{title}\n', fontdict={'fontweight': 'bold', 'fontsize': 18}, color='darkgreen')
    ax.set_ylim(0, 100)

    for i, v in enumerate(area_percent):
        ax.text(i, v + 1, f'{v:.1f}%', ha='center', fontsize=12)

    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, title.replace("O_Satisf_bar", "_") + ".png")
    plt.tight_layout()
    plt.savefig(output_path)

    return fig


def plot_wellbeing_bar(df, level_column, title, output_folder):
    area_column = 'floor_area'
    levels = ['Dissatisfied', 'Neutral', 'Satisfied']
    level_map = {0: 'Dissatisfied', 1: 'Neutral', 2: 'Satisfied'}
    colors = ['#cf7175', '#f3cd5d', '#77b986']

    df['satisf_mapped'] = df[level_column].apply(map_to_wellbeing)
    df['level_label'] = df['satisf_mapped'].map(level_map)

    area_sum = df.groupby('level_label')[area_column].sum()
    total_area = area_sum.sum()
    area_percent = (area_sum / total_area * 100).reindex(levels).fillna(0)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(levels, area_percent, color=colors, width=0.6)
    ax.set_xticks(range(len(levels)))
    ax.set_xticklabels(levels, fontsize=14)
    ax.set_ylabel('%', fontsize=16)
    ax.set_title(f'{title}\n', fontdict={'fontweight': 'bold', 'fontsize': 18}, color='darkgreen')
    ax.set_ylim(0, 100)

    for i, v in enumerate(area_percent):
        ax.text(i, v + 1, f'{v:.1f}%', ha='center', fontsize=12)

    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, title.replace("O_Satisf_bar", "_") + ".png")
    plt.tight_layout()
    plt.savefig(output_path)

    return fig