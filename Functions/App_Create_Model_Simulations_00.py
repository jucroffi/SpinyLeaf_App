
import os
import sys  

import streamlit as st
import math
from pathlib import Path


from ladybug.futil import write_to_file

from ladybug.sql import SQLiteResult

from ladybug.wea import Wea



from honeybee.aperture import Aperture
from honeybee.room import Room
from honeybee.typing import clean_and_id_string


from honeybee_energy.simulation.runperiod import RunPeriod
from honeybee_energy.constructionset import ConstructionSet, ApertureConstructionSet, WallConstructionSet, RoofCeilingConstructionSet, FloorConstructionSet
from honeybee_energy.construction.opaque import OpaqueConstruction
from honeybee_energy.construction.window import WindowConstruction
from honeybee_energy.programtype import ProgramType
from honeybee_energy.lib.programtypes import building_program_type_by_identifier

from honeybee_energy.load.people import People
from honeybee_energy.load.lighting import Lighting
from honeybee_energy.load.equipment import ElectricEquipment, GasEquipment
from honeybee_energy.load.hotwater import ServiceHotWater
from honeybee_energy.load.ventilation import Ventilation
from honeybee_energy.load.setpoint import Setpoint
from honeybee_energy.load.infiltration import Infiltration
from honeybee_energy.run import run_idf
from honeybee_energy.schedule.fixedinterval import ScheduleFixedInterval

from honeybee_energy.ventcool.control import VentilationControl
from honeybee_energy.ventcool.opening import VentilationOpening

from honeybee_energy.material.glazing import EnergyWindowMaterialSimpleGlazSys
from honeybee_energy.material.opaque import EnergyMaterialNoMass
from honeybee_energy.lib.materials import opaque_material_by_identifier
from honeybee_energy.lib.constructionsets import construction_set_by_identifier
from honeybee_energy.lib.constructions import window_construction_by_identifier

from honeybee_energy.simulation.parameter import SimulationParameter
from honeybee_energy.simulation.output import SimulationOutput
from honeybee_energy.simulation.control import SimulationControl
from honeybee_energy.simulation.shadowcalculation import ShadowCalculation

from honeybee_energy.reader import clean_idf_file_contents
from honeybee_radiance.sensorgrid import SensorGrid

from lbt_recipes.recipe import Recipe
import traceback




def create_program(usage, occupants_per_area):
    '''Create and return programs according to the sample usage'''
    prog = {0 : 'MediumOffice',
            1 : 'Retail',
            2 : 'MidriseApartment',
            3 : 'LargeDataCenterHighITE',
            4 : 'SecondarySchool',
            5 : 'Hospital',
            6 : 'Laboratory'}
    
    office = 0.0
    retail = 1.0
    resid = 2.0
    data_c = 3.0
    school = 4.0
    hosp = 5.0
    lab = 6.0

    map_ppl = {office: occupants_per_area, retail: 0.1598, resid: occupants_per_area, data_c: 0.0000, school: 0.3586, hosp: 0.0659, lab: 0.0482}
    map_lgt = {office: 6.6101, retail: 10.4141, resid: 6.3841, data_c: 6.8889, school: 8.2290, hosp: 10.6886, lab: 10.6477}
    map_eqp = {office: 11.6352, retail: 5.2420, resid: 5.9363, data_c: 5381.9500, school: 27.5694, hosp: 28.6696, lab: 46.7928}
    map_gas = {office: 0.0000, retail: 0.0000, resid: 0.0000, data_c: 0.0000, school: 134.2846, hosp: 3.2278, lab: 0.0000}
    map_hwt = {office: 0.0258, retail: 0.0262, resid: 0.1179, data_c: 0.0000, school: 0.2791, hosp: 0.0373, lab: 0.0284}
    map_vpp = {office: 0.00165, retail: 0.00311, resid: 0.0000, data_c: 0.00236, school: 0.00295, hosp: 0.0022, lab: 0.00396}
    map_vpa = {office: 0.00037, retail: 0.00061, resid: 0.000043, data_c: 0.00030, school: 0.00073, hosp: 0.00034, lab: 0.0024}
    map_vac = {office: 0.0000, retail: 0.0000, resid: 0.2765, data_c: 0.0000, school: 0.0000, hosp: 0.7700, lab: 6.6000}
    map_hsp = {office: 20.0, retail: 20.0, resid: 20.0, data_c: 15.0, school: 20.0, hosp: 21.0, lab: 20.5}
    map_csp = {office: 23.0, retail: 23.0, resid: 24.0, data_c: 15.0, school: 23.0, hosp: 21.0, lab: 20.5}
    map_hsb = {office: 16.0, retail: 14.0, resid: 16.0, data_c: 15.0, school: 14.0, hosp: 21.0, lab: 20.5}
    map_csb = {office: 25.0, retail: 28.0, resid: 27.0, data_c: 15.0, school: 28.0, hosp: 21.0, lab: 20.5}
    map_inf = {office: 0.0003, retail: 0.0003, resid: 0.0003, data_c: 0.0003, school: 0.0003, hosp: 0.0003, lab: 0.0003}
    
    program_usage = building_program_type_by_identifier(prog[usage])

    try:
        occ_sch = program_usage.people.occupancy_schedule
        occ_values = occ_sch.values()
        ppl = People(f'ppl_{usage}', map_ppl[usage],occ_sch, activity_schedule=program_usage.people.activity_schedule)
    except:
        ppl = None
        occ_values = [0]*8760

    try:    
        name = f'gas_{usage}'
        _name = clean_and_id_string(name)
        gas = GasEquipment(_name, map_gas[usage], program_usage.gas_equipment.schedule)
    except:
        gas = None

    try:
        name = f'lgt_{usage}'
        _name = clean_and_id_string(name)
        lgt = Lighting(_name, map_lgt[usage], program_usage.lighting.schedule)
    except:
        lgt = None

    try:
        name = f'eqp_{usage}'
        _name = clean_and_id_string(name) 
        eqp = ElectricEquipment(_name, map_eqp[usage], program_usage.electric_equipment.schedule)
    except:
        eqp = None

    try:
        name = f'hwt_{usage}'
        _name = clean_and_id_string(name) 
        hwt = ServiceHotWater(_name, map_hwt[usage], program_usage.service_hot_water.schedule, target_temperature=60, sensible_fraction=0.2, latent_fraction=0.05)
    except:
        hwt = None

    try:
        name = f'vent_{usage}'
        _name = clean_and_id_string(name) 
        vent = Ventilation(_name, flow_per_person= map_vpp[usage], flow_per_area=map_vpa[usage], flow_per_zone=map_vac[usage], air_changes_per_hour=map_vac[usage],
                           schedule=program_usage.ventilation.schedule)
    except:
        vent = None

    try:
        name = f'inf_{usage}'
        _name = clean_and_id_string(name) 
        inf = Infiltration(_name, map_inf[usage], program_usage.infiltration.schedule)
    except:
        inf = None

    # setpoints and setbacks    
    hsp_sch = []
    csp_sch = []
    for o in occ_values:
        if o>.1:
            hsp_sch.append(map_hsp[usage])
            csp_sch.append(map_csp[usage])
        else:
            hsp_sch.append(map_hsb[usage])
            csp_sch.append(map_csb[usage])

    h_name = clean_and_id_string('heating_sch') 
    heating_sch = ScheduleFixedInterval(h_name, hsp_sch)
    c_name = clean_and_id_string('cooling_sch') 
    cooling_sch = ScheduleFixedInterval(c_name, csp_sch)

    setpt = Setpoint('setpoints', heating_sch, cooling_sch)
    pn= f'prog_{prog[usage]}'
    p_name = clean_and_id_string(pn) 
    program = ProgramType(p_name, people=ppl, lighting=lgt, electric_equipment=eqp, gas_equipment=gas, service_hot_water=hwt, 
                                            infiltration=inf, ventilation=vent, setpoint=setpt)
    
    return program
    




# In[33]:


def vent_control(usage, operable):
    '''Create and return ventilation control settings'''
    
    prog = {0 : 'MediumOffice',
            1 : 'Retail',
            2 : 'MidriseApartment',
            3 : 'LargeDataCenterHighITE',
            4 : 'SecondarySchool',
            5 : 'Hospital',
            6 : 'Laboratory'}
    
    office = 0
    retail = 1.0
    resid = 2
    data_c = 3.0
    school = 4.0
    hosp = 5.0
    lab = 6.0
    
    map_hsp = {office: 20.0, retail: 20.0, resid: 20.0, data_c: 15.0, school: 20.0, hosp: 21.0, lab: 20.5}
    map_csp = {office: 23.0, retail: 23.0, resid: 24.0, data_c: 15.0, school: 23.0, hosp: 21.0, lab: 20.5}
    
    program_usage = building_program_type_by_identifier(prog[usage])
    
    if operable == 1:
        min_in_temp = map_hsp[usage] + 1
        max_in_temp = map_csp[usage] - 1
        min_out_temp = -100
        max_out_temp = 100
    else:
        min_in_temp = -100
        max_in_temp = 100
        min_out_temp = -100
        max_out_temp = 100 
    
    vent_c = VentilationControl(min_in_temp, max_in_temp, min_out_temp, max_out_temp, 1, program_usage.ventilation.schedule)
    return vent_c



def construction_set(win_u_value, shgc, wall_r, roof_r, ground_r, n):
    '''Create and return construction sets based on the sample material property values (U values, R values, SHGC)'''

    wind_con = WindowConstruction(f'Uv_{win_u_value}_SHGC_{shgc}', [EnergyWindowMaterialSimpleGlazSys(f'win_nomass', win_u_value, shgc)])
    
    wall_mass = opaque_material_by_identifier('4 in. Normalweight Concrete Wall')
    wall_con = OpaqueConstruction(f'wallR_{wall_r}', [wall_mass, EnergyMaterialNoMass(f'wall_nomass', wall_r - (wall_mass.r_value))])
    
    roof_mass1 = opaque_material_by_identifier('Metal Roofing')
    roof_mass2 = opaque_material_by_identifier('6 in. Heavyweight Concrete Roof')
    roof_con = OpaqueConstruction(f'roofR_{roof_r}', [roof_mass1, roof_mass2, EnergyMaterialNoMass(f'roof_nomass', (roof_r - (roof_mass1.r_value)- (roof_mass2.r_value)))])
    
    ground_mass = opaque_material_by_identifier('100mm Normalweight concrete floor')
    ground_con = OpaqueConstruction(f'groundR_{ground_r}',[ground_mass, EnergyMaterialNoMass(f'ground_nomass', ground_r - (ground_mass.r_value))])

    construction_set_by_identifier

    wind_set = ApertureConstructionSet(window_construction=wind_con)
    wall_set = WallConstructionSet(exterior_construction=wall_con, ground_construction=wall_con)
    roof_set = RoofCeilingConstructionSet(exterior_construction=roof_con)
    ground_set = FloorConstructionSet(ground_construction=ground_con)

    c_set = ConstructionSet(n, wall_set=wall_set, floor_set=ground_set, roof_ceiling_set=roof_set, aperture_set=wind_set)
    
    return c_set



def construction_set_op(window, wall_type, wall_r, roof_type, roof_r, ground_type, ground_r, n):
    
    ins_name = { 2: 'Typical Insulation-R11',
                 3: 'Typical Insulation-R17',
                 4: 'Typical Insulation-R23',
                 5: 'Typical Insulation-R29',
                 6: 'Typical Insulation-R34',
                 7: 'Typical Insulation-R40',
                 8: 'Typical Insulation-R46',
                 9: 'Typical Insulation-R52',
                 10: 'Typical Insulation-R57'}

    wall_ins = opaque_material_by_identifier(ins_name[wall_r])

    concrete = 'Generic LW Concrete'
    gypsum = 'Gypsum Or Plaster Board - 3/8 in.'
    metal_siding = 'Metal Siding'

    if wall_type == 'GRC_Insul_Plasterboard':
        wall_mass = opaque_material_by_identifier(concrete)
        wall_finish = opaque_material_by_identifier(gypsum)
    
        wall_con = OpaqueConstruction(f'wallR_{wall_r}', [wall_mass, wall_ins, wall_finish])

    if wall_type == 'Metal_Insul_GRC':
        wall_mass = opaque_material_by_identifier(concrete)
        wall_finish = opaque_material_by_identifier(metal_siding)
    
        wall_con = OpaqueConstruction(f'wallR_{wall_r}', [wall_finish, wall_ins, wall_mass])
    
    roof_mass1 = opaque_material_by_identifier('Metal Roofing')
    roof_mass2 = opaque_material_by_identifier('6 in. Heavyweight Concrete Roof')
    roof_con = OpaqueConstruction(f'roofR_{roof_r}', [roof_mass1, roof_mass2, EnergyMaterialNoMass(f'roof_nomass', (roof_r - (roof_mass1.r_value)- (roof_mass2.r_value)))])
    
    ground_mass = opaque_material_by_identifier('100mm Normalweight concrete floor')
    ground_con = OpaqueConstruction(f'groundR_{ground_r}',[ground_mass, EnergyMaterialNoMass(f'ground_nomass', ground_r - (ground_mass.r_value))])

    wind_con = window_construction_by_identifier(window)


    wind_set = ApertureConstructionSet(window_construction=wind_con)
    wall_set = WallConstructionSet(exterior_construction=wall_con, ground_construction=wall_con)
    roof_set = RoofCeilingConstructionSet(exterior_construction=roof_con)
    ground_set = FloorConstructionSet(ground_construction=ground_con)

    c_set = ConstructionSet(n, wall_set=wall_set, floor_set=ground_set, roof_ceiling_set=roof_set, aperture_set=wind_set)
    
    return c_set



def apply_prop(rooms, vent_c, c_set, program, operable, window):
    '''Apply windows, construction sets and program'''
    
    
    Room.intersect_adjacency(rooms, tolerance=0.01, angle_tolerance=1)
    Room.solve_adjacency(rooms, 0.01) 
    
    for r in rooms:

        r.properties.energy.add_default_ideal_air()
        r.properties.energy.window_vent_control=vent_c

        r.properties.energy.assign_ventilation_opening(VentilationOpening(fraction_area_operable=0.25, fraction_height_operable=1.0, wind_cross_vent=True))

        r.properties.energy.construction_set = c_set
        r.properties.energy.program_type = program

        for ap in r.apertures:
            
            ap.is_operable = operable
            wind_con = window_construction_by_identifier(window)
            ap.properties.energy.construction = wind_con




def create_sensors(model, dist):
    
    rooms = list(model.rooms)
    grids = []
    meshes = []

    for idx, room in enumerate(rooms):

        r_mesh = room.generate_grid(dist, y_dim=None, offset=1.2)
        meshes.append(r_mesh)
        name = f'grid_{idx}'
        sens_name = clean_and_id_string(name)
        r_grid = SensorGrid.from_mesh3d(sens_name, r_mesh)
        grids.append(r_grid)

    return meshes, grids



def run_sim_comfort(hb_model, sim_path, epw_file, ddy_file, solar, method, period):

    solar_dist = {0: 'MinimalShadowing',
                  1: 'FullExterior',
                  2: 'FullInteriorAndExterior',
                  3: 'FullExteriorWithReflections',
                  4: 'FullInteriorAndExteriorWithReflections'}

    sim_control = SimulationControl() 

    output_optemp = 'Zone Operative Temperature'
    output_rh = 'Zone Air Relative Humidity'
    output_co2 = 'Zone Air CO2 Concentration'
    sim_output = SimulationOutput()
    sim_output.add_output(output_optemp)
    sim_output.add_output(output_rh)
    sim_output.add_output(output_co2)
    sim_output.add_comfort_metrics()

    run_per = RunPeriod.from_analysis_period(period)

    shadow_calc = ShadowCalculation(solar_distribution = solar_dist[solar], calculation_method=method)

    sim_par = SimulationParameter(output= sim_output, run_period=run_per, simulation_control=sim_control, shadow_calculation=shadow_calc)
    sim_par.sizing_parameter.add_from_ddy(ddy_file)

    co2_idf = clean_idf_file_contents("C:/SpinyLeaf/Config/ZoneAirContaminantBalance.idf")
    idf_str = '\n\n'.join((sim_par.to_idf(), hb_model.to.idf(hb_model), co2_idf))
    idf_path = os.path.join(sim_path, f'model.idf')

    write_to_file(idf_path, idf_str, True)
    run_idf(idf_path, epw_file_path=epw_file, expand_objects=True, silent=True)


def read_comf_results(res_folder):
    
    opt_values = []
    rh_values = []
    co2_values = []

    sf = os.listdir(res_folder)
    sql = 'eplusout.sql'
    
    try:

        if sql in sf:    
            res_sql = f"{res_folder}\\eplusout.sql"
            sql_obj = SQLiteResult(res_sql)

            oper_temp_output = 'Zone Operative Temperature'
            oper_temp = sql_obj.data_collections_by_output_name(oper_temp_output)
            rel_humidity_output = 'Zone Air Relative Humidity'
            rh_out = sql_obj.data_collections_by_output_name(rel_humidity_output)
            co2_output = 'Zone Air CO2 Concentration'
            co2_out = sql_obj.data_collections_by_output_name(co2_output)

            for _data in oper_temp:
                
                opt_val = list(_data.values)
                opt_values.append(opt_val)

            for _data in rh_out:
                
                rh_val = list(_data.values)
                rh_values.append(rh_val)

            for _data in co2_out:
                
                co2_val = list(_data.values)
                co2_values.append(co2_val)

    except:
        pass
            
    return opt_values, rh_values, co2_values


def run_udi(model, epw_file, out_folder):
    
    wea = Wea.from_epw_file(epw_file, timestep=1)
    recipe = Recipe('annual-daylight')
    recipe.default_project_folder = out_folder
    recipe.input_value_by_name('model', model)
    recipe.input_value_by_name('wea', wea)
    recipe.input_value_by_name('north', 0)
    recipe.run()

def run_glare(model, epw_file, out_folder):
    
    wea = Wea.from_epw_file(epw_file, timestep=1)
    recipe = Recipe('imageless-annual-glare')
    recipe.default_project_folder = out_folder
    recipe.input_value_by_name('model', model)
    recipe.input_value_by_name('wea', wea)
    recipe.run()


def convert_udi_to_res(source_folder, target_folder):
    source_folder = Path(source_folder)
    target_folder = Path(target_folder)
    target_folder.mkdir(parents=True, exist_ok=True)

    for file in source_folder.glob("*.udi"):
        
        lines = file.read_text().splitlines()

        if lines and lines[-1].strip() == "":
            lines = lines[:-1]

        parts = file.stem.split('_')
        clean_name = '_'.join(parts[:2]) + ".res"
        dst = target_folder / clean_name

        dst.write_text("\n".join(lines))


def convert_da_to_res(source_folder, target_folder):
    source_folder = Path(source_folder)
    target_folder = Path(target_folder)
    target_folder.mkdir(parents=True, exist_ok=True)

    for file in source_folder.glob("*.da"):
        
        lines = file.read_text().splitlines()

        
        if lines and lines[-1].strip() == "":
            lines = lines[:-1]

        parts = file.stem.split('_')
        clean_name = '_'.join(parts[:2]) + ".res"
        dst = target_folder / clean_name

        
        dst.write_text("\n".join(lines))


def convert_ga_to_res(source_folder, target_folder):
    source_folder = Path(source_folder)
    target_folder = Path(target_folder)
    target_folder.mkdir(parents=True, exist_ok=True)

    for file in source_folder.glob("*.ga"):
        
        lines = file.read_text().splitlines()

        if lines and lines[-1].strip() == "":
            lines = lines[:-1]

        parts = file.stem.split('_')
        clean_name = '_'.join(parts[:2]) + ".res"
        dst = target_folder / clean_name

        
        dst.write_text("\n".join(lines))


