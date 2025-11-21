# coding=utf-8
"""Module taken from OpenStudio-standards.

https://github.com/NREL/openstudio-standards/blob/master/
lib/openstudio-standards/prototypes/common/objects/Prototype.CentralAirSourceHeatPump.rb
"""
from __future__ import division

from honeybee_openstudio.openstudio import openstudio_model
from .utilities import ems_friendly_name


def create_central_air_source_heat_pump(model, hot_water_loop, name=None, cop=3.65):
    """Prototype CentralAirSourceHeatPump object using PlantComponentUserDefined.

    Args:
        model: [OpenStudio::Model::Model] OpenStudio model object.
        hot_water_loop: [<OpenStudio::Model::PlantLoop>] a hot water loop served
            by the central air source heat pump.
        name: [String] the name of the central air source heat pump, or nil in
            which case it will be defaulted.
        cop: [Double] air source heat pump rated cop.
    """
    # create the PlantComponentUserDefined object as a proxy for the Central Air Source Heat Pump
    plant_comp = openstudio_model.PlantComponentUserDefined(model)
    if name is None:
        if hot_water_loop is None:
            name = 'Central Air Source Heat Pump'
        else:
            name = '{} Central Air Source Heat Pump'.format(hot_water_loop.nameString())

    # change equipment name for EMS validity
    plant_comp.setName(ems_friendly_name(name))

    # set plant component properties
    plant_comp.setPlantLoadingMode('MeetsLoadWithNominalCapacityHiOutLimit')
    plant_comp.setPlantLoopFlowRequestMode('NeedsFlowIfLoopOn')

    # plant design volume flow rate internal variable
    vdot_des_int_var = openstudio_model.EnergyManagementSystemInternalVariable(
        model, 'Plant Design Volume Flow Rate')
    vdot_des_int_var.setName('{}_Vdot_Des_Int_Var'.format(plant_comp.nameString()))
    vdot_des_int_var.setInternalDataIndexKeyName(str(hot_water_loop.handle()))

    # inlet temperature internal variable
    tin_int_var = openstudio_model.EnergyManagementSystemInternalVariable(
        model, 'Inlet Temperature for Plant Connection 1')
    tin_int_var.setName('{}_Tin_Int_Var'.format(plant_comp.nameString()))
    tin_int_var.setInternalDataIndexKeyName(str(plant_comp.handle()))

    # inlet mass flow rate internal variable
    mdot_int_var = openstudio_model.EnergyManagementSystemInternalVariable(
        model, 'Inlet Mass Flow Rate for Plant Connection 1')
    mdot_int_var.setName('{}_Mdot_Int_Var'.format(plant_comp.nameString()))
    mdot_int_var.setInternalDataIndexKeyName(str(plant_comp.handle()))

    # inlet specific heat internal variable
    cp_int_var = openstudio_model.EnergyManagementSystemInternalVariable(
        model, 'Inlet Specific Heat for Plant Connection 1')
    cp_int_var.setName('{}_Cp_Int_Var'.format(plant_comp.nameString()))
    cp_int_var.setInternalDataIndexKeyName(str(plant_comp.handle()))

    # inlet density internal variable
    rho_int_var = openstudio_model.EnergyManagementSystemInternalVariable(
        model, 'Inlet Density for Plant Connection 1')
    rho_int_var.setName('{}_rho_Int_Var'.format(plant_comp.nameString()))
    rho_int_var.setInternalDataIndexKeyName(str(plant_comp.handle()))

    # load request internal variable
    load_int_var = openstudio_model.EnergyManagementSystemInternalVariable(
        model, 'Load Request for Plant Connection 1')
    load_int_var.setName('{}_Load_Int_Var'.format(plant_comp.nameString()))
    load_int_var.setInternalDataIndexKeyName(str(plant_comp.handle()))

    # supply outlet node setpoint temperature sensor
    setpt_mgr_sch_sen = openstudio_model.EnergyManagementSystemSensor(
        model, 'Schedule Value')
    setpt_mgr_sch_sen.setName('{}_Setpt_Mgr_Temp_Sen'.format(plant_comp.nameString()))
    for m in hot_water_loop.supplyOutletNode().setpointManagers():
        if m.to_SetpointManagerScheduled().is_initialized():
            m_name = m.to_SetpointManagerScheduled().get().schedule().nameString()
            setpt_mgr_sch_sen.setKeyName(m_name)

    # outdoor air drybulb temperature sensor
    oa_dbt_sen = openstudio_model.EnergyManagementSystemSensor(
        model, 'Site Outdoor Air Drybulb Temperature')
    oa_dbt_sen.setName('{}_OA_DBT_Sen'.format(plant_comp.nameString()))
    oa_dbt_sen.setKeyName('Environment')

    # minimum mass flow rate actuator
    mdot_min_act = plant_comp.minimumMassFlowRateActuator().get()
    mdot_min_act.setName('{}_Mdot_Min_Act'.format(plant_comp.nameString()))

    # maximum mass flow rate actuator
    mdot_max_act = plant_comp.maximumMassFlowRateActuator().get()
    mdot_max_act.setName('{}_Mdot_Max_Act'.format(plant_comp.nameString()))

    # design flow rate actuator
    vdot_des_act = plant_comp.designVolumeFlowRateActuator().get()
    vdot_des_act.setName('{}_Vdot_Des_Act'.format(plant_comp.nameString()))

    # minimum loading capacity actuator
    cap_min_act = plant_comp.minimumLoadingCapacityActuator().get()
    cap_min_act.setName('{}_Cap_Min_Act'.format(plant_comp.nameString()))

    # maximum loading capacity actuator
    cap_max_act = plant_comp.maximumLoadingCapacityActuator().get()
    cap_max_act.setName('{}_Cap_Max_Act'.format(plant_comp.nameString()))

    # optimal loading capacity actuator
    cap_opt_act = plant_comp.optimalLoadingCapacityActuator().get()
    cap_opt_act.setName('{}_Cap_Opt_Act'.format(plant_comp.nameString()))

    # outlet temperature actuator
    tout_act = plant_comp.outletTemperatureActuator().get()
    tout_act.setName('{}_Tout_Act'.format(plant_comp.nameString()))

    # mass flow rate actuator
    mdot_req_act = plant_comp.massFlowRateActuator().get()
    mdot_req_act.setName('{}_Mdot_Req_Act'.format(plant_comp.nameString()))

    # heat pump COP curve
    constant_coeff = 1.932 + (cop - 3.65)
    hp_cop_curve = openstudio_model.CurveQuadratic(model)
    hp_cop_curve.setCoefficient1Constant(constant_coeff)
    hp_cop_curve.setCoefficient2x(0.227674286)
    hp_cop_curve.setCoefficient3xPOW2(-0.007313143)
    hp_cop_curve.setMinimumValueofx(1.67)
    hp_cop_curve.setMaximumValueofx(12.78)
    hp_cop_curve.setInputUnitTypeforX('Temperature')
    hp_cop_curve.setOutputUnitType('Dimensionless')

    # heat pump COP curve index variable
    hp_cop_curve_idx_var = \
        openstudio_model.EnergyManagementSystemCurveOrTableIndexVariable(model, hp_cop_curve)

    # high outlet temperature limit actuator
    tout_max_act = openstudio_model.EnergyManagementSystemActuator(
        plant_comp, 'Plant Connection 1', 'High Outlet Temperature Limit')
    tout_max_act.setName('{}_Tout_Max_Act'.format(plant_comp.nameString()))

    # init program
    init_pgrm = plant_comp.plantInitializationProgram().get()
    init_pgrm.setName('{}_Init_Pgrm'.format(plant_comp.nameString()))
    init_pgrm_body = \
        'SET Loop_Exit_Temp = {loop_exit_temp}\n' \
        'SET Loop_Delta_Temp = {design_temp_diff}\n' \
        'SET Cp = @CPHW Loop_Exit_Temp\n' \
        'SET rho = @RhoH2O Loop_Exit_Temp\n' \
        'SET {vdot_des_act_handle} = {vdot_des_int_var_handle}\n' \
        'SET {mdot_min_act_handle} = 0\n' \
        'SET Mdot_Max = {vdot_des_int_var_handle} * rho\n' \
        'SET {mdot_max_act_handle} = Mdot_Max\n' \
        'SET Cap = Mdot_Max * Cp * Loop_Delta_Temp\n' \
        'SET {cap_min_act_handle} = 0\n' \
        'SET {cap_max_act_handle} = Cap\n' \
        'SET {cap_opt_act_handle} = 1 * Cap'.format(
            loop_exit_temp=hot_water_loop.sizingPlant().designLoopExitTemperature(),
            design_temp_diff=hot_water_loop.sizingPlant().loopDesignTemperatureDifference(),
            vdot_des_act_handle=vdot_des_act.handle(),
            vdot_des_int_var_handle=vdot_des_int_var.handle(),
            mdot_min_act_handle=mdot_min_act.handle(),
            mdot_max_act_handle=mdot_max_act.handle(),
            cap_min_act_handle=cap_min_act.handle(),
            cap_max_act_handle=cap_max_act.handle(),
            cap_opt_act_handle=cap_opt_act.handle()
        )
    init_pgrm.setBody(init_pgrm_body)

    # sim program
    sim_pgrm = plant_comp.plantSimulationProgram().get()
    sim_pgrm.setName('{}_Sim_Pgrm'.format(plant_comp.nameString()))
    sim_pgrm_body = \
        'SET tmp = {load_int_var_handle}\n' \
        'SET tmp = {tin_int_var_handle}\n' \
        'SET tmp = {mdot_int_var_handle}\n' \
        'SET {tout_max_act_handle} = 75.0\n' \
        'IF {load_int_var_handle} == 0\n' \
        'SET {tout_act_handle} = {tin_int_var_handle}\n' \
        'SET {mdot_req_act_handle} = 0\n' \
        'SET Elec = 0\n' \
        'RETURN\n' \
        'ENDIF\n' \
        'IF {load_int_var_handle} >= {cap_max_act_handle}\n' \
        'SET Qdot = {cap_max_act_handle}\n' \
        'SET Mdot = {mdot_max_act_handle}\n' \
        'SET {mdot_req_act_handle} = Mdot\n' \
        'SET {tout_act_handle} = (Qdot / (Mdot * {cp_int_var_handle})) + {tin_int_var_handle}\n' \
        'IF {tout_act_handle} > {tout_max_act_handle}\n' \
        'SET {tout_act_handle} = {tout_max_act_handle}\n' \
        'SET Qdot = Mdot * {cp_int_var_handle} * ({tout_act_handle} - {tin_int_var_handle})\n' \
        'ENDIF\n' \
        'ELSE\n' \
        'SET Qdot = {load_int_var_handle}\n' \
        'SET {tout_act_handle} = {setpt_mgr_sch_sen_handle}\n' \
        'SET Mdot = Qdot / ({cp_int_var_handle} * ({tout_act_handle} - {tin_int_var_handle}))\n' \
        'SET {mdot_req_act_handle} = Mdot\n' \
        'ENDIF\n' \
        'SET Tdb = {oa_dbt_sen_handle}\n' \
        'SET COP = @CurveValue {hp_cop_curve_idx_var_handle} Tdb\n' \
        'SET EIR = 1 / COP\n' \
        'SET Pwr = Qdot * EIR\n' \
        'SET Elec = Pwr * SystemTimestep * 3600'.format(
            load_int_var_handle=load_int_var.handle(),
            tin_int_var_handle=tin_int_var.handle(),
            mdot_int_var_handle=mdot_int_var.handle(),
            tout_max_act_handle=tout_max_act.handle(),
            tout_act_handle=tout_act.handle(),
            mdot_req_act_handle=mdot_req_act.handle(),
            cap_max_act_handle=cap_max_act.handle(),
            mdot_max_act_handle=mdot_max_act.handle(),
            cp_int_var_handle=cp_int_var.handle(),
            setpt_mgr_sch_sen_handle=setpt_mgr_sch_sen.handle(),
            oa_dbt_sen_handle=oa_dbt_sen.handle(),
            hp_cop_curve_idx_var_handle=hp_cop_curve_idx_var.handle()
        )
    sim_pgrm.setBody(sim_pgrm_body)

    # init program calling manager
    init_mgr = plant_comp.plantInitializationProgramCallingManager().get()
    init_mgr.setName('{}_Init_Pgrm_Mgr'.format(plant_comp.nameString()))

    # sim program calling manager
    sim_mgr = plant_comp.plantSimulationProgramCallingManager().get()
    sim_mgr.setName('{}_Sim_Pgrm_Mgr'.format(plant_comp.nameString()))

    # metered output variable
    elec_mtr_out_var = openstudio_model.EnergyManagementSystemMeteredOutputVariable(
        model, '{} Electricity Consumption'.format(plant_comp.nameString()))
    elec_mtr_out_var.setName('{} Electricity Consumption'.format(plant_comp.nameString()))
    elec_mtr_out_var.setEMSVariableName('Elec')
    elec_mtr_out_var.setUpdateFrequency('SystemTimestep')
    elec_mtr_out_var.setString(4, str(sim_pgrm.handle()))
    elec_mtr_out_var.setResourceType('Electricity')
    elec_mtr_out_var.setGroupType('HVAC')
    elec_mtr_out_var.setEndUseCategory('Heating')
    elec_mtr_out_var.setEndUseSubcategory('')
    elec_mtr_out_var.setUnits('J')

    # add to supply side of hot water loop if specified
    if hot_water_loop is not None:
        hot_water_loop.addSupplyBranchForComponent(plant_comp)

    # add operation scheme
    htg_op_scheme = openstudio_model.PlantEquipmentOperationHeatingLoad(model)
    htg_op_scheme.addEquipment(1000000000, plant_comp)
    hot_water_loop.setPlantEquipmentOperationHeatingLoad(htg_op_scheme)

    return plant_comp
