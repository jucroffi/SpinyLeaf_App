# coding=utf-8
"""Import OpenStudio SDK classes in different Python environments."""
import sys
import os
from honeybee_energy.config import folders as hbe_folders


def _os_path_cpython(path_str):
    return path_str


def _os_path_ironpython(path_str):
    os_path_obj = openstudio.OpenStudioUtilitiesCore.toPath(path_str)
    return openstudio.Path(os_path_obj)


def _os_vector_len_cpython(vector):
    return len(vector)


def _os_vector_len_ironpython(vector):
    return vector.Count


def _os_create_vector_cpython(val_list, vector):
    return val_list


def _os_create_vector_ironpython(val_list, vector):
    for val in val_list:
        vector.Add(val)
    return vector


if sys.version_info >= (3, 0):  # we are in cPython and can import normally
    import openstudio
    openstudio_model = openstudio.model
    os_path = _os_path_cpython
    os_vector_len = _os_vector_len_cpython
    os_create_vector = _os_create_vector_cpython
else:  # we are in IronPython and we must import the .NET bindings
    try:  # first see if OpenStudio has already been loaded
        import OpenStudio as openstudio
    except ImportError:
        try:
            import clr
        except ImportError as e:  # No .NET being used
            raise ImportError(
                'Failed to import CLR. OpenStudio SDK is unavailable.\n{}'.format(e))
        # check to be sure that the OpenStudio CSharp folder has been installed
        assert hbe_folders.openstudio_path is not None, \
            'No OpenStudio installation was found on this machine.'
        assert hbe_folders.openstudio_csharp_path is not None, \
            'No OpenStudio CSharp folder was found in the OpenStudio installation ' \
            'at:\n{}'.format(os.path.dirname(hbe_folders.openstudio_path))
        # add the OpenStudio DLL to the Common Language Runtime (CLR)
        os_dll = os.path.join(hbe_folders.openstudio_csharp_path, 'OpenStudio.dll')
        clr.AddReferenceToFileAndPath(os_dll)
        if hbe_folders.openstudio_csharp_path not in sys.path:
            sys.path.append(hbe_folders.openstudio_csharp_path)
        import OpenStudio as openstudio
    openstudio_model = openstudio
    os_path = _os_path_ironpython
    os_vector_len = _os_vector_len_ironpython
    os_create_vector = _os_create_vector_ironpython

# load all of the classes used by this package
# geometry classes
OSModel = openstudio_model.Model
OSPoint3dVector = openstudio.Point3dVector
OSPoint3d = openstudio.Point3d
OSShadingSurfaceGroup = openstudio_model.ShadingSurfaceGroup
OSShadingSurface = openstudio_model.ShadingSurface
OSSubSurface = openstudio_model.SubSurface
OSSurface = openstudio_model.Surface
OSSpace = openstudio_model.Space
OSThermalZone = openstudio_model.ThermalZone
OSBuildingStory = openstudio_model.BuildingStory
OSSurfacePropertyOtherSideCoefficients = openstudio_model.SurfacePropertyOtherSideCoefficients
# schedule classes
OSScheduleTypeLimits = openstudio_model.ScheduleTypeLimits
OSScheduleRuleset = openstudio_model.ScheduleRuleset
OSScheduleRule = openstudio_model.ScheduleRule
OSScheduleDay = openstudio_model.ScheduleDay
OSScheduleFixedInterval = openstudio_model.ScheduleFixedInterval
OSExternalFile = openstudio_model.ExternalFile
OSScheduleFile = openstudio_model.ScheduleFile
OSTime = openstudio.Time
OSTimeSeries = openstudio.TimeSeries
OSVector = openstudio.Vector
# material classes
OSMasslessOpaqueMaterial = openstudio_model.MasslessOpaqueMaterial
OSStandardOpaqueMaterial = openstudio_model.StandardOpaqueMaterial
OSRoofVegetation = openstudio_model.RoofVegetation
OSSimpleGlazing = openstudio_model.SimpleGlazing
OSStandardGlazing = openstudio_model.StandardGlazing
OSGas = openstudio_model.Gas
OSGasMixture = openstudio_model.GasMixture
OSBlind = openstudio_model.Blind
OSShade = openstudio_model.Shade
OSWindowPropertyFrameAndDivider = openstudio_model.WindowPropertyFrameAndDivider
# constructions classes
OSConstruction = openstudio_model.Construction
OSMaterialVector = openstudio_model.MaterialVector
OSShadingControl = openstudio_model.ShadingControl
OSConstructionAirBoundary = openstudio_model.ConstructionAirBoundary
OSZoneMixing = openstudio_model.ZoneMixing
# construction set classes
OSDefaultConstructionSet = openstudio_model.DefaultConstructionSet
OSDefaultSurfaceConstructions = openstudio_model.DefaultSurfaceConstructions
OSDefaultSubSurfaceConstructions = openstudio_model.DefaultSubSurfaceConstructions
# internal mass classes
OSInternalMassDefinition = openstudio_model.InternalMassDefinition
OSInternalMass = openstudio_model.InternalMass
# loads classes
OSSpaceType = openstudio_model.SpaceType
OSPeopleDefinition = openstudio_model.PeopleDefinition
OSPeople = openstudio_model.People
OSLightsDefinition = openstudio_model.LightsDefinition
OSLights = openstudio_model.Lights
OSElectricEquipmentDefinition = openstudio_model.ElectricEquipmentDefinition
OSElectricEquipment = openstudio_model.ElectricEquipment
OSGasEquipmentDefinition = openstudio_model.GasEquipmentDefinition
OSGasEquipment = openstudio_model.GasEquipment
OSOtherEquipmentDefinition = openstudio_model.OtherEquipmentDefinition
OSOtherEquipment = openstudio_model.OtherEquipment
OSWaterUseEquipmentDefinition = openstudio_model.WaterUseEquipmentDefinition
OSWaterUseEquipment = openstudio_model.WaterUseEquipment
OSWaterUseConnections = openstudio_model.WaterUseConnections
OSSpaceInfiltrationDesignFlowRate = openstudio_model.SpaceInfiltrationDesignFlowRate
OSDesignSpecificationOutdoorAir = openstudio_model.DesignSpecificationOutdoorAir
OSThermostatSetpointDualSetpoint = openstudio_model.ThermostatSetpointDualSetpoint
OSZoneControlHumidistat = openstudio_model.ZoneControlHumidistat
OSDaylightingControl = openstudio_model.DaylightingControl
# ventilative cooling and AFN classes
OSZoneVentilationWindandStackOpenArea = openstudio_model.ZoneVentilationWindandStackOpenArea
OSZoneVentilationDesignFlowRate = openstudio_model.ZoneVentilationDesignFlowRate
OSAirflowNetworkReferenceCrackConditions = openstudio_model.AirflowNetworkReferenceCrackConditions
OSAirflowNetworkCrack = openstudio_model.AirflowNetworkCrack
OSAirflowNetworkSimpleOpening = openstudio_model.AirflowNetworkSimpleOpening
OSAirflowNetworkHorizontalOpening = openstudio_model.AirflowNetworkHorizontalOpening
# shw System classes
OSPlantLoop = openstudio_model.PlantLoop
OSSetpointManagerScheduled = openstudio_model.SetpointManagerScheduled
OSPumpConstantSpeed = openstudio_model.PumpConstantSpeed
OSWaterHeaterMixed = openstudio_model.WaterHeaterMixed
OSCoilWaterHeatingAirToWaterHeatPump = openstudio_model.CoilWaterHeatingAirToWaterHeatPump
OSFanOnOff = openstudio_model.FanOnOff
OSWaterHeaterHeatPump = openstudio_model.WaterHeaterHeatPump
# HVAC classes
OSZoneHVACIdealLoadsAirSystem = openstudio_model.ZoneHVACIdealLoadsAirSystem
# ems classes
OSOutputVariable = openstudio_model.OutputVariable
OSEnergyManagementSystemProgram = openstudio_model.EnergyManagementSystemProgram
OSEnergyManagementSystemProgramCallingManager = \
    openstudio_model.EnergyManagementSystemProgramCallingManager
OSEnergyManagementSystemSensor = openstudio_model.EnergyManagementSystemSensor
OSEnergyManagementSystemActuator = openstudio_model.EnergyManagementSystemActuator
OSEnergyManagementSystemConstructionIndexVariable = \
    openstudio_model.EnergyManagementSystemConstructionIndexVariable
# generator classes
OSGeneratorPVWatts = openstudio_model.GeneratorPVWatts
OSElectricLoadCenterDistribution = openstudio_model.ElectricLoadCenterDistribution
OSElectricLoadCenterInverterPVWatts = openstudio_model.ElectricLoadCenterInverterPVWatts
# simulation classes
OSRunPeriodControlSpecialDays = openstudio_model.RunPeriodControlSpecialDays
OSMonthOfYear = openstudio.MonthOfYear
OSOutputVariable = openstudio_model.OutputVariable
OSDesignDay = openstudio_model.DesignDay
