import opensim as osim
import subprocess
from processing import coordinate_systems as cs
from processing import tools
import numpy as np


def scale_model(path_scale_setup, path_model_in, path_scalingfile, path_model_out, hum_scalefactor, time_range):
    """
    Scaling pipeline for the saul model.
    1. Use OpenSim ScaleTool and anatomical pose
    2. Change the humerus scaling to manual scales calculated based on all five calibration poses

    Parameters:
        path_scale_setup: Full filepath to the generic scale setup .xml-file
        path_model_in: Full filepath to the unscaled OpenSim model
        path_scalingfile: Full filepath to the marker file in anatomical pose
        path_model_out: Full filepath to which the scaled OpenSim model will be saved
        hum_scalefactor: Pre-computed humerus scale factors (left and right)
        time_range: list containing initial and final time to be used for scaling
    """

    # scale model: load and modify generic scale setup, save and run cmd
    scale_tool = osim.ScaleTool(path_scale_setup)

    # generic model
    scale_tool.getGenericModelMaker().setModelFileName(path_model_in)

    # output filenames
    scale_tool.getModelScaler().setOutputModelFileName(path_model_out)
    scale_tool.getMarkerPlacer().setOutputModelFileName(path_model_out)
    scale_tool.getMarkerPlacer().setOutputMotionFileName(path_model_out.replace('.osim', '_scale.mot'))

    # markerfile for scaling
    scale_tool.getModelScaler().setMarkerFileName(path_scalingfile)
    scale_tool.getMarkerPlacer().setMarkerFileName(path_scalingfile)

    # set time range
    start_end = osim.ArrayDouble()
    start_end.append(time_range[0])
    start_end.append(time_range[1])
    scale_tool.getModelScaler().setTimeRange(start_end)
    scale_tool.getMarkerPlacer().setTimeRange(start_end)

    # use pre-computed manual scales for humerus
    for lr in ['r', 'l']:
        for m, measurement in enumerate(scale_tool.getModelScaler().getMeasurementSet()):
            if 'hum' in measurement.getName():
                scale_tool.getModelScaler().getMeasurementSet().get(m).setApply(False)
        for h in ['humerus_', 'humphant_', 'humphant1_']:
            sc = osim.Scale()
            sc.setApply(True)
            sc.setSegmentName(h + lr)
            sc.setScaleFactors(osim.Vec3(hum_scalefactor[lr][0], hum_scalefactor[lr][1], hum_scalefactor[lr][2]))
            scale_tool.getModelScaler().addScale(sc)

    sc_order = osim.ArrayStr()
    sc_order.append('measurements')
    sc_order.append('manualScale')
    scale_tool.getModelScaler().setScalingOrder(sc_order)

    # save modified scale settings file
    path_scale_setup_out = path_model_out.replace('.osim', '_scalefile.xml')
    scale_tool.printToXML(path_scale_setup_out)

    # run modified file
    subprocess.run(['opensim-cmd', 'run-tool', path_scale_setup_out])


def correct_tracking_markers(path_model_scaled, path_scalingfile, mot_scaling, marker_names):
    """
    Correct the location of the tracking markers on the upper arms and forearms that might have been moved to faulty
    positions within the OpenSim scaling because of the highlty constrained shoulder definition. This function uses the
    positions of the acromion, elbow, and tracking markers in the data to define relative coordinate systems and move
    the tracking markers in the model to the correct location.

    Parameters:
        path_model_scaled: Full filepath to the scaled model
        path_scalingfile: Full filepath to the marker file in anatomical pose
        mot_scaling: motion file created during OpenSim scaling
        marker_names: dict of marker names
    """

    # load model and load scaling motion
    model = osim.Model(path_model_scaled)
    state = model.initSystem()

    storage = osim.Storage(mot_scaling)

    # put model in position
    t = 0
    motdata = storage.getStateVector(t).getData()
    coord_counter = 0

    for i in range(model.getNumJoints()):

        joint_name = model.getJointSet().get(i).getName()

        num_coords = model.getJointSet().get(i).getNumStateVariables() / 2
        for c in range(int(num_coords)):
            j = storage.getStateIndex(
                '/jointset/' + joint_name + '/' + model.getJointSet().get(i).get_coordinates(c).getName() + '/value')
            v = motdata.get(j)
            model.updCoordinateSet().get(coord_counter).setValue(state, v, False)
            coord_counter = coord_counter + 1

    model.realizePosition(state)
    model.assemble(state)

    # load data
    data = tools.read_trc(path_scalingfile)
    data = tools.average_data_nan(data)

    # correct humerus markers
    for lr in ['r', 'l']:
        # compute data coordinate system (use AMC2 instead of ghr)
        t_data = cs.compute_body_coordsys_humerus(data[marker_names[lr + '_AMC2']],
                                                  data[marker_names[lr + '_elbow_lateral']],
                                                  data[marker_names[lr + '_elbow_medial']])

        # transform global humerus marker to local coordinate system
        hum_data_l = np.matmul(np.linalg.inv(t_data), np.append(data[marker_names[lr + '_humerous']], 1))

        # compute model coordinate system (AMC2 instead of ghr)
        markerset = model.getMarkerSet()
        me_model = tools.vec3_to_np(markerset.get(marker_names[lr + '_elbow_medial']).getLocationInGround(state))
        le_model = tools.vec3_to_np(markerset.get(marker_names[lr + '_elbow_lateral']).getLocationInGround(state))
        amc2_model = tools.vec3_to_np(markerset.get(marker_names[lr + '_AMC2']).getLocationInGround(state))
        t_model = cs.compute_body_coordsys_humerus(amc2_model, le_model, me_model)

        # inverse transform humerus marker to global coordinate system
        hum_model_g = np.matmul(t_model, hum_data_l)

        # set new marker coordinates
        hum_model_g_vec = osim.Vec3(hum_model_g[0], hum_model_g[1], hum_model_g[2])
        humerus_body = model.getBodySet().get('humerus_' + lr)
        new_location = model.getGround().findStationLocationInAnotherFrame(state, hum_model_g_vec, humerus_body)

        markerset.get(marker_names[lr + '_humerous']).set_location(new_location)

        # correct forearm markers

        # compute data coordinate system
        t_data = cs.compute_body_coordsys_forearm(data[marker_names[lr + '_elbow_lateral']],
                                                  data[marker_names[lr + '_elbow_medial']],
                                                  data[marker_names[lr + '_wrist_ulna']],
                                                  data[marker_names[lr + '_wrist_radius']])

        # transform global forearm marker to local coordinate system
        fore_data_l = np.matmul(np.linalg.inv(t_data), np.append(data[marker_names[lr + '_forearm']], 1))

        # compute model coordinate system
        ulna_model = tools.vec3_to_np(markerset.get(marker_names[lr + '_wrist_ulna']).getLocationInGround(state))
        radius_model = tools.vec3_to_np(markerset.get(marker_names[lr + '_wrist_radius']).getLocationInGround(state))
        t_model = cs.compute_body_coordsys_forearm(le_model, me_model, ulna_model, radius_model)

        # inverse transform humerus marker to global coordinate system
        fore_model_g = np.matmul(t_model, fore_data_l)

        # set new marker coordinates
        fore_model_g_vec = osim.Vec3([fore_model_g[0], fore_model_g[1], fore_model_g[2]])
        forearm_body = model.getBodySet().get('radius_' + lr)
        new_location = model.getGround().findStationLocationInAnotherFrame(state, fore_model_g_vec, forearm_body)

        markerset.get(marker_names[lr + '_forearm']).set_location(new_location)

    model.finalizeConnections()
    model.printToXML(path_model_scaled)
