from processing import tools
import opensim as osim
from processing import coordinate_systems as cs
import numpy as np
from processing import score
import os
import sympy
from scipy.spatial.transform import Rotation
from scipy import optimize


def compute_constraint_angles(path_model, folder_trials, filenames_trials, left_right,
                              g_humerus, g_scapula, marker_names,
                              move_scapula_markers_to_skin=False, scapula_distance=0, marker_names_scapula_tool=None):
    """
    This method uses the marker data and scaled generic model to compute the sternoclavicular, acromioclavicular, and
    glenohumeral joint angles for all calibration poses.

    Parameters:
        path_model: Full filepath to the scaled model
        folder_trials: file folder in which the calibration trials are stored
        filenames_trials: list of filenames of calibration trials
        left_right: right or left shoulder
        g_humerus: glenohumeral joint center calibrated using SCoRE
        g_scapula: acromioclavicular joint center calibrated using SCoRE
        marker_names: dict for marker names in the trc files as defined in settings file
        move_scapula_markers_to_skin: flag indicating if scapula markers should be moved from measuring device onto
                                      skin. default=False
        scapula_distance: distance of markers on scapula measuring device to skin (in case scapula markers should be
                           moved). default=0
        marker_names_scapula_tool: names of the markers on measuring device as defined in settings file. default=None
    Returns:
        coordinates: num_trials x 7 matrix, each row contains sc protraction/retraction, sc elevation/depression,
                     ac tilt angle, ac protraction/retraction, ac upward/downward rotation,
                     gh elevation plane, gh elevation
    """
    # left or right shoulder
    if 'right' in left_right:
        lr = 'r'
    else:
        lr = 'l'

    ####################################################################################################################
    # Get joint centers from model #
    ####################################################################################################################
    # load model
    model = osim.Model(path_model)
    state = model.initSystem()

    # get marker positions in ground frame
    markerset = model.getMarkerSet()

    aa_g = tools.vec3_to_np(markerset.get(marker_names['aa_' + lr]).getLocationInGround(state))
    ai_g = tools.vec3_to_np(markerset.get(marker_names['ai_' + lr]).getLocationInGround(state))
    ts_g = tools.vec3_to_np(markerset.get(marker_names['ts_' + lr]).getLocationInGround(state))
    sternum_g = tools.vec3_to_np(markerset.get(marker_names['Sternum']).getLocationInGround(state))
    px_g = tools.vec3_to_np(markerset.get(marker_names['PX']).getLocationInGround(state))
    c7_g = tools.vec3_to_np(markerset.get(marker_names['C7']).getLocationInGround(state))
    t8_g = tools.vec3_to_np(markerset.get(marker_names['T8']).getLocationInGround(state))

    # get joint and marker locations in global frame
    sc_g = tools.vec3_to_np(model.getJointSet().get('sternoclavicular_' + lr).getParentFrame().
                            findStationLocationInGround(state, osim.Vec3([0, 0, 0])))
    ac_g = tools.vec3_to_np(model.getJointSet().get('unrotscap_' + lr).getParentFrame().
                            findStationLocationInGround(state, osim.Vec3([0, 0, 0])))

    # define body frames from model markers in ground frame
    t_t = cs.compute_body_coordsys_thorax(sternum_g, c7_g, px_g, t8_g)
    t_s = cs.compute_body_coordsys_scapula(aa_g, ai_g, ts_g)

    # get joint rotation centers in local thorax and scapula frame
    sc_t = np.matmul(np.linalg.inv(t_t), np.append(sc_g, 1))
    ac_s = np.matmul(np.linalg.inv(t_s), np.append(ac_g, 1))

    # define joint coordinate frame
    t_ac = cs.compute_joint_coordsys_ac(t_s[0:3, 2], t_t[0: 3, 1], ac_g, left_right)

    # get joint rotations from model coordinate axes (Spatial transform)
    sc_joint = osim.CustomJoint.safeDownCast(model.getJointSet().get('sternoclavicular_' + lr))
    sc_joint_transform = sc_joint.getSpatialTransform()
    rot1_sc_model = tools.vec3_to_np(sc_joint_transform.get_rotation1().get_axis(0))
    rot2_sc_model = tools.vec3_to_np(sc_joint_transform.get_rotation2().get_axis(0))
    rot3_sc_model = tools.vec3_to_np(sc_joint_transform.get_rotation3().get_axis(0))
    r_c_model = np.column_stack([rot1_sc_model, rot2_sc_model, rot3_sc_model])

    ac_joint = osim.CustomJoint.safeDownCast(model.getJointSet().get('acromioclavicular_' + lr))
    ac_joint_transform = ac_joint.getSpatialTransform()
    rot1_ac_model = tools.vec3_to_np(ac_joint_transform.get_rotation1().get_axis(0))
    rot2_ac_model = tools.vec3_to_np(ac_joint_transform.get_rotation2().get_axis(0))
    rot3_ac_model = tools.vec3_to_np(ac_joint_transform.get_rotation3().get_axis(0))
    r_s_model = np.column_stack([rot1_ac_model, rot2_ac_model, rot3_ac_model])

    gh0_joint = osim.CustomJoint.safeDownCast(model.getJointSet().get('shoulder0_' + lr))
    gh0_joint_transform = gh0_joint.getSpatialTransform()
    rot1_s0_model = tools.vec3_to_np(gh0_joint_transform.get_rotation1().get_axis(0))

    gh1_joint = osim.CustomJoint.safeDownCast(model.getJointSet().get('shoulder1_' + lr))
    gh1_joint_transform = gh1_joint.getSpatialTransform()
    rot1_s1_model = tools.vec3_to_np(gh1_joint_transform.get_rotation1().get_axis(0))

    gh2_joint = osim.CustomJoint.safeDownCast(model.getJointSet().get('shoulder2_' + lr))
    gh2_joint_transform = gh2_joint.getSpatialTransform()
    rot1_s2_model = tools.vec3_to_np(gh2_joint_transform.get_rotation1().get_axis(0))

    ####################################################################################################################
    # load data and register to model #
    ####################################################################################################################
    coordinates = np.zeros([len(filenames_trials), 7])

    for i, filename in enumerate(filenames_trials):

        # load data
        trial = tools.read_trc(str(os.path.join(folder_trials, filename)))
        trial = tools.average_data_nan(trial)
        if move_scapula_markers_to_skin:
            trial = tools.move_scapula_markers_onto_skin(trial, marker_names, left_right,
                                                         scapula_distance, marker_names_scapula_tool)

        # register to model default pose
        [r, t] = tools.rigid_transform_3d(
            np.array([trial[marker_names['C7']], trial[marker_names['Sternum']],
                      trial[marker_names['PX']], trial[marker_names['T8']]]),
            np.array([c7_g, sternum_g, px_g, t8_g]))

        c7_meas = np.matmul(r, trial[marker_names['C7']]) + t
        sternum_meas = np.matmul(r, trial[marker_names['Sternum']]) + t
        px_meas = np.matmul(r, trial[marker_names['PX']]) + t
        t8_meas = np.matmul(r, trial[marker_names['T8']]) + t
        me_meas = np.matmul(r, trial[marker_names[lr + '_elbow_medial']]) + t
        le_meas = np.matmul(r, trial[marker_names[lr + '_elbow_lateral']]) + t
        hum_meas = np.matmul(r, trial[marker_names[lr + '_humerous']]) + t
        ts_meas = np.matmul(r, trial[marker_names['ts_' + lr]]) + t
        ai_meas = np.matmul(r, trial[marker_names['ai_' + lr]]) + t
        aa_meas = np.matmul(r, trial[marker_names['aa_' + lr]]) + t

        ################################################################################################################
        # define frames from measurement #
        ################################################################################################################

        # thorax
        t_t_meas = cs.compute_body_coordsys_thorax(sternum_meas, c7_meas, px_meas, t8_meas)
        sc_meas = np.matmul(t_t_meas, sc_t)

        # scapula
        t_s_meas = cs.compute_body_coordsys_scapula(aa_meas, ai_meas, ts_meas)
        ac_meas = np.matmul(t_s_meas, ac_s)

        # ghr estimation using SCoRE
        t_h_score = score.compute_score_coordsys_humerus(hum_meas, le_meas, me_meas)
        ghr_meas = score.estimate_ghr_score(t_h_score[0:3, 0:3], t_h_score[0:3, 3], g_humerus)

        ################################################################################################################
        # compute joint coordinates from marker data #
        ################################################################################################################

        ##########################
        # sternoclavicular joint #
        ##########################
        r_c_meas = r_c_model

        # define vector along clavicle long axis
        v_clav_meas = ac_meas[0:3] - sc_meas[0:3]
        v_clav_meas = v_clav_meas/np.linalg.norm(v_clav_meas)

        # 1. clavicle protraction
        # project clavicle vector along clavicle protraction axis (r_c_meas[:, 0])
        v_clav_meas_x = tools.orthogonal_projection(np.zeros([1, 3]), r_c_meas[:, 0], v_clav_meas)
        v_clav_meas_x = v_clav_meas_x/np.linalg.norm(v_clav_meas_x)
        # project 0 degree clavicle protraction vector (r_c_meas[:,2]) along clavicle protraction axis (r_c_meas[:, 0])
        v_clav_meas_x_0deg = tools.orthogonal_projection(np.zeros([1, 3]), r_c_meas[:, 0], r_c_meas[:, 2])
        v_clav_meas_x_0deg = v_clav_meas_x_0deg/np.linalg.norm(v_clav_meas_x_0deg)

        # compute angle between the two vectors depending on right/left shoulder
        if 'right' in left_right:
            clav_prot_meas = np.real(np.arccos(np.dot(v_clav_meas_x, v_clav_meas_x_0deg)))
            if np.dot(tools.xprod(v_clav_meas_x, v_clav_meas_x_0deg), r_c_meas[:, 0]) > 0:
                clav_prot_meas = -clav_prot_meas
        else:
            clav_prot_meas = np.real(np.arccos(np.dot(v_clav_meas_x, -v_clav_meas_x_0deg)))
            if np.dot(tools.xprod(v_clav_meas_x, v_clav_meas_x_0deg), r_c_meas[:, 0]) < 0:
                clav_prot_meas = -clav_prot_meas

        # save measured coordinate
        coordinates[i, 0] = clav_prot_meas

        # 2. clavicle elevation
        # rotate sc frame by clavicle protraction angle around the clavicle protraction axis (r_c_meas[:, 0])
        axis_c_meas = r_c_meas[:, 0]
        angle_c_meas = clav_prot_meas
        vec_c_meas = angle_c_meas * axis_c_meas
        quaternion_c_meas = Rotation.from_rotvec(vec_c_meas)
        rotation_c_meas = quaternion_c_meas.as_matrix()
        r_c_meas_rotated = np.matmul(rotation_c_meas, r_c_meas)

        # project clavicle vector along rotated clavicle elevation axis (r_c_meas_rotated[:, 1])
        v_clav_meas_y = tools.orthogonal_projection(np.zeros([1, 3]), r_c_meas_rotated[:, 1], v_clav_meas)
        v_clav_meas_y = v_clav_meas_y/np.linalg.norm(v_clav_meas_y)
        # no need to project clavicle protraction vector (r_c_meas[:,2]) along clavicle elevation axis (r_c_meas[:, 1])
        # as they are orthogonal by definition
        v_clav_meas_y_0deg = r_c_meas_rotated[:, 2]

        # compute angle between the two vectors depending on right/left shoulder
        if 'right' in left_right:
            clav_elev_meas = np.real(np.arccos(np.dot(v_clav_meas_y, v_clav_meas_y_0deg)))
            if np.dot(tools.xprod(v_clav_meas_y, v_clav_meas_y_0deg), r_c_meas_rotated[:, 1]) > 0:
                clav_elev_meas = -clav_elev_meas
        else:
            clav_elev_meas = np.real(np.arccos(np.dot(v_clav_meas_y, -v_clav_meas_y_0deg)))
            if np.dot(tools.xprod(v_clav_meas_y, v_clav_meas_y_0deg), r_c_meas_rotated[:, 1]) < 0:
                clav_elev_meas = -clav_elev_meas

        # save measured coordinate
        coordinates[i, 1] = clav_elev_meas

        ###########################
        # acromioclavicular joint #
        ###########################
        r_s_meas = r_s_model

        # define vector along scapular spine
        v_scap_meas = aa_meas - ts_meas
        v_scap_meas = v_scap_meas/np.linalg.norm(v_scap_meas)

        # 1. scapula protraction/retraction (rot1)
        # project scapular spine vector along protraction/retraction axis (r_s_meas[:, 0])
        v_scap_meas_1 = tools.orthogonal_projection(np.zeros([1, 3]), r_s_meas[:, 0], v_scap_meas)
        v_scap_meas_1 = v_scap_meas_1/np.linalg.norm(v_scap_meas_1)
        # project 0 degree scapula protraction vector (r_s_meas[:,2]) along scapula protraction axis (r_s_meas[:, 0])
        v_scap_meas_1_0deg = tools.orthogonal_projection(np.zeros([1, 3]), r_s_meas[:, 0], r_s_meas[:, 2])
        v_scap_meas_1_0deg = v_scap_meas_1_0deg/np.linalg.norm(v_scap_meas_1_0deg)

        # compute angle between the two vectors depending on right/left shoulder
        if 'right' in left_right:
            angle_s1_meas = np.real(np.arccos(np.dot(v_scap_meas_1, v_scap_meas_1_0deg)))
            if np.dot(tools.xprod(v_scap_meas_1, v_scap_meas_1_0deg), r_s_meas[:, 0]) > 0:
                angle_s1_meas = -angle_s1_meas
        else:
            angle_s1_meas = np.real(np.arccos(np.dot(v_scap_meas_1, -v_scap_meas_1_0deg)))
            if np.dot(tools.xprod(v_scap_meas_1, v_scap_meas_1_0deg), r_s_meas[:, 0]) < 0:
                angle_s1_meas = -angle_s1_meas

        # 2. scapula upward/downward rotation (rot2)
        # rotate ac frame by scapula protraction angle around the scapula protraction axis (r_s_meas[:, 0])
        axis_s1_meas = r_s_meas[:, 0]
        vec_s1_meas = angle_s1_meas * axis_s1_meas
        quaternion_s1_meas = Rotation.from_rotvec(vec_s1_meas)
        rotation_s1_meas = quaternion_s1_meas.as_matrix()
        r_s_meas_rotated1 = np.matmul(rotation_s1_meas, r_s_meas)

        # project scapular spine vector along upward/downward rotation axis (r_s_meas_rotated1[:, 1])
        v_scap_meas_2 = tools.orthogonal_projection(np.zeros([1, 3]), r_s_meas_rotated1[:, 1], v_scap_meas)
        v_scap_meas_2 = v_scap_meas_2/np.linalg.norm(v_scap_meas_2)
        # project 0 degree scapula upward/downward rotation vector (r_s_meas_rotated1[:,2]) along scapula upward/down-
        # ward rotation axis (r_s_meas_rotated1[:, 1])
        v_scap_meas_2_0deg = tools.orthogonal_projection(np.zeros([1, 3]), r_s_meas_rotated1[:, 1],
                                                         r_s_meas_rotated1[:, 2])
        v_scap_meas_2_0deg = v_scap_meas_2_0deg/np.linalg.norm(v_scap_meas_2_0deg)

        # compute angle between the two vectors depending on right/left shoulder
        if 'right' in left_right:
            angle_s2_meas = np.real(np.arccos(np.dot(v_scap_meas_2, v_scap_meas_2_0deg)))
            if np.dot(tools.xprod(v_scap_meas_2, v_scap_meas_2_0deg), r_s_meas_rotated1[:, 1]) > 0:
                angle_s2_meas = -angle_s2_meas
        else:
            angle_s2_meas = np.real(np.arccos(np.dot(v_scap_meas_2, -v_scap_meas_2_0deg)))
            if np.dot(tools.xprod(v_scap_meas_2, v_scap_meas_2_0deg), r_s_meas_rotated1[:, 1]) < 0:
                angle_s2_meas = -angle_s2_meas

        # 3. scapula tilt (rot3)
        # rotate rotated ac frame by scapula upward/downward rotation angle around the scapula upward/downward rotation
        # axis (r_s_meas_rotated1[:, 1])
        axis_s2_meas = r_s_meas_rotated1[:, 1]
        vec_s2_meas = angle_s2_meas * axis_s2_meas
        quaternion_s2_meas = Rotation.from_rotvec(vec_s2_meas)
        rotation_s2_meas = quaternion_s2_meas.as_matrix()
        r_s_meas_rotated2 = np.matmul(rotation_s2_meas, r_s_meas_rotated1)

        # this rotation is around the scapular spine, so we need to define new vector to project along the rotation axis
        # (r_s_meas_rotated2[:, 2]) and measure angle with 0 degree vector. We use the 2nd ac axis for this.
        # transform rot2_ac_s to global frame using measurement scapula frame to get "measurement" of rot2_ac
        # express global 2nd ac axis in local scapula coordinate system (both from model in default pose)
        rot2_ac_s = np.matmul(np.transpose(t_s[0: 3, 0: 3]), t_ac[0:3, 1])
        # transform local rot2_ac to current global position using measured scapula coordinate system (in current pose)
        rot2_ac_meas = np.matmul(t_s_meas[0:3, 0:3], rot2_ac_s)

        # project new scapula vector (rot2_ac_meas) along tilt axis (r_s_meas_rotated2[:, 2])
        v_scap_meas_3 = tools.orthogonal_projection(np.zeros([1, 3]), r_s_meas_rotated2[:, 2], rot2_ac_meas)
        v_scap_meas_3 = v_scap_meas_3/np.linalg.norm(v_scap_meas_3)
        # project 0 degree scapula tilt vector (r_s_meas_rotated2[:, 1]) along scapula tilt axis
        # (r_s_meas_rotated2[:, 2])
        v_scap_meas_3_0deg = tools.orthogonal_projection(np.zeros([1, 3]), r_s_meas_rotated2[:, 2],
                                                         r_s_meas_rotated2[:, 1])
        v_scap_meas_3_0deg = v_scap_meas_3_0deg/np.linalg.norm(v_scap_meas_3_0deg)

        # compute angle between the two vectors depending on right/left shoulder
        if 'right' in left_right:
            angle_s3_meas = np.real(np.arccos(np.dot(v_scap_meas_3, v_scap_meas_3_0deg)))
        else:
            angle_s3_meas = np.pi - np.real(np.arccos(np.dot(v_scap_meas_3, -v_scap_meas_3_0deg)))

        # save measured coordinates in their rotation order
        coordinates[i, 2:5] = np.array([angle_s3_meas, angle_s1_meas, angle_s2_meas])

        ######################
        # glenohumeral joint #
        ######################
        # define vector along long axis of humerus
        v_hum_meas = ghr_meas - 0.5 * (me_meas + le_meas)
        v_hum_meas = v_hum_meas/np.linalg.norm(v_hum_meas)

        # invert if left arm
        if 'left' in left_right:
            v_hum_meas = -v_hum_meas

        # YXY rotation order, joint definition, and unrotation constraints complicate the direct calculation of the
        # glenohumeral angles (elv_angle and shoulder_elv).
        # We instead apply the rotations defined by coordinate systems to zero-position humerus vector using the two
        # glenohumeral angles as unknowns. We then find the glenohumeral angles by minimizing the distance between the
        # rotated vector and the actual measured humerus vector.

        # rotation axes of glenohumeral joints: u for elv_plane, m for shoulder_elv
        u_meas = sympy.Matrix(rot1_s0_model)
        m_meas = sympy.Matrix(rot1_s1_model)

        # default and target humerus vectors. Default from orientation of the shoulder_rot axis, target from measurement
        v0_meas = sympy.Matrix(rot1_s2_model)
        v_meas = sympy.Matrix(v_hum_meas)

        # unknown angles a = elv_plane, b = shoulder_elv
        a_meas, b_meas = sympy.symbols('a_meas b_meas')

        # 1. elv_plane rotation
        # v1 = rotate v0 by a around u
        v1_meas = tools.rot_vec_around_axis_by_angle(v0_meas, u_meas, a_meas)

        # 2. shoulder_elv rotation (around rotated axis)
        # axis m1 = rotate shoulder_elv axis m by a around u
        m1_meas = tools.rot_vec_around_axis_by_angle(m_meas, u_meas, a_meas)

        # v2 = rotate v1 by b around m1
        v2_meas = tools.rot_vec_around_axis_by_angle(v1_meas, m1_meas, b_meas)

        # 3. apply implemented constraint shoulder1_r2 = -a
        # axis u1 = rotate (u rotated by a around u) by b around m1 = rotate u by b around m1
        u1_meas = tools.rot_vec_around_axis_by_angle(u_meas, m1_meas, b_meas)

        # rotate v2 by -a around u1
        v3_meas = tools.rot_vec_around_axis_by_angle(v2_meas, u1_meas, -a_meas)

        # objective: find a and b that minimize difference between the rotated vector and the measured humerus vector
        v_target = v3_meas - v_meas
        f_meas = sympy.sqrt(v_target.dot(v_target))

        # bounds for angles from coordinates
        a_bounds = [gh0_joint.getCoordinate(0).getRangeMin(), gh0_joint.getCoordinate(0).getRangeMax()]
        b_bounds = [gh1_joint.getCoordinate(0).getRangeMin(), gh1_joint.getCoordinate(0).getRangeMax()]
        bounds = [a_bounds, b_bounds]

        # initial guess in the middle of range
        x0_meas = np.array([np.mean(a_bounds), np.mean(b_bounds)])

        # optimize and store
        solution = optimize.minimize(tools.func_sym_to_float, x0_meas, bounds=bounds, args=(f_meas, a_meas, b_meas))
        elv_angle_meas = solution.x[0]
        shoulder_elv_meas = solution.x[1]

        # save measured coordinates
        coordinates[i, 5] = elv_angle_meas
        coordinates[i, 6] = shoulder_elv_meas

    return coordinates


def unconstrain_clav_elev_add_ac_marker(path_model_in, path_model_out):
    """
    This method uses the marker data and scaled generic model to compute the sternoclavicular, acromioclavicular, and
    glenohumeral joint angles for all calibration poses.

    Parameters:
        path_model_in: Full filepath to the scaled and personalized model
        path_model_out: Full filepath to which the scaled and personalized model with removed clavicle elevation
                        constraint will be stored
    """
    # load model
    model = osim.Model(path_model_in)
    model.initSystem()

    # add marker at ac l and r
    # ac_r marker
    m = osim.Marker()
    m.setParentFrame(model.getBodySet().get('clavicle_r'))
    m.set_location(model.getJointSet().get('unrotscap_r').getParentFrame().findTransformInBaseFrame().T())
    m.setName('ac_r')
    model.addMarker(m)

    # ac_r marker
    m2 = osim.Marker()
    m2.setParentFrame(model.getBodySet().get('clavicle_l'))
    m2.set_location(model.getJointSet().get('unrotscap_l').getParentFrame().findTransformInBaseFrame().T())
    m2.setName('ac_l')
    model.addMarker(m2)

    # turn off clavicle elevation constraints l and r and change range
    model.getJointSet().get('sternoclavicular_r').get_coordinates(1).setRangeMin(-0.4363)
    model.getJointSet().get('sternoclavicular_r').get_coordinates(1).setRangeMax(0.5236)
    model.getJointSet().get('sternoclavicular_r').get_coordinates(1).set_clamped(True)
    model.getConstraintSet().get('sternoclavicular_r3_con_r').set_isEnforced(False)

    model.getJointSet().get('sternoclavicular_l').get_coordinates(1).setRangeMin(-0.4363)
    model.getJointSet().get('sternoclavicular_l').get_coordinates(1).setRangeMax(0.5236)
    model.getJointSet().get('sternoclavicular_l').get_coordinates(1).set_clamped(True)
    model.getConstraintSet().get('sternoclavicular_r3_con_l').set_isEnforced(False)

    model.finalizeConnections()

    # save model
    model.printToXML(path_model_out)


def extend_coordinate_ranges(model):
    """
    This method extends the coordinate ranges for the groundthorax joint, the shoulder elevation plane angle, and the
    shoulder humerus internal/external rotation (e.g. for an overhead sports analysis).

    Parameters:
        model: OpenSim shoulder model the default ranges should be extended for
    """
    model.getJointSet().get('groundthorax').get_coordinates(0).setRangeMin(-np.pi / 2)
    model.getJointSet().get('groundthorax').get_coordinates(0).setRangeMax(np.pi / 2)
    model.getJointSet().get('groundthorax').get_coordinates(1).setRangeMin(-np.pi / 2)
    model.getJointSet().get('groundthorax').get_coordinates(1).setRangeMax(np.pi / 2)
    model.getJointSet().get('groundthorax').get_coordinates(2).setRangeMin(-np.pi / 2)
    model.getJointSet().get('groundthorax').get_coordinates(2).setRangeMax(np.pi / 2)

    for lr in ['r', 'l']:

        model.getJointSet().get('shoulder0_' + lr).get_coordinates(0).setRangeMin(-np.pi)
        model.getJointSet().get('shoulder0_' + lr).get_coordinates(0).setRangeMax(np.pi)

        model.getJointSet().get('shoulder2_' + lr).get_coordinates(0).setRangeMin(-np.pi)
        model.getJointSet().get('shoulder2_' + lr).get_coordinates(0).setRangeMax(np.pi)
