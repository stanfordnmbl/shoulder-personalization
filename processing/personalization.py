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

    # left vs. right
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
    # me_g = tools.vec3_to_np(markerset.get(marker_names[lr + '_elbow_medial']).getLocationInGround(state))
    # le_g = tools.vec3_to_np(markerset.get(marker_names[lr + '_elbow_lateral']).getLocationInGround(state))

    # get joint and marker locations in global frame
    sc_g = tools.vec3_to_np(model.getJointSet().get('sternoclavicular_' + lr).getParentFrame().
                            findStationLocationInGround(state, osim.Vec3([0, 0, 0])))
    ac_g = tools.vec3_to_np(model.getJointSet().get('unrotscap_' + lr).getParentFrame().
                            findStationLocationInGround(state, osim.Vec3([0, 0, 0])))
    # ghr_g = tools.vec3_to_np(model.getJointSet().get('unrothum_' + lr).getParentFrame().
    # findStationLocationInGround(state, osim.Vec3([0, 0, 0])))

    # define body frames from model markers in ground frame
    t_t = cs.compute_body_coordsys_thorax(sternum_g, c7_g, px_g, t8_g)
    # t_c = cs.compute_body_coordsys_clavicle(sc_g, ac_g, t_t[0:3, 1])
    t_s = cs.compute_body_coordsys_scapula(aa_g, ai_g, ts_g)
    # t_h = cs.compute_body_coordsys_humerus(ghr_g, le_g, me_g)

    # get model markers in marker-based thorax frame
    sc_t = np.matmul(np.linalg.inv(t_t), np.append(sc_g, 1))
    ac_s = np.matmul(np.linalg.inv(t_s), np.append(ac_g, 1))

    # define joint coordinate frames
    t_ac = cs.compute_joint_coordsys_ac(t_s[0:3, 2], t_t[0: 3, 1], ac_g, left_right)

    # get joint rotations from model coordinate axes(Spatial transform)
    sc_joint = osim.CustomJoint.safeDownCast(model.getJointSet().get('sternoclavicular_' + lr))
    sc_joint_transform = sc_joint.getSpatialTransform()
    rot1_sc_saul = tools.vec3_to_np(sc_joint_transform.get_rotation1().get_axis(0))
    rot2_sc_saul = tools.vec3_to_np(sc_joint_transform.get_rotation2().get_axis(0))
    rot3_sc_saul = tools.vec3_to_np(sc_joint_transform.get_rotation3().get_axis(0))
    r_c_saul = np.column_stack([rot1_sc_saul, rot2_sc_saul, rot3_sc_saul])

    ac_joint = osim.CustomJoint.safeDownCast(model.getJointSet().get('acromioclavicular_' + lr))
    ac_joint_transform = ac_joint.getSpatialTransform()
    rot1_ac_saul = tools.vec3_to_np(ac_joint_transform.get_rotation1().get_axis(0))
    rot2_ac_saul = tools.vec3_to_np(ac_joint_transform.get_rotation2().get_axis(0))
    rot3_ac_saul = tools.vec3_to_np(ac_joint_transform.get_rotation3().get_axis(0))
    r_s_saul = np.column_stack([rot1_ac_saul, rot2_ac_saul, rot3_ac_saul])

    gh0_joint = osim.CustomJoint.safeDownCast(model.getJointSet().get('shoulder0_' + lr))
    gh0_joint_transform = gh0_joint.getSpatialTransform()
    rot1_s0_saul = tools.vec3_to_np(gh0_joint_transform.get_rotation1().get_axis(0))

    gh1_joint = osim.CustomJoint.safeDownCast(model.getJointSet().get('shoulder1_' + lr))
    gh1_joint_transform = gh1_joint.getSpatialTransform()
    rot1_s1_saul = tools.vec3_to_np(gh1_joint_transform.get_rotation1().get_axis(0))
    # rot2_s1_saul = tools.vec3_to_np(gh1_joint_transform.get_rotation2().get_axis(0))

    gh2_joint = osim.CustomJoint.safeDownCast(model.getJointSet().get('shoulder2_' + lr))
    gh2_joint_transform = gh2_joint.getSpatialTransform()
    rot1_s2_saul = tools.vec3_to_np(gh2_joint_transform.get_rotation1().get_axis(0))

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
        t_s_score = score.compute_score_coordsys_scapula(aa_meas, ai_meas, ts_meas)
        ghr_humerus_estimate = score.estimate_ghr_score(t_h_score[0:3, 0:3], t_h_score[0:3, 3], g_humerus)
        ghr_scapula_estimate = score.estimate_ghr_score(t_s_score[0:3, 0:3], t_s_score[0:3, 3], g_scapula)
        ghr_meas = ghr_humerus_estimate

        # humerus
        # t_h_meas = cs.compute_body_coordsys_humerus(ghr_meas, le_meas, me_meas)

        ################################################################################################################
        # compute joint coordinates from marker data #
        ################################################################################################################

        # get clavicle protraction and elevation
        r_c_saul_meas = r_c_saul

        v_clav_meas = ac_meas[0:3] - sc_meas[0:3]
        v_clav_meas = v_clav_meas/np.linalg.norm(v_clav_meas)

        # sc_r2
        v_clav_meas_x = tools.orthogonal_projection(np.zeros([1, 3]), r_c_saul_meas[:, 0], v_clav_meas)
        v_clav_meas_x = v_clav_meas_x/np.linalg.norm(v_clav_meas_x)
        v_clav_meas_x_0deg = tools.orthogonal_projection(np.zeros([1, 3]), r_c_saul_meas[:, 0], r_c_saul_meas[:, 2])
        v_clav_meas_x_0deg = v_clav_meas_x_0deg/np.linalg.norm(v_clav_meas_x_0deg)

        if 'right' in left_right:
            clav_prot_meas = np.real(np.arccos(np.dot(v_clav_meas_x, v_clav_meas_x_0deg)))
            if np.dot(tools.xprod(v_clav_meas_x, v_clav_meas_x_0deg), r_c_saul_meas[:, 0]) > 0:
                clav_prot_meas = -clav_prot_meas
        else:
            clav_prot_meas = np.real(np.arccos(np.dot(v_clav_meas_x, -v_clav_meas_x_0deg)))
            if np.dot(tools.xprod(v_clav_meas_x, v_clav_meas_x_0deg), r_c_saul_meas[:, 0]) < 0:
                clav_prot_meas = -clav_prot_meas

        coordinates[i, 0] = clav_prot_meas

        # sc_r3
        # rotate R_c_saul by clav_prot around clav_prot axis
        axis_c_meas = r_c_saul_meas[:, 0]
        angle_c_meas = clav_prot_meas
        vec_c_meas = angle_c_meas * axis_c_meas
        qrv = Rotation.from_rotvec(vec_c_meas)
        r_c_meas = qrv.as_matrix()
        r_c_saul_meas_rotated = np.matmul(r_c_meas, r_c_saul_meas)

        v_clav_meas_y = tools.orthogonal_projection(np.zeros([1, 3]), r_c_saul_meas_rotated[:, 1], v_clav_meas)
        v_clav_meas_y = v_clav_meas_y/np.linalg.norm(v_clav_meas_y)
        if 'right' in left_right:
            clav_elev_meas = np.real(np.arccos(np.dot(v_clav_meas_y, r_c_saul_meas_rotated[:, 2])))
            if np.dot(tools.xprod(v_clav_meas_y, r_c_saul_meas_rotated[:, 2]), r_c_saul_meas_rotated[:, 1]) > 0:
                clav_elev_meas = -clav_elev_meas
        else:
            clav_elev_meas = np.real(np.arccos(np.dot(v_clav_meas_y, -r_c_saul_meas_rotated[:, 2])))
            if np.dot(tools.xprod(v_clav_meas_y, r_c_saul_meas_rotated[:, 2]), r_c_saul_meas_rotated[:, 1]) < 0:
                clav_elev_meas = -clav_elev_meas

        coordinates[i, 1] = clav_elev_meas

        # get scapula r1, r2, r3
        r_s_saul_meas = r_s_saul

        v_scap_meas = aa_meas - ts_meas
        v_scap_meas = v_scap_meas/np.linalg.norm(v_scap_meas)

        # scapula angle 1
        v_scap_meas_1 = tools.orthogonal_projection(np.zeros([1, 3]), r_s_saul_meas[:, 0], v_scap_meas)
        v_scap_meas_1 = v_scap_meas_1/np.linalg.norm(v_scap_meas_1)

        v_scap_meas_1_0deg = tools.orthogonal_projection(np.zeros([1, 3]), r_s_saul_meas[:, 0], r_s_saul_meas[:, 2])
        v_scap_meas_1_0deg = v_scap_meas_1_0deg/np.linalg.norm(v_scap_meas_1_0deg)

        if 'right' in left_right:
            angle_s1_meas = np.real(np.arccos(np.dot(v_scap_meas_1, v_scap_meas_1_0deg)))
            if np.dot(tools.xprod(v_scap_meas_1, v_scap_meas_1_0deg), r_s_saul_meas[:, 0]) > 0:
                angle_s1_meas = -angle_s1_meas
        else:
            angle_s1_meas = np.real(np.arccos(np.dot(v_scap_meas_1, -v_scap_meas_1_0deg)))
            if np.dot(tools.xprod(v_scap_meas_1, v_scap_meas_1_0deg), r_s_saul_meas[:, 0]) < 0:
                angle_s1_meas = -angle_s1_meas

        # scapula angle 2
        # rotate R_s_saul by angle_s1 around first rotation axis
        axis_s1_meas = r_s_saul_meas[:, 0]
        vec_s1_meas = angle_s1_meas * axis_s1_meas
        qrv = Rotation.from_rotvec(vec_s1_meas)
        r_s1_meas = qrv.as_matrix()
        r_s_saul_meas_rotated1 = np.matmul(r_s1_meas, r_s_saul_meas)

        # compute angle
        v_scap_meas_2 = tools.orthogonal_projection(np.zeros([1, 3]), r_s_saul_meas_rotated1[:, 1], v_scap_meas)
        v_scap_meas_2 = v_scap_meas_2/np.linalg.norm(v_scap_meas_2)

        v_scap_meas_2_0deg = tools.orthogonal_projection(np.zeros([1, 3]), r_s_saul_meas_rotated1[:, 1],
                                                         r_s_saul_meas_rotated1[:, 2])
        v_scap_meas_2_0deg = v_scap_meas_2_0deg/np.linalg.norm(v_scap_meas_2_0deg)

        if 'right' in left_right:
            angle_s2_meas = np.real(np.arccos(np.dot(v_scap_meas_2, v_scap_meas_2_0deg)))
            if np.dot(tools.xprod(v_scap_meas_2, v_scap_meas_2_0deg), r_s_saul_meas_rotated1[:, 1]) > 0:
                angle_s2_meas = -angle_s2_meas
        else:
            angle_s2_meas = np.real(np.arccos(np.dot(v_scap_meas_2, -v_scap_meas_2_0deg)))
            if np.dot(tools.xprod(v_scap_meas_2, v_scap_meas_2_0deg), r_s_saul_meas_rotated1[:, 1]) < 0:
                angle_s2_meas = -angle_s2_meas

        # scapula angle 3
        # rotate R_s_saul_rotated1 by angle_s2 around second rotated rotation axis
        axis_s2_meas = r_s_saul_meas_rotated1[:, 1]
        vec_s2_meas = angle_s2_meas * axis_s2_meas
        qrv = Rotation.from_rotvec(vec_s2_meas)
        r_s2_meas = qrv.as_matrix()
        r_s_saul_meas_rotated2 = np.matmul(r_s2_meas, r_s_saul_meas_rotated1)

        # compute angle
        # transform rot2_ac_s to global frame using measurement scapula frame to get "measurement" of rot2_ac
        rot2_ac_s = np.matmul(np.transpose(t_s[0: 3, 0: 3]), t_ac[0:3, 1])
        rot2_ac_meas = np.matmul(t_s_meas[0:3, 0:3], rot2_ac_s)

        v_scap_meas_3 = tools.orthogonal_projection(np.zeros([1, 3]), r_s_saul_meas_rotated2[:, 2], rot2_ac_meas)
        v_scap_meas_3 = v_scap_meas_3/np.linalg.norm(v_scap_meas_3)

        v_scap_meas_3_0deg = tools.orthogonal_projection(np.zeros([1, 3]), r_s_saul_meas_rotated2[:, 2],
                                                         r_s_saul_meas_rotated2[:, 1])
        v_scap_meas_3_0deg = v_scap_meas_3_0deg/np.linalg.norm(v_scap_meas_3_0deg)

        if 'right' in left_right:
            angle_s3_meas = np.real(np.arccos(np.dot(v_scap_meas_3, v_scap_meas_3_0deg)))
        else:
            angle_s3_meas = np.pi - np.real(np.arccos(np.dot(v_scap_meas_3, -v_scap_meas_3_0deg)))

        coordinates[i, 2:5] = np.array([angle_s3_meas, angle_s1_meas, angle_s2_meas])

        # get humerus plane and elevation
        v_hum_meas = ghr_meas - 0.5 * (me_meas + le_meas)
        v_hum_meas = v_hum_meas/np.linalg.norm(v_hum_meas)

        if 'left' in left_right:
            v_hum_meas = -v_hum_meas

        # optimization to obtain glenohumeral angles (elv_angle and shoulder_elv)
        # process: apply rotations defined by coordinate systems to zero-position humerus vector and find glenohumeral
        # angles by minimizing the distance between the rotated vector and the actual measured humerus vector

        # rotation axes of glenohumeral joints: u for elv_plane, m for shoulder_elv
        u_meas = sympy.Matrix(rot1_s0_saul)
        m_meas = sympy.Matrix(rot1_s1_saul)

        # default and target humerus vectors. Default from orientation of the shoulder_rot axis, target from measurement
        v0_meas = sympy.Matrix(rot1_s2_saul)
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
        coordinates[i, 5] = elv_angle_meas
        coordinates[i, 6] = shoulder_elv_meas

    return coordinates


def unconstrain_clav_elev_add_ac_marker(path_model_in, path_model_out):

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
