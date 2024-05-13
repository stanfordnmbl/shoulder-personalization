import numpy as np
from processing import tools
import opensim as osim


def compute_body_coordsys_thorax(sternum, c7, px, t8):
    """
    Thorax body coordinate system:
    - origin in sternal notch
    - y_t: line connecting the midpoint between PX and T8 and the midpoint between IJ and C7, pointing upward
    - z_t: perpendicular to the plane formed by IJ, C7, and the midpoint between PX and T8, pointing to the right
    - x_t: perpendicular to the z-and y-axis, pointing forwards

    Parameters:
        sternum: 3D location of marker at sternal notch
        c7: 3D location of marker at c7 vertebra
        px: 3D location of marker at xiphoid process
        t8: 3D location of marker at t8 vertebra
    """
    o = sternum
    y = 0.5*(sternum + c7) - 0.5*(px + t8)
    y = y/np.linalg.norm(y)
    z = tools.xprod(0.5*(px + t8) - c7, sternum - c7)
    z = z/np.linalg.norm(z)
    x = tools.xprod(y, z)
    x = x/np.linalg.norm(x)

    t_t = np.row_stack([np.column_stack([x, y, z, o]), [0, 0, 0, 1]])
    return t_t


def compute_body_coordsys_clavicle(sc, ac, y_t):
    """
    Clavicle body coordinate system:
    - origin in SC joint center
    - z_c: line connecting SC and AC, pointing to AC
    - x_c: line perpendicular to z_c and y_t, pointing forward
    - y_c: perpendicular to the x_c- and z_c-axis, pointing upward

    Parameters:
        sc: 3D location of the sternoclavicular joint rotation center
        ac: 3D location of the acromioclavicular joint rotation center
        y_t: y-axis of the thorax coordinate system
    """
    o = sc
    z = ac - sc
    z = z/np.linalg.norm(z)
    x = tools.xprod(y_t, z)
    x = x/np.linalg.norm(x)
    y = tools.xprod(z, x)
    y = y/np.linalg.norm(y)

    t_c = np.row_stack([np.column_stack([x, y, z, o]), [0, 0, 0, 1]])
    return t_c


def compute_body_coordsys_scapula(aa, ai, ts):
    """
    Scapula body coordinate system:
    - origin in aa
    - z_s: line connecting TS and AA, pointing to AA
    - x_s: line perpendicular to the plane formed by AI, AA, and TS, pointing forward
    - y_s: perpendicular to the x_s- and z_s-axis, pointing upward

    Parameters:
        aa: 3D location of marker at acromial angle of the scapula
        ai: 3D location of marker at inferior angle of the scapula
        ts: 3D location of marker at root of the spine of the scapula
    """
    o = aa
    z = aa - ts
    z = z/np.linalg.norm(z)
    x = tools.xprod(aa - ts, ai - ts)
    x = x/np.linalg.norm(x)
    y = tools.xprod(z, x)
    y = y/np.linalg.norm(y)

    t_s = np.row_stack([np.column_stack([x, y, z, o]), [0, 0, 0, 1]])
    return t_s


def compute_body_coordsys_humerus(ghr, le, me):
    """
    Humerus body coordinate system:
    - origin in ghr
    - y_h: line connecting ghr and the midpoint of le and me, pointing to ghr
    - x_h: line perpendicular to the plane formed by le, me, and ghr, pointing forward
    - z_h: perpendicular to the y_h- and z_h-axis, pointing to the right

    Parameters:
        ghr: 3D location of glenohumeral rotation center
        le: 3D location of marker at lateral epicondyle of the elbow
        me: 3D location of marker at medial epicondyle of the elbow
    """
    o = ghr
    y = ghr - 0.5 * (le + me)
    y = y/np.linalg.norm(y)
    x = tools.xprod(le - ghr, me - ghr)
    x = x/np.linalg.norm(x)
    z = tools.xprod(x, y)
    z = z/np.linalg.norm(z)

    t_h = np.row_stack([np.column_stack([x, y, z, o]), [0, 0, 0, 1]])
    return t_h


def compute_joint_coordsys_sc(sc, z_c, y_t, left_right):
    """
    Sternoclavicular joint coordinate system (YXZ order):
    - origin in sc
    - rot1: coincident with y_t-axis of the thorax coordinate system (-retraction, +protraction)
    - rot3: coincident with z_c-axis of the clavicle coordinate system (+rotation of the top backwards, -forwards)
    - rot2: perpendicular to rot1 and rot3 (+elevation, -depression, which is opposite ISB recs)

    Parameters:
        sc: 3D location of the sternoclavicular joint rotation center
        z_c: z-axis of the clavicle body coordinate system
        y_t: y-axis of the thorax body coordiante system
        left_right: indicates left or right shoulder
    """
    rot1_sc = y_t
    rot3_sc = z_c
    # invert axes in case of left shoulder
    if 'left' in left_right:
        rot1_sc = -rot1_sc
        rot3_sc = -rot3_sc
    rot2_sc = tools.xprod(rot3_sc, rot1_sc)
    rot2_sc = rot2_sc/np.linalg.norm(rot2_sc)

    t_sc = np.row_stack([np.column_stack([rot1_sc, rot2_sc, rot3_sc, sc]), [0, 0, 0, 1]])
    return t_sc


def compute_joint_coordsys_ac(z_s, y_t, ac, left_right):
    """
    Acromioclavicular joint coordinate system (YXZ order):
    - origin in ac
    - rot1: coincident with the y_t-axis of the thorax coordinate system (-retraction, +protraction)
    - rot3: coincident with the z_s-axis of the scapular coordinate system (-anterior, +posterior tilt)
    - rot2: perpendicular to rot1 and rot3 (-lateral, +medial rotation)

    Parameters:
        z_s: z-axis of the scapula body coordinate system
        y_t: y-axis of the thorax body coordiante system
        ac: 3D location of the acromioclavicular joint rotation center
        left_right: indicates left or right shoulder
    """
    rot1_ac = y_t
    rot3_ac = z_s
    # invert axes in case of left shoulder
    if 'left' in left_right:
        rot1_ac = -rot1_ac
        rot3_ac = -rot3_ac
    rot2_ac = tools.xprod(rot3_ac, rot1_ac)  # -lateral rot, +medial rot
    rot2_ac = rot2_ac/np.linalg.norm(rot2_ac)

    t_ac = np.row_stack([np.column_stack([rot1_ac, rot2_ac, rot3_ac, ac]), [0, 0, 0, 1]])
    return t_ac


def compute_joint_coordsys_gh(y_h, y_t, x_h, ghr, left_right):
    """
    Glenohumeral joint coordinate system (YXY order):
    - origin in ghr
    - rot1: coincident with the y_t-axis of the thorax coordinate system
      (plane of elevation, 0 is abduction, 90 is forward flexion)
    - rot3: coincident with y_h-axis of the humerus body coordinate system
      (axial rotation, +internal rotation, -external-rotation)
    - rot2: coincident with the x-h-axis of the humerus coordinate system
      (elevation angle, +elevation, -depression (opposite ISB recs))

    Parameters:
        y_h: z-axis of the humerus body coordinate system
        y_t: y-axis of the thorax body coordiante system
        x_h: x-axis of the humerus body coordinate system
        ghr: 3D location of the glenohumeral joint rotation center
        left_right: indicates left or right shoulder
    """
    rot1_gh = y_t
    rot3_gh = y_h
    # invert axes in case of left shoulder
    if 'left' in left_right:
        rot1_gh = -rot1_gh
        rot3_gh = -rot3_gh
    rot2_gh = -x_h

    t_gh = np.row_stack([np.column_stack([rot1_gh, rot2_gh, rot3_gh, ghr]), [0, 0, 0, 1]])
    return t_gh


def compute_body_coordsys_forearm(le, me, ulna, radius):
    """
    Forearm body coordinate system (not following ISB recommendation, only used for tracking marker position correction)
    - origin in midpoint between le and me
    - y_f: line connecting the midpoint of le and me and the midpoint of ulna and radius
    - x_f: line perpendicular to plane formed by the midpoint of le and me
    - z_f: perpendicular to the x_f- and z_f-axis

    Parameters:
        le: 3D location of marker at the lateral epicondyle of the elbow
        me: 3D location marker at the medial epicondyle of the elbow
        ulna: 3D location of marker at ulnar styloid
        radius: 3D location of marker at radial styloid
    """
    o = 0.5 * (le + me)
    y = o - 0.5 * (ulna + radius)
    y = y / np.linalg.norm(y)
    x = tools.xprod(ulna - o, radius - o)
    x = x / np.linalg.norm(x)
    z = tools.xprod(x, y)
    z = z / np.linalg.norm(z)

    t_h = np.row_stack([np.column_stack([x, y, z, o]), [0, 0, 0, 1]])
    return t_h


def update_isb_coords(model_filepath):
    """
    Update the coordinate systems after OpenSim model scaling using the virtual marker locations of the scaled model.

    Parameters:
        model_filepath: Full filepath to the scaled model
    """
    # load model
    model = osim.Model(model_filepath)
    state = model.initSystem()

    # update right and left shoulder coordinate systems
    for left_right in ['right', 'left']:
        if 'right' in left_right:
            lr = 'r'
        else:
            lr = 'l'

        # get joint locations in ground frame
        sc_g = tools.vec3_to_np(model.getJointSet().get('sternoclavicular_' + lr).getParentFrame().
                                findStationLocationInGround(state, osim.Vec3([0, 0, 0])))
        ac_g = tools.vec3_to_np(model.getJointSet().get('unrotscap_' + lr).getParentFrame().
                                findStationLocationInGround(state, osim.Vec3([0, 0, 0])))
        ghr_g = tools.vec3_to_np(model.getJointSet().get('unrothum_' + lr).getParentFrame().
                                 findStationLocationInGround(state, osim.Vec3([0, 0, 0])))

        # get model marker locations in ground frame
        markerset = model.getMarkerSet()
        aa_g = tools.vec3_to_np(markerset.get('aa_' + lr).getLocationInGround(state))
        ai_g = tools.vec3_to_np(markerset.get('ai_' + lr).getLocationInGround(state))
        ts_g = tools.vec3_to_np(markerset.get('ts_' + lr).getLocationInGround(state))
        sternum_g = tools.vec3_to_np(markerset.get('Sternum').getLocationInGround(state))
        px_g = tools.vec3_to_np(markerset.get('PX').getLocationInGround(state))
        c7_g = tools.vec3_to_np(markerset.get('C7').getLocationInGround(state))
        t8_g = tools.vec3_to_np(markerset.get('T8').getLocationInGround(state))
        me_g = tools.vec3_to_np(markerset.get(lr + '_elbow_medial').getLocationInGround(state))
        le_g = tools.vec3_to_np(markerset.get(lr + '_elbow_lateral').getLocationInGround(state))

        # define body frames from model markers in ground frame
        t_t = compute_body_coordsys_thorax(sternum_g, c7_g, px_g, t8_g)
        t_c = compute_body_coordsys_clavicle(sc_g, ac_g, t_t[0:3, 1])
        t_s = compute_body_coordsys_scapula(aa_g, ai_g, ts_g)
        t_h = compute_body_coordsys_humerus(ghr_g, le_g, me_g)

        # define joint frames from model markers in ground frame
        t_sc = compute_joint_coordsys_sc(sc_g, t_c[0:3, 2], t_t[0:3, 1], left_right)
        t_ac = compute_joint_coordsys_ac(t_s[0:3, 2], t_t[0:3, 1], ac_g, left_right)
        t_gh = compute_joint_coordsys_gh(t_h[0:3, 1], t_t[0:3, 1], t_h[0:3, 0], ghr_g, left_right)

        # update joint spatial transforms
        sc_joint = osim.CustomJoint.safeDownCast(model.getJointSet().get('sternoclavicular_' + lr))
        sc_joint_transform = sc_joint.getSpatialTransform()
        sc_joint_transform.get_rotation1().set_axis(osim.Vec3([t_sc[0, 0], t_sc[1, 0], t_sc[2, 0]]))
        sc_joint_transform.get_rotation2().set_axis(osim.Vec3([t_sc[0, 1], t_sc[1, 1], t_sc[2, 1]]))
        sc_joint_transform.get_rotation3().set_axis(osim.Vec3([t_sc[0, 2], t_sc[1, 2], t_sc[2, 2]]))

        sc_unrot_joint = osim.CustomJoint.safeDownCast(model.getJointSet().get('unrotscap_' + lr))
        sc_unrot_joint_transform = sc_unrot_joint.getSpatialTransform()
        sc_unrot_joint_transform.get_rotation1().set_axis(osim.Vec3([t_sc[0, 1], t_sc[1, 1], t_sc[2, 1]]))
        sc_unrot_joint_transform.get_rotation2().set_axis(osim.Vec3([t_sc[0, 0], t_sc[1, 0], t_sc[2, 0]]))
        sc_unrot_joint_transform.get_rotation3().set_axis(osim.Vec3([t_sc[0, 2], t_sc[1, 2], t_sc[2, 2]]))

        ac_joint = osim.CustomJoint.safeDownCast(model.getJointSet().get('acromioclavicular_' + lr))
        ac_joint_transform = ac_joint.getSpatialTransform()
        ac_joint_transform.get_rotation1().set_axis(osim.Vec3([t_ac[0, 0], t_ac[1, 0], t_ac[2, 0]]))
        ac_joint_transform.get_rotation2().set_axis(osim.Vec3([t_ac[0, 1], t_ac[1, 1], t_ac[2, 1]]))
        ac_joint_transform.get_rotation3().set_axis(osim.Vec3([t_ac[0, 2], t_ac[1, 2], t_ac[2, 2]]))

        ac_unrot_joint = osim.CustomJoint.safeDownCast(model.getJointSet().get('unrothum_' + lr))
        ac_unrot_joint_transform = ac_unrot_joint.getSpatialTransform()
        ac_unrot_joint_transform.get_rotation1().set_axis(osim.Vec3([t_ac[0, 2], t_ac[1, 2], t_ac[2, 2]]))
        ac_unrot_joint_transform.get_rotation2().set_axis(osim.Vec3([t_ac[0, 1], t_ac[1, 1], t_ac[2, 1]]))
        ac_unrot_joint_transform.get_rotation3().set_axis(osim.Vec3([t_ac[0, 0], t_ac[1, 0], t_ac[2, 0]]))

        shoulder0_joint = osim.CustomJoint.safeDownCast(model.getJointSet().get('shoulder0_' + lr))
        shoulder0_joint_transform = shoulder0_joint.getSpatialTransform()
        shoulder0_joint_transform.get_rotation1().set_axis(osim.Vec3([t_gh[0, 0], t_gh[1, 0], t_gh[2, 0]]))

        shoulder1_joint = osim.CustomJoint.safeDownCast(model.getJointSet().get('shoulder1_' + lr))
        shoulder1_joint_transform = shoulder1_joint.getSpatialTransform()
        shoulder1_joint_transform.get_rotation1().set_axis(osim.Vec3([t_gh[0, 1], t_gh[1, 1], t_gh[2, 1]]))
        shoulder1_joint_transform.get_rotation2().set_axis(osim.Vec3([t_gh[0, 0], t_gh[1, 0], t_gh[2, 0]]))

        shoulder2_joint = osim.CustomJoint.safeDownCast(model.getJointSet().get('shoulder2_' + lr))
        shoulder2_joint_transform = shoulder2_joint.getSpatialTransform()
        shoulder2_joint_transform.get_rotation1().set_axis(osim.Vec3([t_gh[0, 2], t_gh[1, 2], t_gh[2, 2]]))

        # save model
        model.printToXML(model_filepath)
