import numpy as np
from processing import tools
import opensim as osim


def compute_body_coordsys_thorax(sternum, c7, px, t8):
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
    # SC joint: YXZ
    rot1_sc = y_t  # +protraction, -retraction
    rot3_sc = z_c  # axial rotation
    if 'left' in left_right:
        rot1_sc = -rot1_sc
        rot3_sc = -rot3_sc
    rot2_sc = tools.xprod(rot3_sc, rot1_sc)  # -depression, +elevation (opposite ISB recs)
    rot2_sc = rot2_sc/np.linalg.norm(rot2_sc)

    t_sc = np.row_stack([np.column_stack([rot1_sc, rot2_sc, rot3_sc, sc]), [0, 0, 0, 1]])
    return t_sc


def compute_joint_coordsys_ac(z_s, y_t, ac, left_right):

    # AC joint: YXZ
    rot1_ac = y_t  # -retraction, +protracion
    rot3_ac = z_s  # -anterior tilt, +posterior tilt
    if 'left' in left_right:
        rot1_ac = -rot1_ac
        rot3_ac = -rot3_ac
    rot2_ac = tools.xprod(rot3_ac, rot1_ac)  # -lateral rot, +medial rot
    rot2_ac = rot2_ac/np.linalg.norm(rot2_ac)

    t_ac = np.row_stack([np.column_stack([rot1_ac, rot2_ac, rot3_ac, ac]), [0, 0, 0, 1]])
    return t_ac


def compute_joint_coordsys_gh(y_h, y_t, x_h, ghr, left_right):

    # GH joint: YXY
    rot1_gh = y_t  # elv_plane
    rot3_gh = y_h  # +internal rot, -external rot
    if 'left' in left_right:
        rot1_gh = -rot1_gh
        rot3_gh = -rot3_gh
    rot2_gh = -x_h  # +elevation (opposite ISB recs)

    t_gh = np.row_stack([np.column_stack([rot1_gh, rot2_gh, rot3_gh, ghr]), [0, 0, 0, 1]])
    return t_gh


def compute_body_coordsys_forearm(le, me, ulna, radius):
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
    # load model
    model = osim.Model(model_filepath)
    state = model.initSystem()

    for left_right in ['right', 'left']:
        if 'right' in left_right:
            lr = 'r'
        else:
            lr = 'l'

        # get joint and marker locations in global frame
        sc_g = tools.vec3_to_np(model.getJointSet().get('sternoclavicular_' + lr).getParentFrame().
                                findStationLocationInGround(state, osim.Vec3([0, 0, 0])))
        ac_g = tools.vec3_to_np(model.getJointSet().get('unrotscap_' + lr).getParentFrame().
                                findStationLocationInGround(state, osim.Vec3([0, 0, 0])))
        ghr_g = tools.vec3_to_np(model.getJointSet().get('unrothum_' + lr).getParentFrame().
                                 findStationLocationInGround(state, osim.Vec3([0, 0, 0])))

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
