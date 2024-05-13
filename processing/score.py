import numpy as np
from processing import tools
import opensim as osim
import os


def compute_score_coordsys_humerus(humerus, elbow_lateral, elbow_medial):
    """
    Define humerus coordinate system used for SCoRE ghr estimation method

    Parameters:
        humerus: 3D location of the tracking marker on the upper arm
        elbow_lateral: 3D location of the marker on the lateral epicondyle of the elbow
        elbow_medial: 3D location of the marker on the medial epicondyle of the elbow
    """
    o = 0.5 * (elbow_lateral + elbow_medial)
    x = elbow_medial - elbow_lateral
    x = x/np.linalg.norm(x)
    y = humerus - o
    y = y/np.linalg.norm(y)
    z = tools.xprod(x, y)
    z = z/np.linalg.norm(z)

    t_h = np.row_stack([np.column_stack([x, y, z, o]), [0, 0, 0, 1]])
    return t_h


def compute_score_coordsys_scapula(aa, ai, ts):
    """
    Define scapula coordinate system used for SCoRE ghr estimation method

    Parameters:
        aa: 3D location of marker at acromial angle of the scapula
        ai: 3D location of marker at inferior angle of the scapula
        ts: 3D location of marker at root of the spine of the scapula
    """
    o = aa
    z = aa - ts
    z = z / np.linalg.norm(z)

    x = tools.xprod(ts - aa, ai - aa)
    x = x / np.linalg.norm(x)

    y = tools.xprod(z, x)
    y = y / np.linalg.norm(y)

    t_s = np.row_stack([np.column_stack([x, y, z, o]), [0, 0, 0, 1]])
    return t_s


def calibrate_score(trc_data_folder: str, filenames_calibration: list[str], left_right: str, marker_names,
                    move_scapula_markers_to_skin=False, scapula_distance=0,
                    marker_names_scapula_tool=None):
    """
    Define scapula coordinate system used for SCoRE ghr estimation method

    Parameters:
        trc_data_folder: folder in which the calibration .trc marker files are stored
        filenames_calibration: list of the filenames of the files used for calibration
        left_right: right or left shoulder
        marker_names: dict for marker names in the trc files as defined in settings file
        move_scapula_markers_to_skin: flag indicating if scapula markers should be moved from measuring device onto
                                      skin. default=False
        scapula_distance: distance of markers on scapula measuring device to skin (in case scapula markers should be
                           moved). default=0
        marker_names_scapula_tool: names of the markers on measuring device as defined in settings file. default=None
    Returns:
        g_humerus: glenohumeral joint center calibrated using SCoRE
        g_scapula: acromioclavicular joint center calibrated using SCoRE
    """
    if 'right' in left_right:
        lr = 'r'
    else:
        lr = 'l'

    # implementation following the publication:
    # Monnet, T., Desailly, E., Begon, M., Valleé, C., & Lacouture, P. (2007). Comparison of the SCoRE and HA methods
    # for locating in vivo the glenohumeral joint centre. Journal of Biomechanics, 40, 3487–3492.
    # https://doi.org/10.1016/j.jbiomech.2007.05.030

    # compute the b and c matrix using marker data from calibration files
    b = np.zeros([3 * len(filenames_calibration), 6])
    c = np.zeros([3 * len(filenames_calibration), 1])

    for i, filename in enumerate(filenames_calibration):
        data_dict = tools.read_trc(os.path.join(trc_data_folder, filename))
        data = tools.average_data_nan(data_dict)
        if move_scapula_markers_to_skin:
            data = tools.move_scapula_markers_onto_skin(data, marker_names, left_right,
                                                        scapula_distance, marker_names_scapula_tool)

        t_h = compute_score_coordsys_humerus(data[marker_names[lr + '_humerous']],
                                             data[marker_names[lr + '_elbow_lateral']],
                                             data[marker_names[lr + '_elbow_medial']])
        r_h = t_h[0:3, 0:3]
        p_h = t_h[0:3, 3]

        t_s = compute_score_coordsys_scapula(data[marker_names['aa_' + lr]],
                                             data[marker_names['ai_' + lr]],
                                             data[marker_names['ts_' + lr]])
        r_s = t_s[0:3, 0:3]
        p_s = t_s[0:3, 3]

        b[3 * i: 3 * (i + 1), 0:3] = r_h
        b[3 * i: 3 * (i + 1), 3:6] = -r_s

        c[3 * i: 3 * (i + 1), 0] = p_s - p_h

    # compute vectors v and u that describe the position of the ghr in the local humerus and scapula coordinate system
    vu = np.matmul(np.matmul(np.linalg.inv(np.matmul(np.transpose(b), b)), np.transpose(b)), c)
    g_humerus = vu[0:3]
    g_scapula = vu[3:6]

    return [g_humerus, g_scapula]


def estimate_ghr_score(r, p, v):
    """
    transform ghr in local coordinate system to global coordinate system using current pose (humerus or scapula)

    Parameters:
        r: rotation matrix of current humerus/scapula coordinate system in global frame
        p: origin of current humerus/scapula coordinate system in global frame
        v: calibrated ghr in local humerus/scapula coordinate system
    Returns:
        ghr: current estimated position of ghr in global frame
    """
    ghr = np.squeeze(np.matmul(r, v)) + p
    return ghr


def compute_humerus_scale_factor(path_model, folder, filenames, left_right, g_humerus, marker_names,
                                 move_scapula_markers_to_skin=False, scapula_distance=0,
                                 marker_names_scapula_tool=None):
    """
    use all calibration poses and SCoRE ghr estimateion to update the humerus scaling factor for model scaling

    Parameters:
        path_model: Full filepath to the OpenSim unscaled model
        folder: folder in which the .trc marker files are stored
        filenames: list of the filenames of the files used for computation
        left_right: right or left shoulder
        g_humerus: location of ghr in local humerus coordinate system
        marker_names: dict for marker names in the trc files as defined in settings file
        move_scapula_markers_to_skin: flag indicating if scapula markers should be moved from measuring device onto
                                      skin. default=False
        scapula_distance: distance of markers on scapula measuring device to skin (in case scapula markers should be
                           moved). default=0
        marker_names_scapula_tool: names of the markers on measuring device as defined in settings file. default=None
    Returns:
        sf: the three scale factors for the humerus (width used for both dimensions perpendicular to long axis)
    """
    if 'right' in left_right:
        lr = 'r'
    else:
        lr = 'l'

    # compute humerus length for each of the calibration poses
    hum_lengths = np.zeros([len(filenames), 1])
    hum_widths = np.zeros([len(filenames), 1])

    for i, filename in enumerate(filenames):
        # load data
        trial = tools.read_trc(str(os.path.join(folder, filename)))
        trial = tools.average_data_nan(trial)
        # move markers from scapula measurement tool onto skin
        if move_scapula_markers_to_skin:
            trial = tools.move_scapula_markers_onto_skin(trial, marker_names, left_right,
                                                         scapula_distance, marker_names_scapula_tool)
        # get from data: markers at medial and lateral epicondyle and tracking marker on upper arm
        me_meas = trial[marker_names[lr + '_elbow_medial']]
        le_meas = trial[marker_names[lr + '_elbow_lateral']]
        hum_meas = trial[marker_names[lr + '_humerous']]

        # ghr estimation using SCoRE: compute humerus coord system and use it to transform local ghr to global ghr
        t_h_score = compute_score_coordsys_humerus(hum_meas, le_meas, me_meas)
        ghr_humerus_estimate = estimate_ghr_score(t_h_score[0:3, 0:3], t_h_score[0:3, 3], g_humerus)

        # use estimate of ghr and markers on elbow to compute estimated humerus length for this trial
        v_hum_meas = ghr_humerus_estimate - 0.5 * (me_meas + le_meas)
        hum_lengths[i] = np.linalg.norm(v_hum_meas)
        hum_widths[i] = np.linalg.norm(me_meas - le_meas)

    # average estimated humerus length and width over all calibration trials
    hum_length_meas = np.mean(hum_lengths)
    hum_width_meas = np.mean(hum_widths)

    # load unscaled model
    model = osim.Model(path_model)
    state = model.initSystem()

    # compute unscaled humerus length from model markers and glenohumeral joint rotation center
    markerset = model.getMarkerSet()
    me_g = tools.vec3_to_np(markerset.get(marker_names[lr + '_elbow_medial']).getLocationInGround(state))
    le_g = tools.vec3_to_np(markerset.get(marker_names[lr + '_elbow_lateral']).getLocationInGround(state))
    ghr_g = tools.vec3_to_np(model.getJointSet().get('unrothum_' + lr).getParentFrame().
                             findStationLocationInGround(state, osim.Vec3([0, 0, 0])))

    v_hum_model = ghr_g - 0.5 * (me_g + le_g)
    hum_length_model = np.linalg.norm(v_hum_model)
    hum_width_model = np.linalg.norm(me_g - le_g)

    # scale factor is measured / model
    sf = [hum_width_meas / hum_width_model, hum_length_meas / hum_length_model, hum_width_meas / hum_width_model]

    return sf
