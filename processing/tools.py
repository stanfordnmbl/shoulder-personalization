import csv
import numpy as np
import opensim as osim
import sympy


def vec3_to_np(vec3):
    return np.array([vec3[0], vec3[1], vec3[2]])


def read_trc(filepath: str):
    with open(filepath, "r", encoding="utf8") as trc_file:
        trc_reader = csv.reader(trc_file, delimiter="\t")

        # Skip the first row
        next(trc_reader)

        # create dict with keys and values from header
        keys = next(trc_reader)
        vals = next(trc_reader)
        header_info = dict(zip(keys, [i.strip() for i in vals]))

        marker_names = next(trc_reader)

        # skip X1 Y1 Z1 X2 Y2 Z2 ... line
        next(trc_reader)

        # read all data
        data_list = list(trc_reader)
        # remove empty starting line if present
        if len(data_list[0]) == 0 or not str.isdigit(data_list[0][0]):
            data_list = data_list[1:]
        data_matrix = np.array(data_list)

        data_dict = {marker_names[0]: data_matrix[:, 0], marker_names[1]: data_matrix[:, 1]}
        for idx, marker_name in enumerate(marker_names[2:]):
            if marker_name:
                data_dict[marker_name] = data_matrix[:, 2 + idx:2 + idx + 3].astype(np.float_)

    return data_dict


def average_data_nan(data_dict):
    fns = data_dict.keys()
    data_average = dict()
    for fn in fns:
        if 'Frame' not in fn and 'Time' not in fn:
            data_average[fn] = np.nanmean(data_dict[fn], 0)/1000
    return data_average


def move_scapula_markers_onto_skin(data, marker_names, left_right, scap_dist, marker_names_scapula_tool):

    if 'right' in left_right:
        n_scap = xprod(data[marker_names_scapula_tool['AI']] - data[marker_names_scapula_tool['TS']],
                       data[marker_names_scapula_tool['AA']] - data[marker_names_scapula_tool['TS']])
        lr = 'r'
    else:
        n_scap = -xprod(data[marker_names_scapula_tool['AI']] - data[marker_names_scapula_tool['TS']],
                        data[marker_names_scapula_tool['AA']] - data[marker_names_scapula_tool['TS']])
        lr = 'l'
    n_scap = n_scap/np.linalg.norm(n_scap)

    data[marker_names['ai_' + lr]] = data[marker_names_scapula_tool['AI']] - scap_dist / 1000 * n_scap
    data[marker_names['aa_' + lr]] = data[marker_names_scapula_tool['AA']] - scap_dist / 1000 * n_scap
    data[marker_names['ts_' + lr]] = data[marker_names_scapula_tool['TS']] - scap_dist / 1000 * n_scap

    del data[marker_names_scapula_tool['AI']]
    del data[marker_names_scapula_tool['AA']]
    del data[marker_names_scapula_tool['TS']]

    return data


def rigid_transform_3d(a, b):

    # This function finds the optimal Rigid/Euclidean transform in 3D space
    # It expects as input a 3xN matrix of 3D points. It returns R, t

    # find mean column wise
    centroid_a = np.mean(a, 0)
    centroid_b = np.mean(b, 0)

    num_cols = a.shape[0]

    # subtract mean
    am = np.transpose(a - np.tile(centroid_a, [num_cols, 1]))
    bm = np.transpose(b - np.tile(centroid_b, [num_cols, 1]))

    # calculate covariance matrix (is this the correct terminology?)
    h = np.matmul(am, np.transpose(bm))

    # find rotation
    [u, _, v] = np.linalg.svd(h)

    r = np.matmul(np.transpose(v), np.transpose(u))

    if np.linalg.det(r) < 0:
        v[2, :] = v[2, :] * (-1)
        r = np.matmul(np.transpose(v), np.transpose(u))

    t = np.matmul(-r, centroid_a) + centroid_b

    return [r, t]


def orthogonal_projection(p, n, q):
    return q - np.dot(q - p, n) * n


def func_sym_to_float(x, f, a_symbol, b_symbol):
    a_float, b_float = x
    return f.subs([(a_symbol, a_float), (b_symbol, b_float)])


def fit_bilinear_function(x, y, z):
    ad = osim.ArrayDouble(0, 3)

    v = np.array([np.ones(len(x)), x, y])
    coefficients = np.linalg.lstsq(v.T, z, rcond=None)

    ad.set(0, coefficients[0][2])
    ad.set(1, coefficients[0][1])
    ad.set(2, coefficients[0][0])
    return ad


def xprod(a, b):
    ax = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.matmul(ax, b)


def rot_vec_around_axis_by_angle(v, axis, angle):
    return v * sympy.cos(angle) + axis.cross(v) * sympy.sin(angle) + axis * axis.dot(v) * (1 - sympy.cos(angle))
