import csv
import numpy as np
import opensim as osim
import sympy


def vec3_to_np(vec3):
    """
    transform an opensim vec3 to a numpy array

    Parameters:
        vec3: vector in opensim vec3 format
    Returns:
        numpy array
    """
    return np.array([vec3[0], vec3[1], vec3[2]])


def read_trc(filepath: str):
    """
    read the content of a trc file into a data dict

    Parameters:
        filepath: Full path to trc file that should be read
    Returns:
        data_dict: dictionary containing marker trajectories
    """
    with open(filepath, "r", encoding="utf8") as trc_file:
        # open file reader
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
    """
    create a dict that contains the average over the data array of every field of the input dictionary excluding the
    frame numbers and time. Nan values are ignored when averaging.

    Parameters:
        data_dict: input dictionary, contains a 3xn data array for each field
    Returns:
        data_average: dict that contains the average over the n data array entries for every field of input
    """
    fns = data_dict.keys()
    data_average = dict()
    for fn in fns:
        if 'Frame' not in fn and 'Time' not in fn:
            data_average[fn] = np.nanmean(data_dict[fn], 0)/1000
    return data_average


def move_scapula_markers_onto_skin(data, marker_names, left_right, scap_dist, marker_names_scapula_tool):
    """
    Method to move the markers on the scapula location tool onto the skin. Assumptions: the scapula location tool is an
    L-frame with markers on each ends and a marker on the corner. At each marker's location, a pin points
    perpendicularly downwards and is used for the placement of the scapula location tool on the palpated AA, AI, and TS
    positions. Using the length of the pins, this method performs an orthogonal projection of the measured marker
    positions onto the skin.

    Parameters:
        data: data dict containing one 3D location in m per marker (e.g., average over static trial)
        marker_names: names of the fields (=markersnames) in the data dict
        left_right: left or right shoulder
        scap_dist: lenghts of the pins = distance between marker on scapula location tool and skin in mm
        marker_names_scapula_tool: names of the markers on the tool in the recording. The markers will be renamed to
                                   the markernames given for the scapula markers in the marker_names input. Can NOT be
                                   the same name.
    Returns:
        data: same as input data dict, but with scapula markers moved onto skin.
    """

    # compute normal vector pointing away from the skin
    if 'right' in left_right:
        n_scap = xprod(data[marker_names_scapula_tool['AI']] - data[marker_names_scapula_tool['TS']],
                       data[marker_names_scapula_tool['AA']] - data[marker_names_scapula_tool['TS']])
        lr = 'r'
    else:
        n_scap = -xprod(data[marker_names_scapula_tool['AI']] - data[marker_names_scapula_tool['TS']],
                        data[marker_names_scapula_tool['AA']] - data[marker_names_scapula_tool['TS']])
        lr = 'l'
    n_scap = n_scap/np.linalg.norm(n_scap)

    # calculate new marker positions by moving the current location by scap_dist along the normal onto the skin.
    # divide scap dist (in mm) by 1000 because input marker locations are in m.
    data[marker_names['ai_' + lr]] = data[marker_names_scapula_tool['AI']] - scap_dist / 1000 * n_scap
    data[marker_names['aa_' + lr]] = data[marker_names_scapula_tool['AA']] - scap_dist / 1000 * n_scap
    data[marker_names['ts_' + lr]] = data[marker_names_scapula_tool['TS']] - scap_dist / 1000 * n_scap

    # remove old markers
    del data[marker_names_scapula_tool['AI']]
    del data[marker_names_scapula_tool['AA']]
    del data[marker_names_scapula_tool['TS']]

    return data


def rigid_transform_3d(a, b):
    """
    This function finds the optimal Rigid transform (rotation and translation) in 3D space that transforms a onto b.

    Parameters:
        a: 3xN matrix of 3D points
        b: 3xN matrix of 3D points
    Returns:
        r: rotation matrix
        t: translation vector
    """

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

    # correct direction if necessary
    if np.linalg.det(r) < 0:
        v[2, :] = v[2, :] * (-1)
        r = np.matmul(np.transpose(v), np.transpose(u))

    t = np.matmul(-r, centroid_a) + centroid_b

    return [r, t]


def orthogonal_projection(p, n, q):
    """
    Project point q orthogonally onto the line that is defined by point p and normal n

    Parameters:
        p: 3D point on line
        n: 3D normal vector of line
        q: point to orthogonally project onto line
    Returns:
        projected point
    """
    return q - np.dot(q - p, n) * n


def func_sym_to_float(x, f, a_symbol, b_symbol):
    """
    Replace the symbolic variables by float variables to enable the use of the function for optimization

    Parameters:
        x: float variable vector with intial guess for a and b
        f: function defined with symbolic variables a_symbol, b_symbol
        a_symbol: symbolic variable a to replace
        b_symbol: symbolic variable b to replace
    Returns:
        function in which symbolic variables have been replaced with float variables
    """
    a_float, b_float = x
    return f.subs([(a_symbol, a_float), (b_symbol, b_float)])


def fit_bilinear_function(x, y, z):
    """
    fit a bilinear plane to the given points via a linear least squares fit. x and y are the independent dimensions,
    z is the dependent dimension

    Parameters:
        x: x coordinates of points (here: shoulder elevation plane angle)
        y: y coordinates of points (here: shoulder elevation angle)
        z: z coordinates of points (here: dependent shoulder coordinate, e.g., scapula upward rotation)
    Returns:
        OpenSim double array containing the slopes and z-zero-intercept of the plane
    """
    ad = osim.ArrayDouble(0, 3)

    # create array with ones (slopes) and x-y-coordinates
    v = np.array([np.ones(len(x)), x, y])
    # find plane that best fits the xyz positions.
    # Return order intercept, slope in x direction (shoulder elevation plane), slope in y direction (shoulder elevation)
    coefficients = np.linalg.lstsq(v.T, z, rcond=None)

    # store in order that used in CoordinateCouplerConstraints:
    # slope shoulder elevation, slope shoulder elevation plane, intercept
    ad.set(0, coefficients[0][2])
    ad.set(1, coefficients[0][1])
    ad.set(2, coefficients[0][0])
    return ad


def xprod(a, b):
    """
    compute the cross product of vectors a and b (NOT normalized to unit length).

    Parameters:
        a: 3D vector
        b: 3D vector
    Returns:
        cross product of the vectors
    """
    ax = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.matmul(ax, b)


def rot_vec_around_axis_by_angle(v, axis, angle):
    """
    Symbolic function. Rotate the vector v by angle around axis using Rodrigues' rotation formula

    Parameters:
        v: 3x1 sympy symbolic matrix
        axis: 3x1 sympy symbolic matrix
        angle: 1D sympy symbol
    Returns:
        rotated symbolic vector
    """
    return v * sympy.cos(angle) + axis.cross(v) * sympy.sin(angle) + axis * axis.dot(v) * (1 - sympy.cos(angle))
