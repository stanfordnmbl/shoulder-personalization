import os
import xml.etree.ElementTree as ElementTree
import opensim as osim
from processing import tools
from processing import score
from processing import personalization
import sys

########################################################################################################################
# INPUT
########################################################################################################################

# open provided file
settings_file = sys.argv[1]
if not os.path.exists(settings_file):
    settings_file = str(input('Please enter path to settings file'))
tree = ElementTree.parse(settings_file)
root = tree.getroot()

# input model
folder_model_input = root.find('folder_model_input').text
filename_model_input = root.find('filename_model_input').text
path_model_input = os.path.join(folder_model_input, filename_model_input)

# output model
folder_model_output = root.find('folder_model_output').text
filename_model_output = root.find('filename_model_output').text
path_model_output = os.path.join(folder_model_output, filename_model_output)
model_output_name = root.find('model_output_name').text

# files
folder_trc = root.find('folder_trc').text
filenames_calibration_right = [root.find('filenames_calibration_right').find('abd0').text,
                               root.find('filenames_calibration_right').find('abd180').text,
                               root.find('filenames_calibration_right').find('flex180').text,
                               root.find('filenames_calibration_right').find('horiz_abd').text,
                               root.find('filenames_calibration_right').find('horiz_add').text]
filenames_calibration_left = [root.find('filenames_calibration_left').find('abd0').text,
                              root.find('filenames_calibration_left').find('abd180').text,
                              root.find('filenames_calibration_left').find('flex180').text,
                              root.find('filenames_calibration_left').find('horiz_abd').text,
                              root.find('filenames_calibration_left').find('horiz_add').text]

# constraint names
constraints = ['sternoclavicular_r2_con_',
               'sternoclavicular_r3_con_',
               'acromioclavicular_r1_con_',
               'acromioclavicular_r2_con_',
               'acromioclavicular_r3_con_']

# constraint names
marker_names = {'Sternum': root.find('marker_names').find('IJ').text,
                'C7': root.find('marker_names').find('C7').text,
                'PX': root.find('marker_names').find('PX').text,
                'T8': root.find('marker_names').find('T8').text,
                'ts_r': root.find('marker_names').find('TS_r').text,
                'ai_r': root.find('marker_names').find('AI_r').text,
                'aa_r': root.find('marker_names').find('AA_r').text,
                'r_elbow_lateral': root.find('marker_names').find('EL_r').text,
                'r_elbow_medial': root.find('marker_names').find('EM_r').text,
                'r_humerous': root.find('marker_names').find('upperarm_tracking_marker_r').text,
                'ts_l': root.find('marker_names').find('TS_l').text,
                'ai_l': root.find('marker_names').find('AI_l').text,
                'aa_l': root.find('marker_names').find('AA_l').text,
                'l_elbow_lateral': root.find('marker_names').find('EL_l').text,
                'l_elbow_medial': root.find('marker_names').find('EM_l').text,
                'l_humerous': root.find('marker_names').find('upperarm_tracking_marker_l').text
                }

move_scapula_markers_to_skin = root.find('move_scapula_markers_to_skin').text
if move_scapula_markers_to_skin:
    # distance from scapula locator to skin in mm
    scapula_distance = float(root.find('scapula_distance').text)
    marker_names_scapula_tool = {'AA': root.find('marker_names_scapula_tool').find('AA').text,
                                 'AI': root.find('marker_names_scapula_tool').find('AI').text,
                                 'TS': root.find('marker_names_scapula_tool').find('TS').text}
else:
    scapula_distance = 0
    marker_names_scapula_tool = None

extend_coordinate_ranges = root.find('extend_coordinate_ranges').text

################################################################################################################
# Personalization: use calibration poses left/right to adapt constraint slopes
################################################################################################################

# Load OpenSim scaled model
model = osim.Model(path_model_input)
model.initSystem()

# name of model = mode for fitting
model.setName(model_output_name)

for left_right in ['right', 'left']:
    # left / right
    if 'right' in left_right:
        lr = 'r'
        filenames_calibration = filenames_calibration_right
    else:
        lr = 'l'
        filenames_calibration = filenames_calibration_left

    # calibrate GH joint center computation
    [g_humerus, g_scapula] = score.calibrate_score(folder_trc, filenames_calibration, left_right, marker_names,
                                                   move_scapula_markers_to_skin, scapula_distance,
                                                   marker_names_scapula_tool)

    # compute coordinate angles for all trials
    coordinates = personalization.compute_constraint_angles(path_model_input, folder_trc, filenames_calibration,
                                                            left_right, g_humerus, g_scapula, marker_names,
                                                            move_scapula_markers_to_skin, scapula_distance,
                                                            marker_names_scapula_tool)

    # fit bilinear surface to all data points per constraint and update constraint slopes
    for i, constraint in enumerate(constraints):
        # get constraint and pointer to linear function
        con = osim.CoordinateCouplerConstraint.safeDownCast(model.getConstraintSet().get(constraint + lr))
        lf = osim.LinearFunction.safeDownCast(con.getFunction())

        # fit bilinear plane
        ad = tools.fit_bilinear_function(coordinates[:, 5], coordinates[:, 6], coordinates[:, i])

        # update constraint coefficients
        lf.setCoefficients(ad)

if extend_coordinate_ranges:
    personalization.extend_coordinate_ranges(model)

# save adapted model
model.printToXML(os.path.join(folder_model_output, filename_model_output))

################################################################################################################
# create models with clavicle elevation constraints turned off and ac marker added
################################################################################################################
if root.find('create_clavicle_elevation_model').text:
    filename_model_clav_output = root.find('filename_model_clav_output').text
    path_model_clav_output = os.path.join(folder_model_output, filename_model_clav_output)
    personalization.unconstrain_clav_elev_add_ac_marker(path_model_output, path_model_clav_output)
