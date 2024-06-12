import xml.etree.ElementTree as ElementTree
from processing import score
from processing import scaling
import processing.coordinate_systems as cs
import os
import sys


########################################################################################################################
# READ INPUT FILE
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

# scale setup
folder_scale_setup = root.find('folder_scale_setup').text
filename_scale_setup = root.find('filename_scale_setup').text
path_scale_setup_in = os.path.join(folder_scale_setup, filename_scale_setup)

# scaling file
folder_trc = root.find('folder_trc').text
filename_scalingfile = root.find('filename_scalingfile').text
path_scalingfile = os.path.join(folder_trc, filename_scalingfile)
time_str = str.split(root.find('time_range').text, ' ')
time_range = [float(i) for i in time_str]

# calibration files used for ghr estimation
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

# output model
folder_model_output = root.find('folder_model_output').text
if not os.path.exists(folder_model_output):  # create path if it doesn't exist
    os.makedirs(folder_model_output)
filename_model_scaled = root.find('filename_model_scaled').text
path_model_scaled = os.path.join(folder_model_output, filename_model_scaled)

# marker names
marker_names = {'Sternum': root.find('marker_names').find('IJ').text,
                'C7': root.find('marker_names').find('C7').text,
                'PX': root.find('marker_names').find('PX').text,
                'T8': root.find('marker_names').find('T8').text,
                'ts_r': root.find('marker_names').find('TS_r').text,
                'ai_r': root.find('marker_names').find('AI_r').text,
                'aa_r': root.find('marker_names').find('AA_r').text,
                'r_AMC2': root.find('marker_names').find('Acromion_r').text,
                'r_elbow_lateral': root.find('marker_names').find('EL_r').text,
                'r_elbow_medial': root.find('marker_names').find('EM_r').text,
                'r_humerous': root.find('marker_names').find('upperarm_tracking_marker_r').text,
                'r_wrist_ulna': root.find('marker_names').find('wrist_ulna_r').text,
                'r_wrist_radius': root.find('marker_names').find('wrist_radius_r').text,
                'r_forearm': root.find('marker_names').find('forearm_tracking_marker_r').text,
                'ts_l': root.find('marker_names').find('TS_l').text,
                'ai_l': root.find('marker_names').find('AI_l').text,
                'aa_l': root.find('marker_names').find('AA_l').text,
                'l_AMC2': root.find('marker_names').find('Acromion_l').text,
                'l_elbow_lateral': root.find('marker_names').find('EL_l').text,
                'l_elbow_medial': root.find('marker_names').find('EM_l').text,
                'l_humerous': root.find('marker_names').find('upperarm_tracking_marker_l').text,
                'l_wrist_ulna': root.find('marker_names').find('wrist_ulna_l').text,
                'l_wrist_radius': root.find('marker_names').find('wrist_radius_l').text,
                'l_forearm': root.find('marker_names').find('forearm_tracking_marker_l').text
                }

move_scapula_markers_to_skin_text = root.find('move_scapula_markers_to_skin').text
if move_scapula_markers_to_skin_text == "True":
    move_scapula_markers_to_skin = True
    # distance from scapula locator to skin in mm
    scapula_distance = float(root.find('scapula_distance').text)
    marker_names_scapula_tool = {'AA': root.find('marker_names_scapula_tool').find('AA').text,
                                 'AI': root.find('marker_names_scapula_tool').find('AI').text,
                                 'TS': root.find('marker_names_scapula_tool').find('TS').text}
else:
    move_scapula_markers_to_skin = False
    scapula_distance = 0
    marker_names_scapula_tool = None

########################################################################################################################
# MODEL SCALING
########################################################################################################################

# right arm: use calibaration files to calibrate glenohumeral joint center computation
[g_humerus, _] = score.calibrate_score(folder_trc, filenames_calibration_right, 'right', marker_names,
                                       move_scapula_markers_to_skin, scapula_distance, marker_names_scapula_tool)
# right arm: pre-compute humerus scale factor using calibration files
humerus_scale_factor_r = score.compute_humerus_scale_factor(path_model_input, folder_trc, filenames_calibration_right,
                                                            'right', g_humerus, marker_names,
                                                            move_scapula_markers_to_skin, scapula_distance,
                                                            marker_names_scapula_tool)

# left arm: use calibaration files to calibrate glenohumeral joint center computation
[g_humerus, _] = score.calibrate_score(folder_trc, filenames_calibration_left, 'left', marker_names,
                                       move_scapula_markers_to_skin, scapula_distance, marker_names_scapula_tool)
# left arm: pre-compute humerus scale factor using calibration files
humerus_scale_factor_l = score.compute_humerus_scale_factor(path_model_input, folder_trc, filenames_calibration_left,
                                                            'left', g_humerus, marker_names,
                                                            move_scapula_markers_to_skin, scapula_distance,
                                                            marker_names_scapula_tool)

humerus_scale_factors = {'r': humerus_scale_factor_r, 'l': humerus_scale_factor_l}

# scale model
scaling.scale_model(path_scale_setup_in, path_model_input, path_scalingfile, path_model_scaled, humerus_scale_factors,
                    time_range)

# update ISB shoulder coordinate systems after scaling
if root.find('update_ISB_coordinate_systems').text:
    cs.update_isb_coords(path_model_scaled)

# correct the position of the tracking markers on the humerus and forearm
mot_scaling_ISB = path_model_scaled.replace('.osim', '_scale.mot')
scaling.correct_tracking_markers(path_model_scaled, path_scalingfile, mot_scaling_ISB, marker_names)
