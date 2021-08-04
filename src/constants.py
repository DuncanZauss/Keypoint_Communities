import numpy as np
import copy

CAR_KEYPOINTS = [
    "top_left_c_left_front_car_light",      # 0
    "bottom_left_c_left_front_car_light",   # 1
    "top_right_c_left_front_car_light",     # 2
    "bottom_right_c_left_front_car_light",  # 3
    "top_right_c_left_front_fog_light",     # 4
    "bottom_right_c_left_front_fog_light",  # 5
    "front_section_left_front_wheel",       # 6
    "center_left_front_wheel",              # 7
    "top_right_c_front_glass",              # 8
    "top_left_c_left_front_door",           # 9
    "bottom_left_c_left_front_door",        # 10
    "top_right_c_left_front_door",          # 11
    "middle_c_left_front_door",             # 12
    "front_c_car_handle_left_front_door",   # 13
    "rear_c_car_handle_left_front_door",    # 14
    "bottom_right_c_left_front_door",       # 15
    "top_right_c_left_rear_door",           # 16
    "front_c_car_handle_left_rear_door",    # 17
    "rear_c_car_handle_left_rear_door",     # 18
    "bottom_right_c_left_rear_door",        # 19
    "center_left_rear_wheel",               # 20
    "rear_section_left_rear_wheel",         # 21
    "top_left_c_left_rear_car_light",       # 22
    "bottom_left_c_left_rear_car_light",    # 23
    "top_left_c_rear_glass",                # 24
    "top_right_c_left_rear_car_light",      # 25
    "bottom_right_c_left_rear_car_light",   # 26
    "bottom_left_c_trunk",                  # 27
    "Left_c_rear_bumper",                   # 28
    "Right_c_rear_bumper",                  # 29
    "bottom_right_c_trunk",                 # 30
    "bottom_left_c_right_rear_car_light",   # 31
    "top_left_c_right_rear_car_light",      # 32
    "top_right_c_rear_glass",               # 33
    "bottom_right_c_right_rear_car_light",  # 34
    "top_right_c_right_rear_car_light",     # 35
    "rear_section_right_rear_wheel",        # 36
    "center_right_rear_wheel",              # 37
    "bottom_left_c_right_rear_car_door",    # 38
    "rear_c_car_handle_right_rear_car_door",    # 39
    "front_c_car_handle_right_rear_car_door",   # 40
    "top_left_c_right_rear_car_door",       # 41
    "bottom_left_c_right_front_car_door",   # 42
    "rear_c_car_handle_right_front_car_door",   # 43
    "front_c_car_handle_right_front_car_door",  # 44
    "middle_c_right_front_car_door",        # 45
    "top_left_c_right_front_car_door",      # 46
    "bottom_right_c_right_front_car_door",  # 47
    "top_right_c_right_front_car_door",     # 48
    "top_left_c_front_glass",               # 49
    "center_right_front_wheel",             # 50
    "front_section_right_front_wheel",      # 51
    "bottom_left_c_right_fog_light",        # 52
    "top_left_c_right_fog_light",           # 53
    "bottom_left_c_right_front_car_light",  # 54
    "top_left_c_right_front_car_light",     # 55
    "bottom_right_c_right_front_car_light",  # 56
    "top_right_c_right_front_car_light",     # 57
    "top_right_c_front_lplate",             # 58
    "top_left_c_front_lplate",              # 59
    "bottom_right_c_front_lplate",           # 60
    "bottom_left_c_front_lplate",          # 61
    "top_left_c_rear_lplate",               # 62
    "top_right_c_rear_lplate",              # 63
    "bottom_right_c_rear_lplate",           # 64
    "bottom_left_c_rear_lplate", ]            # 65


HFLIP_ids = {
    0: 57,
    1: 56,
    2: 55,
    3: 54,
    4: 53,
    5: 52,
    6: 51,
    7: 50,
    8: 49,
    9: 48,
    10: 47,
    11: 46,
    12: 45,
    13: 44,
    14: 43,
    15: 42,
    16: 41,
    17: 40,
    18: 39,
    19: 38,
    20: 37,
    21: 36,
    22: 35,
    23: 34,
    24: 33,
    25: 32,
    26: 31,
    27: 30,
    28: 29,
    59: 58,
    61: 60,
    62: 63,
    65: 64
}

HFLIP_66 = {}
checklist = []
for ind in HFLIP_ids:
    HFLIP_66[CAR_KEYPOINTS[ind]] = CAR_KEYPOINTS[HFLIP_ids[ind]]
    HFLIP_66[CAR_KEYPOINTS[HFLIP_ids[ind]]] = CAR_KEYPOINTS[ind]
    checklist.append(ind)
    checklist.append(HFLIP_ids[ind])
assert sorted(checklist) == list(range(len(CAR_KEYPOINTS)))
assert len(HFLIP_66) == len(CAR_KEYPOINTS)

split, error = divmod(len(CAR_KEYPOINTS), 4)
CAR_SCORE_WEIGHTS = [10.0] * split + [3.0] * split + \
    [1.0] * split + [0.1] * split + [0.1] * error

# number plate offsets
P_X = 0.3
P_Y_TOP = -0.2
P_Y_BOTTOM = -0.4

# z for front
FRONT_Z = -2.0
FRONT_Z_SIDE = -1.8
FRONT_Z_CORNER = -1.7
FRONT_Z_WHEEL = -1.4
FRONT_Z_DOOR = -1.0

# lights x offset
LIGHT_X_INSIDE = 0.8
X_OUTSIDE = 1.0

# y offsets
TOP_CAR = 0.5
BOTTOM_LINE = -0.75
TOP_LINE = 0.1

# z for the back
BACK_Z_WHEEL = 1.0
BACK_Z = 1.5
BACK_Z_SIDE = 1.3

CAR_POSE_HALF = np.array([
    [-LIGHT_X_INSIDE, 0.0, FRONT_Z],    # 0
    [-LIGHT_X_INSIDE, -0.2, FRONT_Z],  # 1
    [-X_OUTSIDE, 0.0, FRONT_Z_SIDE],  # 2
    [-X_OUTSIDE, -0.2, FRONT_Z_SIDE],  # 3
    [-X_OUTSIDE, P_Y_BOTTOM, FRONT_Z_SIDE],  # 4
    [-X_OUTSIDE, P_Y_BOTTOM - 0.2, FRONT_Z_SIDE],  # 5
    [-X_OUTSIDE, BOTTOM_LINE, FRONT_Z_CORNER],  # 6
    [-X_OUTSIDE, BOTTOM_LINE + 0.1, FRONT_Z_WHEEL],  # 7
    [-X_OUTSIDE + 0.1, TOP_CAR, FRONT_Z_DOOR + 0.5],  # 8
    [-X_OUTSIDE, TOP_LINE, FRONT_Z_DOOR],  # 9
    [-X_OUTSIDE, BOTTOM_LINE, FRONT_Z_DOOR],  # 10
    [-X_OUTSIDE + 0.1, TOP_CAR, 0.1],  # 11
    [-X_OUTSIDE, TOP_LINE, 0.05],  # 12
    [-X_OUTSIDE, 0.0, -0.1],  # 13
    [-X_OUTSIDE, 0.0, 0.0],  # 14
    [-X_OUTSIDE, BOTTOM_LINE, 0.0],  # 15
    [-X_OUTSIDE, TOP_LINE, BACK_Z_WHEEL],  # 16
    [-X_OUTSIDE, 0.0, BACK_Z_WHEEL * 0.8],  # 17
    [-X_OUTSIDE, 0.0, BACK_Z_WHEEL * 0.9],  # 18
    [-X_OUTSIDE, BOTTOM_LINE, BACK_Z_WHEEL * 0.6],  # 19
    [-X_OUTSIDE, BOTTOM_LINE + 0.1, BACK_Z_WHEEL],  # 20
    [-X_OUTSIDE, BOTTOM_LINE, BACK_Z_SIDE - 0.2],  # 21
    [-X_OUTSIDE, 0.0, BACK_Z_SIDE],  # 22
    [-X_OUTSIDE, -0.2, BACK_Z_SIDE],  # 23
    [-X_OUTSIDE + 0.1, TOP_CAR - 0.1, BACK_Z_WHEEL],  # 24
    [-LIGHT_X_INSIDE, 0.0, BACK_Z],  # 25
    [-LIGHT_X_INSIDE, -0.2, BACK_Z],  # 26
    [-LIGHT_X_INSIDE + 0.1, -0.3, BACK_Z],  # 27
    [-X_OUTSIDE + 0.1, BOTTOM_LINE, BACK_Z]] + \
    [[np.nan, np.nan, np.nan]] * 30 + \
    [[-P_X, P_Y_TOP, FRONT_Z]] + \
    [[np.nan, np.nan, np.nan]] + \
    [[-P_X, P_Y_BOTTOM, FRONT_Z],  # 61
     [-P_X, P_Y_TOP, BACK_Z]] + \
    [[np.nan, np.nan, np.nan]] * 2 + \
    [[-P_X, P_Y_BOTTOM, BACK_Z]])  # 65

CAR_POSE = CAR_POSE_HALF
for key in HFLIP_ids:
    CAR_POSE[HFLIP_ids[key], :] = CAR_POSE_HALF[key, :]
    CAR_POSE[HFLIP_ids[key], 0] = -CAR_POSE_HALF[key, 0]
assert not np.any(CAR_POSE == np.nan)


SKELETON_LEFT = [
    [59, 61], [59, 1], [61, 5], [0, 1], [0, 2], [2, 3], [3, 1], [3, 4], [4, 5],  # front
    [5, 6], [6, 7], [4, 7], [2, 9], [9, 8], [8, 11], [7, 10], [6, 10], [9, 10],  # side front part
    [11, 12], [11, 24], [9, 12], [10, 15], [12, 15],
    [9, 13], [13, 14], [14, 12], [14, 15],  # side middle part
    [24, 16], [12, 16], [12, 17], [17, 18], [18, 16],
    [15, 19], [19, 20], [19, 18], [20, 21], [16, 21],  # side back part
    [16, 22], [21, 28], [22, 23], [23, 28], [22, 25], [25, 26],
    [23, 26], [26, 27], [25, 62], [27, 65], [62, 65], [28, 65]]

SKELETON_RIGHT = [[HFLIP_ids[bone[0]], HFLIP_ids[bone[1]]] for bone in SKELETON_LEFT]

SKELETON_CONNECT = [
    [28, 29], [62, 63], [65, 64], [24, 33], [46, 11],
    [48, 9], [59, 58], [60, 61], [0, 57], [49, 8]]

SKELETON_ALL = SKELETON_LEFT + SKELETON_RIGHT + SKELETON_CONNECT

CAR_SKELETON = [(bone[0] + 1, bone[1] + 1) for bone in SKELETON_ALL]  # COCO style skeleton

body_foot_skeleton = [
    (16, 14), (14, 12), (17, 15), (15, 13), (12, 13), (6, 12), (7, 13),
    (6, 7), (6, 8), (7, 9), (8, 10), (9, 11), (2, 3), (1, 2), (1, 3),
    (2, 4), (3, 5), (4, 6), (5, 7),
    (16, 20), (16, 19), (16, 18),    # left foot
    (17, 23), (17, 21), (17, 22)     # right foot
]

face_skeleton = [
    (25, 5), (39, 4),  # ear to ear body
    (54, 1),  # nose to nose body
    (60, 3), (3, 63), (66, 2), (2, 69),  # eyes to eyes body
    ] + [(x, x+1) for x in range(24, 40)] + [   # face outline
    (24, 41), (41, 42), (42, 43), (43, 44), (44, 45), (45, 51),  # right eyebrow
    (40, 50), (50, 49), (49, 48), (48, 47), (47, 46), (46, 51),  # left eyebrow
    (24, 60), (60, 61), (61, 62), (62, 63), (63, 51), (63, 64), (64, 65), (65, 60),  # right eye
    (40, 69), (69, 68), (68, 67), (67, 66), (66, 51), (66, 71), (71, 70), (70, 69),  # left eye
    ] + [(x, x+1) for x in range(51, 59)] + [  # nose
    (59, 54), (57, 75), (78, 36), (72, 28), (72, 83)] + [(x, x+1) for x in range(72, 83)] + [  # m.
    (72, 84), (84, 85), (85, 86), (86, 87), (87, 88), (88, 78),  # upper lip
    (72, 91), (91, 90), (90, 89), (89, 78)  # lower lip
    ]

lefthand_skeleton = [
    (92, 10),  # connect to wrist
    (92, 93), (92, 97), (92, 101), (92, 105), (92, 109)  # connect to finger starts
    ] + [(x, x+1) for s in [93, 97, 101, 105, 109] for x in range(s, s+3)]  # four finger

righthand_skeleton = [
    (113, 11),  # connect to wrist
    (113, 114), (113, 118), (113, 122), (113, 126), (113, 130)  # connect to finger starts
    ] + [(x, x+1) for s in [114, 118, 122, 126, 130] for x in range(s, s+3)]  # four finger

WHOLEBODY_SKELETON = body_foot_skeleton + face_skeleton + lefthand_skeleton + righthand_skeleton

body_kps = [
    'nose',            # 1
    'left_eye',        # 2
    'right_eye',       # 3
    'left_ear',        # 4
    'right_ear',       # 5
    'left_shoulder',   # 6
    'right_shoulder',  # 7
    'left_elbow',      # 8
    'right_elbow',     # 9
    'left_wrist',      # 10
    'right_wrist',     # 11
    'left_hip',        # 12
    'right_hip',       # 13
    'left_knee',       # 14
    'right_knee',      # 15
    'left_ankle',      # 16
    'right_ankle',     # 17
    ]

foot_kps = [
    'left_big_toe',    # 18
    'left_small_toe',  # 19
    'left_heel',       # 20
    'right_big_toe',   # 21
    'right_small_toe',  # 22
    'right_heel',      # 23
]

face_kps = ['f_' + str(x) for x in range(24, 92)]
lefth_kps = ['lh_' + str(x) for x in range(92, 113)]
righth_kps = ['rh_' + str(x) for x in range(113, 134)]

WHOLEBODY_KEYPOINTS = body_kps + foot_kps + face_kps + lefth_kps + righth_kps

SCALE_FACE = 1.05

body_pose = np.array([
    [0.0, 9.3, 2.0],    # 'nose',            # 1
    [-0.35 * SCALE_FACE, 9.7, 2.0],  # 'left_eye',        # 2
    [0.35 * SCALE_FACE, 9.7, 2.0],   # 'right_eye',       # 3
    [-0.7 * SCALE_FACE, 9.5, 2.0],   # 'left_ear',        # 4
    [0.7 * SCALE_FACE, 9.5, 2.0],    # 'right_ear',       # 5
    [-1.4, 8.0, 2.0],   # 'left_shoulder',   # 6
    [1.4, 8.0, 2.0],    # 'right_shoulder',  # 7
    [-1.75 - 0.4, 6.2 + 0.2, 2.0],  # 'left_elbow',      # 8
    [1.75 + 0.4, 6.2 + 0.2, 2.0],   # 'right_elbow',     # 9
    [-1.75 - 0.5, 4.2 + 0.5, 2.0],  # 'left_wrist',      # 10
    [1.75 + 0.5, 4.2 + 0.5, 2.0],   # 'right_wrist',     # 11
    [-1.26, 4.0, 2.0],  # 'left_hip',        # 12
    [1.26, 4.0, 2.0],   # 'right_hip',       # 13
    [-1.4, 2.0, 2.0],   # 'left_knee',       # 14
    [1.4, 2.0, 2.0],    # 'right_knee',      # 15
    [-1.4, 0.0, 2.0],   # 'left_ankle',      # 16
    [1.4, 0.0, 2.0], ])    # 'right_ankle',     # 17

foot_pose = np.array([
    [-1.2, -0.45, 3.0],   # 'left_big_toe',    # 18
    [-1.65, -0.45, 2.9],   # 'left_small_toe',  # 19
    [-1.4, -0.25, 1.9],   # 'left_heel',       # 20
    [1.2, -0.45, 3.0],    # 'right_big_toe',   # 21
    [1.65, -0.45, 2.9],    # 'right_small_toe', # 22
    [1.4, -0.25, 1.9], ])    # 'right_heel',      # 23

face_pose = np.array([
    # face outline
    [0.7, 9.7, 2.0],    # 24
    [0.7, 9.55, 2.0],   # 25
    [0.65, 9.3, 2.0],   # 26
    [0.59, 9.05, 2.0],  # 27
    [0.53, 8.8, 2.0],   # 28
    [0.45, 8.65, 2.0],  # 29
    [0.3, 8.55, 2.0],   # 30
    [0.15, 8.45, 2.0],  # 31
    [0.0, 8.4, 2.0],    # 32
    [-0.15, 8.45, 2.0],  # 33
    [-0.3, 8.55, 2.0],  # 34
    [-0.45, 8.65, 2.0],  # 35
    [-0.53, 8.8, 2.0],  # 36
    [-0.59, 9.05, 2.0],  # 37
    [-0.65, 9.3, 2.0],  # 38
    [-0.7, 9.55, 2.0],  # 39
    [-0.7, 9.7, 2.0],   # 40
    # eyebrows
    [0.6, 9.8, 2.0],    # 41
    [0.5, 9.9, 2.0],    # 42
    [0.4, 9.95, 2.0],   # 43
    [0.3, 9.9, 2.0],    # 44
    [0.2, 9.85, 2.0],   # 45
    [-0.2, 9.85, 2.0],  # 46
    [-0.3, 9.9, 2.0],   # 47
    [-0.4, 9.95, 2.0],  # 48
    [-0.5, 9.9, 2.0],   # 49
    [-0.6, 9.8, 2.0],   # 50
    # nose
    [0.0, 9.7, 2.0],    # 51
    [0.0, 9.566, 2.0],  # 52
    [0.0, 9.433, 2.0],  # 53
    [0.0, 9.3, 2.0],    # 54 equivalent to nose kp from body
    [0.2, 9.2, 2.0],    # 55
    [0.1, 9.15, 2.0],   # 56
    [0.0, 9.1, 2.0],    # 57
    [-0.1, 9.15, 2.0],  # 58
    [-0.2, 9.2, 2.0],   # 59
    # eyes
    [0.45, 9.7, 2.0],   # 60
    [0.4, 9.75, 2.0],   # 61
    [0.3, 9.75, 2.0],   # 62
    [0.2, 9.7, 2.0],    # 63
    [0.3, 9.65, 2.0],   # 64
    [0.4, 9.65, 2.0],   # 65
    [-0.2, 9.7, 2.0],   # 66
    [-0.3, 9.75, 2.0],  # 67
    [-0.4, 9.75, 2.0],  # 68
    [-0.45, 9.7, 2.0],  # 69
    [-0.4, 9.65, 2.0],  # 70
    [-0.3, 9.65, 2.0],  # 71
    # mouth
    [0.3, 8.8, 2.0],    # 72
    [0.2, 8.85, 2.0],   # 73
    [0.1, 8.9, 2.0],    # 74
    [0.0, 8.85, 2.0],   # 75
    [-0.1, 8.9, 2.0],   # 76
    [-0.2, 8.85, 2.0],  # 77
    [-0.3, 8.8, 2.0],   # 78
    [-0.2, 8.75, 2.0],  # 79
    [-0.1, 8.7, 2.0],   # 80
    [0.0, 8.65, 2.0],   # 81
    [0.1, 8.7, 2.0],    # 82
    [0.2, 8.75, 2.0],   # 83
    [0.2, 8.82, 2.0],   # 84
    [0.1, 8.82, 2.0],   # 85
    [0.0, 8.82, 2.0],   # 86
    [-0.1, 8.82, 2.0],  # 87
    [-0.2, 8.82, 2.0],  # 88
    [-0.1, 8.79, 2.0],  # 89
    [0.0, 8.79, 2.0],   # 90
    [0.1, 8.79, 2.0]])    # 91

face_pose[:, 0] = face_pose[:, 0] * SCALE_FACE


lefthand_pose = np.array([
    [-1.75, 3.9, 2.0],  # 92
    [-1.65, 3.8, 2.0],  # 93
    [-1.55, 3.7, 2.0],  # 94
    [-1.45, 3.6, 2.0],  # 95
    [-1.35, 3.5, 2.0],  # 96
    [-1.6, 3.5, 2.0],   # 97
    [-1.566, 3.4, 2.0],  # 98
    [-1.533, 3.3, 2.0],  # 99
    [-1.5, 3.2, 2.0],   # 100
    [-1.75, 3.5, 2.0],  # 101
    [-1.75, 3.4, 2.0],  # 102
    [-1.75, 3.3, 2.0],  # 103
    [-1.75, 3.2, 2.0],  # 104
    [-1.9, 3.5, 2.0],   # 105
    [-1.933, 3.4, 2.0],  # 106
    [-1.966, 3.3, 2.0],  # 107
    [-2.0, 3.2, 2.0],   # 108
    [-2.1, 3.5, 2.0],   # 109
    [-2.133, 3.433, 2.0],   # 110
    [-2.166, 3.366, 2.0],   # 111
    [-2.2, 3.3, 2.0], ])      # 112

lefthand_pose[:, 0] = (lefthand_pose[:, 0] + 1.75) * 1.0 - 2.25
lefthand_pose[:, 1] = (lefthand_pose[:, 1] - 3.9) * 1.5 + 4.4

righthand_pose = copy.deepcopy(lefthand_pose)
righthand_pose[:, 0] = -lefthand_pose[:, 0]

# [width, height, depth]
WHOLEBODY_STANDING_POSE = np.vstack((body_pose, foot_pose, face_pose,
                                     lefthand_pose, righthand_pose))

WHOLEBODY_SCORE_WEIGHTS = [100.0] * 3 + [1.0] * (len(WHOLEBODY_KEYPOINTS) - 3)
