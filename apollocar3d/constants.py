import numpy as np

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
    "bottom_right_c_right_front_car_light", # 56
    "top_right_c_right_front_car_light",     # 57
    "top_right_c_front_lplate",             # 58
    "top_left_c_front_lplate",              # 59
    "bottom_right_c_front_lplate",           # 60
    "bottom_left_c_front_lplate",          # 61
    "top_left_c_rear_lplate",               # 62
    "top_right_c_rear_lplate",              # 63
    "bottom_right_c_rear_lplate",           # 64
    "bottom_left_c_rear_lplate",            # 65
    ]


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

HFLIP = {}
checklist = []
for ind in HFLIP_ids:
    HFLIP[CAR_KEYPOINTS[ind]] = CAR_KEYPOINTS[HFLIP_ids[ind]]
    HFLIP[CAR_KEYPOINTS[HFLIP_ids[ind]]] = CAR_KEYPOINTS[ind]
    checklist.append(ind)
    checklist.append(HFLIP_ids[ind])
assert sorted(checklist) == list(range(len(CAR_KEYPOINTS)))
assert len(HFLIP) == len(CAR_KEYPOINTS)

CAR_CATEGORIES = ['car']

SKELETON_LEFT = [
    [59, 61], [59,1], [61, 5], [0,1], [0,2], [2,3], [3,1], [3,4], [4,5], #front
    [5, 6], [6,7], [4, 7], [2,9], [9,8], [8,11], [7,10], [6, 10], [9,10], #side front part
    [11,12], [11,24], [9,12], [10,15], [12,15], [9, 13], [13,14], [14,12], [14,15], #side middle part
    [24,16], [12,16], [12, 17], [17, 18], [18, 16], [15, 19], [19, 20], [19,18], [20, 21], [16,21], # side back part
    [16, 22], [21, 28], [22, 23], [23,28], [22,25], [25,26], [23,26], [26,27], [25,62], [27,65], [62, 65], [28, 65]
    ]

SKELETON_RIGHT = [[HFLIP_ids[bone[0]], HFLIP_ids[bone[1]]] for bone in SKELETON_LEFT]

SKELETON_CONNECT = [
    [28, 29], [62,63], [65,64], [24,33], [46, 11], [48,9], [59, 58], [60, 61], [0, 57], [49, 8]
    ]

SKELETON_ALL = SKELETON_LEFT + SKELETON_RIGHT + SKELETON_CONNECT

CAR_SKELETON = [(bone[0] + 1, bone[1] + 1) for bone in SKELETON_ALL]  # COCO style skeletons label the first kp with 1

CAR_SIGMAS = [0.05] * len(CAR_KEYPOINTS)

split, error = divmod(len(CAR_KEYPOINTS), 4)
CAR_SCORE_WEIGHTS = [10.0] * split + [3.0] * split + [1.0] * split + [0.1] * split + [0.1] * error
assert len(CAR_SCORE_WEIGHTS) == len(CAR_KEYPOINTS)


#number plate offsets
p_x = 0.3
p_y_top = -0.2
p_y_bottom = -0.4

# z for front
front_z = -2.0
front_z_side = -1.8
front_z_corner = -1.7
front_z_wheel = -1.4
front_z_door = -1.0

# lights x offset
light_x_inside = 0.8
x_outside = 1.0

# y offsets
top_car = 0.5
bottom_line = -0.75
top_line = 0.1

# z for the back
back_z_wheel = 1.0
back_z = 1.5
back_z_side = 1.3

CAR_POSE_HALF = np.array([
    [-light_x_inside, 0.0, front_z],    # 0
    [-light_x_inside, -0.2, front_z],  # 1
    [-x_outside, 0.0, front_z_side],  # 2 
    [-x_outside, -0.2, front_z_side],  # 3
    [-x_outside, p_y_bottom, front_z_side],  # 4
    [-x_outside, p_y_bottom - 0.2, front_z_side],  # 5
    [-x_outside, bottom_line, front_z_corner],  # 6
    [-x_outside, bottom_line + 0.1, front_z_wheel,],  # 7
    [-x_outside + 0.1, top_car, front_z_door + 0.5,],  # 8
    [-x_outside, top_line, front_z_door,],  # 9
    [-x_outside, bottom_line,  front_z_door,],  # 10
    [-x_outside + 0.1,  top_car, 0.1,],  # 11
    [-x_outside, top_line, 0.05,],  # 12
    [-x_outside, 0.0, -0.1,],  # 13
    [-x_outside, 0.0, 0.0,],  # 14
    [-x_outside, bottom_line, 0.0,],  # 15
    [-x_outside, top_line, back_z_wheel,],  # 16
    [-x_outside, 0.0, back_z_wheel * 0.8,],  # 17
    [-x_outside, 0.0, back_z_wheel * 0.9,],  # 18
    [-x_outside, bottom_line, back_z_wheel * 0.6,],  # 19
    [-x_outside, bottom_line + 0.1, back_z_wheel,],  # 20
    [-x_outside, bottom_line, back_z_side - 0.2,],  # 21
    [-x_outside, 0.0, back_z_side,],  # 22
    [-x_outside, -0.2, back_z_side,],  # 23
    [-x_outside + 0.1,  top_car - 0.1, back_z_wheel,],  # 24
    [-light_x_inside, 0.0, back_z,],  # 25
    [-light_x_inside, -0.2, back_z,],  # 26
    [-light_x_inside + 0.1, -0.3, back_z,],  # 27
    [-x_outside + 0.1, bottom_line, back_z,]]  # 28
    + [[np.nan, np.nan, np.nan,]] * 30 + # will later be mirrored, see figure 3, ApolloCar 3D paper
    [[-p_x, p_y_top, front_z],]  # 59
    + [[np.nan, np.nan, np.nan,]] * 1 + # will later be mirrored, see figure 3, ApolloCar 3D paper
    [[-p_x, p_y_bottom, front_z],  # 61
    [-p_x, p_y_top, back_z]] +  # 62
    [[np.nan, np.nan, np.nan,]] * 2 + # will later be mirrored, see figure 3, ApolloCar 3D paper
    [[-p_x, p_y_bottom, back_z],]  # 65
)

CAR_POSE = CAR_POSE_HALF
for key in HFLIP_ids:
    CAR_POSE[HFLIP_ids[key],:] = CAR_POSE_HALF[key,:]
    CAR_POSE[HFLIP_ids[key],0] = -CAR_POSE_HALF[key,0]
CAR_POSE[:,2] = CAR_POSE[:,2]+2.0
assert not np.any(CAR_POSE==np.nan)


def draw_ann(ann, *, keypoint_painter, filename=None, margin=0.5, aspect=None, **kwargs):
    from openpifpaf import show  # pylint: disable=import-outside-toplevel

    bbox = ann.bbox()
    xlim = bbox[0] - margin, bbox[0] + bbox[2] + margin
    ylim = bbox[1] - margin, bbox[1] + bbox[3] + margin
    if aspect == 'equal':
        fig_w = 5.0
    else:
        fig_w = 5.0 / (ylim[1] - ylim[0]) * (xlim[1] - xlim[0])

    with show.canvas(filename, figsize=(fig_w, 5), nomargin=True, **kwargs) as ax:
        ax.set_axis_off()
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)

        if aspect is not None:
            ax.set_aspect(aspect)

        keypoint_painter.annotation(ax, ann)


def draw_skeletons(pose, prefix):
    from openpifpaf.annotation import Annotation  # pylint: disable=import-outside-toplevel
    from openpifpaf import show  # pylint: disable=import-outside-toplevel
    scale = np.sqrt(
        (np.max(pose[:, 0]) - np.min(pose[:, 0]))
        * (np.max(pose[:, 1]) - np.min(pose[:, 1]))
    )
    show.KeypointPainter.show_joint_scales = False
    keypoint_painter = show.KeypointPainter(line_width=3.0, )
    ann = Annotation(keypoints=CAR_KEYPOINTS, skeleton=CAR_SKELETON, score_weights=CAR_SCORE_WEIGHTS)
    pose[:,2]=1.0
    ann.set(pose, np.array(CAR_SIGMAS) * scale)
    draw_ann(ann, filename='./docs/'+prefix+'skeleton_car.png', keypoint_painter=keypoint_painter)
    

def print_associations():
    for j1, j2 in CAR_SKELETON:
        print(CAR_KEYPOINTS[j1 - 1], '-', CAR_KEYPOINTS[j2 - 1])

def rotate(pose, angle=45, axis=2):
    sin = np.sin(np.radians(angle))
    cos = np.cos(np.radians(angle))
    pose_copy = np.copy(pose)
    pose_copy[:, 2] = pose_copy[:, 2]-2 # COOS at human center
    if axis==0:
        rot_mat = np.array([[1, 0, 0],
                            [0, cos, -sin],
                            [0, sin, cos]])
    elif axis==1:
        rot_mat = np.array([[cos, 0, sin],
                            [0, 1, 0],
                            [-sin, 0, cos]])
    elif axis==2:
        rot_mat = np.array([[cos, -sin, 0],
                            [sin, cos, 0],
                            [0, 0, 1]])
    else:
        raise Exception("Axis must be 0,1 or 2 (corresponding to x,y,z).")
    rotated_pose = np.transpose(np.matmul(rot_mat, np.transpose(pose_copy)))
    rotated_pose[:,2] = rotated_pose[:,2]+7 #assure sufficient depth for plotting
    return rotated_pose

if __name__ == '__main__':
    print_associations()
    #Use this to create a gif
    for deg in range(360):
        rot = rotate(CAR_POSE, angle=deg, axis=1)
        top_view = rotate(rot, angle=20, axis=0)
        draw_skeletons(top_view, prefix="rotate_"+str(deg)+"deg_")
    draw_skeletons(CAR_POSE, prefix="Front_")