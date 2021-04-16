import numpy as np
import json
import time
from constants import WHOLEBODY_SKELETON as skel

### calculate average dist ###
# script takes about 5min on a desktop computer
data_path = "./person_keypoints_train2017_wholebody_pifpaf_style.json"
with open(data_path, 'r') as f:
        data = json.load(f)
skel = [(bone[0]-1, bone[1]-1) for bone in skel]
distance = np.zeros((len(skel)), dtype=np.float64)
no_instance = np.zeros((len(skel)), dtype=np.int)
start = time.time()
for ann in data["annotations"]:
    kp = np.array(ann["keypoints"])
    visible_kps = np.where(kp[2::3]>0)[0]
    kp = kp.reshape((133,3))
    for bone_id, bone in enumerate(skel):
        if bone[0] in visible_kps and bone[1] in visible_kps:
            no_instance[bone_id] += 1
            distance[bone_id] += np.linalg.norm(kp[bone[0],:2]-kp[bone[1],:2])
normalized_dist = distance / no_instance
with open("Edge_weights_train_wb.json", 'w') as f:
            json.dump(normalized_dist.tolist(), f)
#print(normalized_dist)
print("Processing took {:.2f} seconds".format(time.time()-start))
