from constants import WHOLEBODY_SKELETON, WHOLEBODY_KEYPOINTS, CAR_SKELETON, CAR_KEYPOINTS
import numpy as np
import json
import time


def main(data):
    for dataset in data:
        d_name = dataset["name"]
        kps = dataset["keypoints"]
        skel = dataset["skeleton"]
        data_path = dataset["path"]
        num = len(kps)
        with open(data_path, 'r') as f:
            data = json.load(f)
        skel = [(bone[0]-1, bone[1]-1) for bone in skel]
        distance = np.zeros((len(skel)), dtype=np.float64)
        no_instance = np.zeros((len(skel)), dtype=int)
        start = time.time()
        for ann in data["annotations"]:
            kp = np.array(ann["keypoints"])
            if len(kp.flatten()) == num*3:
                visible_kps = np.where(kp[2::3] > 0)[0]
                kp = kp.reshape((num, 3))
                for bone_id, bone in enumerate(skel):
                    if bone[0] in visible_kps and bone[1] in visible_kps:
                        no_instance[bone_id] += 1
                        distance[bone_id] += np.linalg.norm(kp[bone[0], :2]-kp[bone[1], :2])
        normalized_dist = distance / no_instance
        with open("Edge_weights_train_" + d_name + ".json", 'w') as f:
            json.dump(normalized_dist.tolist(), f)
        print("\nEdge weights for " + d_name)
        print("Edge \t\t Average distance in pixels")
        for i in range(len(normalized_dist)):
            print(str(skel[i]).ljust(16), normalized_dist[i])
        print("Processing took {:.2f} seconds for the {} dataset.".format(time.time()-start,
              d_name))


if __name__ == "__main__":
    data = [{"name": "wb",
             "path": "./person_keypoints_train2017_wholebody_pifpaf_style.json",
             "skeleton": WHOLEBODY_SKELETON,
             "keypoints": WHOLEBODY_KEYPOINTS},
            {"name": "apollocar",
             "path": "./apollo_keypoints_66_train.json",
             "skeleton": CAR_SKELETON,
             "keypoints": CAR_KEYPOINTS}, ]
    main(data)
