import json
import copy
import shutil

### List of all the annotation types that should be used ###
ann_types = ["keypoints", "foot_kpts", "face_kpts", "lefthand_kpts", "righthand_kpts"]

# =============================================================================
# orig_file = "../data-mscoco/annotations_wholebody/coco_wholebody_train_v1.0.json"
# new_file = "../data-mscoco/annotations/person_keypoints_train2017_wholebody_pifpaf_style.json"
# =============================================================================

orig_file = "../../data-mscoco/annotations_wholebody/coco_wholebody_val_v1.0.json"
new_file = "../../data-mscoco/annotations/person_keypoints_val2017_wholebody_pifpaf_style.json"

drop_attribute_list = ["face_box", "face_kpts", "face_valid", "foot_kpts", "foot_valid", 
                      "lefthand_box", "lefthand_kpts", "lefthand_valid", 
                      "num_keypoints", "righthand_box", "righthand_kpts", 
                      "righthand_valid", "segmentation"]

with open(orig_file, 'r') as f:
        orig_data = json.load(f)

new_data = copy.deepcopy(orig_data)
new_data["annotations"]=[]

discard_count=0        
for ann_dict in orig_data["annotations"]:
    if not(all(x==0 for x in ann_dict["keypoints"])): #If all zero, only bbox (and sometimes other kpts were annotated (e.g. face, hand))
        new_dict=copy.deepcopy(ann_dict)
        for entry in drop_attribute_list:
            new_dict.pop(entry)
        ann = []
        for key in ann_types:
            ann = ann + ann_dict[key]
        new_dict["keypoints"] = ann
        new_dict['num_keypoints'] = sum(x>0 for x in ann[2::3])
        new_data["annotations"].append(new_dict)
    else:
        discard_count+=1
        
with open(new_file, 'w') as f:
        json.dump(new_data, f)

print("\nCreated a new json file with "+ str(len(new_data["annotations"]))+ " poses annotated and discarded " +str(discard_count)+" annotations from the original file.")
