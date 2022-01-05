import os
import json
from collections import Counter

def _ori_dict_to_vec(ori_dict):
    vector = [0, 0, 0, 0, 0, 0]
    keys = ori_dict.keys()
    if "n/a" in keys or len(keys) == 0:
        return tuple(vector)
    elif "+x" in keys:
        vector[0] = 1
    elif "-x" in keys:
        vector[1] = 1
    elif "+y" in keys:
        vector[2] = 1
    elif "-y" in keys:
        vector[3] = 1
    elif "+z" in keys:
        vector[4] = 1
    elif "-z" in keys:
        vector[5] = 1
    else:
        print(ori_dict)
        print("!!!!!!!!!!!!!!!!!")
    return tuple(vector)

train_path = os.path.join("D:/Corpora/HICO-DET", "orientation_annotation", "ALL_train.json")
test_path = os.path.join("D:/Corpora/HICO-DET", "orientation_annotation", "ALL_test.json")
print(train_path)

check_most_frequent_front = {}
check_most_frequent_up = {}
with open(train_path, 'r') as json_file:
    data = json.load(json_file)

    for img_id, anno in data["_via_img_metadata"].items():
        for region in anno["regions"]:
            front = _ori_dict_to_vec(region["region_attributes"]["front"])
            up = _ori_dict_to_vec(region["region_attributes"]["up"])
            if "category" not in region["region_attributes"]:
                continue
            elif region["region_attributes"]["category"] == "object":
                obj_name = region["region_attributes"]["obj name"]
            elif region["region_attributes"]["category"] == "human":
                obj_name = "human"
            else:
                print(region["region_attributes"]["category"] + " !!!!!!!!!!!!!!!!")
                exit()

            if not all(v == 0 for v in front):
                if obj_name in check_most_frequent_front:
                    check_most_frequent_front[obj_name].append(front)
                else:
                    check_most_frequent_front[obj_name] = [front]

            if not all(v == 0 for v in up):
                if obj_name in check_most_frequent_up:
                    check_most_frequent_up[obj_name].append(up)
                else:
                    check_most_frequent_up[obj_name] = [up]

map_front = {}
for_all_front = []
for key, values in check_most_frequent_front.items():
    map_front[key] = max(values, key=values.count)
    for_all_front += values
map_front["all"] = max(for_all_front, key=for_all_front.count)

map_up = {}
for_all_up = []
for key, values in check_most_frequent_up.items():
    map_up[key] = max(values, key=values.count)
    for_all_up += values
print(map_front)
map_up["all"] = max(for_all_up, key=for_all_up.count)
print(map_up)
up_all = 0
up_correct = 0
front_all = 0
front_correct = 0
missed = 0
with open(test_path, 'r') as json_file:
    data = json.load(json_file)

    for img_id, anno in data["_via_img_metadata"].items():
        for region in anno["regions"]:
            front = _ori_dict_to_vec(region["region_attributes"]["front"])
            up = _ori_dict_to_vec(region["region_attributes"]["up"])
            if "category" not in region["region_attributes"]:
                continue
            elif region["region_attributes"]["category"] == "object":
                obj_name = region["region_attributes"]["obj name"]
            elif region["region_attributes"]["category"] == "human":
                obj_name = "human"
                #continue
            else:
                print(region["region_attributes"]["category"] + " !!!!!!!!!!!!!!!!")
                exit()


            if not all(v == 0 for v in front):
                front_all += 1
                #if obj_name in map_front and front == map_front[obj_name]:
                #    front_correct += 1
                if front == map_front["all"]:
                    front_correct += 1

            if not all(v == 0 for v in up):
                up_all += 1
                #if obj_name in map_up and up == map_up[obj_name]:
                #    up_correct += 1
                if up == map_up["all"]:
                    up_correct += 1

print(str(up_correct) + " - " + str(up_all))
print(str(front_correct) + " - " + str(front_all))

print(str(up_correct / up_all))
print(str(front_correct / front_all))