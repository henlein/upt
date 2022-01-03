import json
import os

train_file = "D:/Corpora/HICO-DET/instances_train2015.json"
test_file = "D:/Corpora/HICO-DET/instances_test2015.json"

split_folder = "D:/Corpora/HICO-DET/hicodet_obj_split"

def merge_train_test(train_data, test_data):
    print(len(train_data["annotation"]))
    print(len(test_data["annotation"]))
    merged_data = {}
    start = len(train_data["annotation"])
    merged_data["annotation"] = train_data["annotation"] + test_data["annotation"]
    merged_data["filenames"] = train_data["filenames"] + test_data["filenames"]
    merged_data["objects"] = train_data["objects"]
    merged_data["verbs"] = train_data["verbs"]
    merged_data["correspondence"] = train_data["correspondence"]
    merged_data["size"] = train_data["size"] + test_data["size"]

    remapped_empty = [x+start for x in test_data["empty"]]
    merged_data["empty"] = train_data["empty"] + remapped_empty
    return merged_data


def map_objects(merged_data):
    object_to_file_dict = {}
    for idx, annotation in enumerate(merged_data["annotation"]):
        for obj in list(set(annotation["object"])):
            if obj in object_to_file_dict:
                object_to_file_dict[obj].append(idx)
            else:
                object_to_file_dict[obj] = [idx]
    return object_to_file_dict


def split_list(input_list, idx_list):
    list_1 = []
    list_2 = []
    for idx, elem in enumerate(input_list):
        if idx in idx_list:
            list_1.append(elem)
        else:
            list_2.append(elem)
    return list_1, list_2


with open(train_file) as json_file:
    train_data = json.load(json_file)

with open(test_file) as json_file:
    test_data = json.load(json_file)

merged = merge_train_test(train_data, test_data)
object_to_file_map = map_objects(merged)

sum = 0
#47701
#target_cats = ["apple", "bicycle", "bottle", "car", "chair", "cup", "dog", "horse", "knife", "umbrella"]
for key, val in object_to_file_map.items():
    #if merged["objects"][key] in target_cats:
    print(merged["objects"][key], len(val))
    sum += len(val)
print("==========")
print(sum)


for key, obj_docs in object_to_file_map.items():
    train_dict = {}
    test_dict = {}

    test_anno, train_anno = split_list(merged["annotation"], obj_docs)
    train_dict["annotation"] = train_anno
    test_dict["annotation"] = test_anno

    test_anno, train_anno = split_list(merged["filenames"], obj_docs)
    train_dict["filenames"] = train_anno
    test_dict["filenames"] = test_anno

    train_dict["objects"] = train_anno
    test_dict["objects"] = test_anno

    train_dict["verbs"] = train_anno
    test_dict["verbs"] = test_anno

    train_dict["correspondence"] = train_anno
    test_dict["correspondence"] = test_anno


    test_anno, train_anno = split_list(merged["size"], obj_docs)
    train_dict["size"] = train_anno
    test_dict["size"] = test_anno

    new_empty = [0 for _ in range(len(merged["annotation"]))]
    for x in merged["empty"]:
        new_empty[x] = 1
    test_anno, train_anno = split_list(new_empty, obj_docs)

    test_empty = []
    for idx, val in enumerate(test_anno):
        if val == 1:
            test_empty.append(idx)
    test_dict["empty"] = test_empty

    train_empty = []
    for idx, val in enumerate(test_anno):
        if val == 1:
            train_empty.append(idx)
    train_dict["empty"] = train_empty

    target_cat = merged["objects"][key]
    target_cat_count = str(len(obj_docs))
    with open(os.path.join(split_folder, target_cat + "_test_" + target_cat_count + ".json"), 'w') as f:
        json.dump(test_dict, f)

    with open(os.path.join(split_folder, target_cat + "_train_" + target_cat_count + ".json"), 'w') as f:
        json.dump(train_dict, f)

