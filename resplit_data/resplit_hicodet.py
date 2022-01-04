import json
import os
from tqdm import tqdm
train_file = "D:/Corpora/HICO-DET/instances_train2015.json"
test_file = "D:/Corpora/HICO-DET/instances_test2015.json"

split_folder = "D:/Corpora/HICO-DET/hicodet_obj_split"
split_verb_folder = "D:/Corpora/HICO-DET/hicodet_verb_split"
split_hoi_folder = "D:/Corpora/HICO-DET/hicodet_hoi_split"

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


def map_verbs(merged_data):
    verb_to_file_dict = {}
    for idx, annotation in enumerate(merged_data["annotation"]):
        for obj in list(set(annotation["verb"])):
            if obj in verb_to_file_dict:
                verb_to_file_dict[obj].append(idx)
            else:
                verb_to_file_dict[obj] = [idx]
    return verb_to_file_dict


def map_hoi(merged_data):
    hoi_to_file_dict = {}
    for idx, annotation in enumerate(merged_data["annotation"]):
        for obj in list(set(annotation["hoi"])):
            if obj in hoi_to_file_dict:
                hoi_to_file_dict[obj].append(idx)
            else:
                hoi_to_file_dict[obj] = [idx]
    return hoi_to_file_dict


def split_list(input_list, idx_list):
    list_1 = []
    list_2 = []
    for idx, elem in enumerate(input_list):
        if idx in idx_list:
            list_1.append(elem)
        else:
            list_2.append(elem)
    return list_1, list_2


def create_object_filter(merged_data):
    object_to_file_map = map_objects(merged_data)
    for key, val in object_to_file_map.items():
        # if merged["objects"][key] in target_cats:
        print(merged_data["objects"][key], len(val))

    for key, obj_docs in object_to_file_map.items():
        train_dict = {}
        test_dict = {}

        test_anno, train_anno = split_list(merged_data["annotation"], obj_docs)
        train_dict["annotation"] = train_anno
        test_dict["annotation"] = test_anno

        test_anno, train_anno = split_list(merged_data["filenames"], obj_docs)
        train_dict["filenames"] = train_anno
        test_dict["filenames"] = test_anno

        train_dict["objects"] = train_anno
        test_dict["objects"] = test_anno

        train_dict["verbs"] = train_anno
        test_dict["verbs"] = test_anno

        train_dict["correspondence"] = train_anno
        test_dict["correspondence"] = test_anno

        test_anno, train_anno = split_list(merged_data["size"], obj_docs)
        train_dict["size"] = train_anno
        test_dict["size"] = test_anno

        new_empty = [0 for _ in range(len(merged_data["annotation"]))]
        for x in merged_data["empty"]:
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

        target_cat = merged_data["objects"][key]
        target_cat_count = str(len(obj_docs))
        with open(os.path.join(split_folder, target_cat + "_test_" + target_cat_count + ".json"), 'w') as f:
            json.dump(test_dict, f)

        with open(os.path.join(split_folder, target_cat + "_train_" + target_cat_count + ".json"), 'w') as f:
            json.dump(train_dict, f)


def create_verb_filter(merged_data):
    verb_to_file_map = map_verbs(merged_data)
    for key, val in verb_to_file_map.items():
        print(merged_data["verbs"][key], len(val))

    for key, obj_docs in verb_to_file_map.items():
        train_dict = {}
        test_dict = {}

        test_anno, train_anno = split_list(merged_data["annotation"], obj_docs)
        train_dict["annotation"] = train_anno
        test_dict["annotation"] = test_anno

        test_anno, train_anno = split_list(merged_data["filenames"], obj_docs)
        train_dict["filenames"] = train_anno
        test_dict["filenames"] = test_anno

        train_dict["objects"] = train_anno
        test_dict["objects"] = test_anno

        train_dict["verbs"] = train_anno
        test_dict["verbs"] = test_anno

        train_dict["correspondence"] = train_anno
        test_dict["correspondence"] = test_anno

        test_anno, train_anno = split_list(merged_data["size"], obj_docs)
        train_dict["size"] = train_anno
        test_dict["size"] = test_anno

        new_empty = [0 for _ in range(len(merged_data["annotation"]))]
        for x in merged_data["empty"]:
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

        target_cat = merged_data["verbs"][key]
        target_cat_count = str(len(obj_docs))
        with open(os.path.join(split_verb_folder, target_cat + "_test_" + target_cat_count + ".json"), 'w') as f:
            json.dump(test_dict, f)

        with open(os.path.join(split_verb_folder, target_cat + "_train_" + target_cat_count + ".json"), 'w') as f:
            json.dump(train_dict, f)


def create_hoi_filter(merged_data):
    object_to_file_map = map_objects(merged_data)
    verb_to_file_map = map_verbs(merged_data)
    hoi_to_file_map = map_hoi(merged_data)

    hoid_to_vo_map = {}
    for x, y, z in merged_data["correspondence"]:
        hoid_to_vo_map[x] = (y, z)
    print(hoid_to_vo_map)

    for key, val in hoi_to_file_map.items():
        obj, verb = hoid_to_vo_map[key]
        print(merged_data["objects"][obj] + " - " + merged_data["verbs"][verb], len(val))

    for key, _ in tqdm(hoi_to_file_map.items()):
        train_dict = {}
        test_dict = {}

        obj, verb = hoid_to_vo_map[key]
        obj_docs = object_to_file_map[obj] + verb_to_file_map[verb]
        obj_docs = sorted(list(set(obj_docs)))
        #print(obj_docs)
        test_anno, train_anno = split_list(merged_data["annotation"], obj_docs)
        train_dict["annotation"] = train_anno
        test_dict["annotation"] = test_anno

        test_anno, train_anno = split_list(merged_data["filenames"], obj_docs)
        train_dict["filenames"] = train_anno
        test_dict["filenames"] = test_anno

        train_dict["objects"] = train_anno
        test_dict["objects"] = test_anno

        train_dict["verbs"] = train_anno
        test_dict["verbs"] = test_anno

        train_dict["correspondence"] = train_anno
        test_dict["correspondence"] = test_anno

        test_anno, train_anno = split_list(merged_data["size"], obj_docs)
        train_dict["size"] = train_anno
        test_dict["size"] = test_anno

        new_empty = [0 for _ in range(len(merged_data["annotation"]))]
        for x in merged_data["empty"]:
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


        target_verb = merged_data["verbs"][verb]
        target_obj = merged_data["objects"][obj]
        target_cat_count = str(len(obj_docs))
        with open(os.path.join(split_hoi_folder, target_obj + "_" + target_verb + "_test_" + target_cat_count + ".json"), 'w') as f:
            json.dump(test_dict, f)

        with open(os.path.join(split_hoi_folder, target_obj + "_" + target_verb + "_train_" + target_cat_count + ".json"), 'w') as f:
            json.dump(train_dict, f)


with open(train_file) as json_file:
    train_data = json.load(json_file)

with open(test_file) as json_file:
    test_data = json.load(json_file)

merged_json = merge_train_test(train_data, test_data)

#create_object_filter(merged_json)

#create_verb_filter(merged_json)

create_hoi_filter(merged_json)
