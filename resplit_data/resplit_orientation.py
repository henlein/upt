import json
import os.path
import random
random.seed(123)

anju_anno_file = "C:/Users/Alexander/Downloads/via234_780 items_Dec 23.json"
split_folder = "D:/Corpora/HICO-DET/orientation_annotation"

cat_to_image_map = {}
with open(anju_anno_file) as json_file:
    data = json.load(json_file)
    data = data["_via_img_metadata"]

    keys = list(data.keys())
    for anno_key in keys:
        anno_value = data[anno_key]
        for region in anno_value["regions"]:
            if "obj name" in region["region_attributes"]:
                obj_name = region["region_attributes"]["obj name"]
                if obj_name in cat_to_image_map:
                    cat_to_image_map[obj_name].append(anno_key)
                else:
                    cat_to_image_map[obj_name] = [anno_key]

    print(keys)
    random.shuffle(keys)
    train_keys = keys[:650]
    train_json = {}
    for key in train_keys:
        train_json[key] = data[key]
    train_save_json = {"_via_img_metadata": train_json}
    with open(os.path.join(split_folder, "ALL_train.json"), 'w') as f:
        json.dump(train_save_json, f)

    test_keys = keys[650:]
    test_json = {}
    for key in test_keys:
        test_json[key] = data[key]
    test_save_json = {"_via_img_metadata": test_json}
    with open(os.path.join(split_folder, "ALL_test.json"), 'w') as f:
        json.dump(test_save_json, f)


target_cats = ["apple", "bicycle", "bottle", "car", "chair", "cup", "dog", "horse", "knife", "umbrella"]

for target_cat in target_cats:
    if target_cat in cat_to_image_map and len(cat_to_image_map[target_cat]) > 100:
        marked_files = []

        test_json = {}
        for anno in cat_to_image_map[target_cat]:
            test_json[anno] = data[anno]
            marked_files.append(anno)

        train_json = {}
        for anno_key, anno_value in data.items():
            if anno_key not in marked_files:
                train_json[anno_key] = anno_value

        test_save_json = {"_via_img_metadata": test_json}
        with open(os.path.join(split_folder, target_cat + "_test.json"), 'w') as f:
            json.dump(test_save_json, f)

        train_save_json = {"_via_img_metadata": train_json}
        with open(os.path.join(split_folder, target_cat + "_train.json"), 'w') as f:
            json.dump(train_save_json, f)