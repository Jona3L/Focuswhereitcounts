from detectron2.structures import BoxMode
from pathlib import Path
import json
import cv2
from collections import defaultdict

def get_sifr_dicts(root, mode):
    """
    Load and parse the official COCO 'instances_val2014.json' in a Detectron2-friendly format.
    NOTE: This assumes your images are in:
         path/QAGNet-main/coco_dataset/test/val2014/
    and your annotations file is at:
         path/QAGNet-main/coco_dataset/test/val2014/instances_val2014.json

    root: the root path you pass in (e.g., "path/QAGNet-main/coco_dataset")
    mode: typically "test" or "val2014" or something you use in your code
    """
    # Keep the same variable style
    root = Path(root)
    
    # 1) Point to the JSON file (COCO 'instances_val2014.json')
    
    # json_file = f"path/salience_llava/dataset/split/Train_coco.json"
    
    json_file = f"path/salience_llava/dataset/vizwiz/vizwiz/vizwiz_test/annotations/instances.json"
    
    # If you later switch to path/QAGNet-main/coco_dataset/annotations/instances_val2014.json,
    # just change it here. The rest of the code remains the same.

    with open(json_file, "r") as f:
        imgs_anns = json.load(f)
    images_anns = imgs_anns["images"]
    coco_annotations = imgs_anns["annotations"]

    # 3) Group all annotations by the image_id so we can attach them to the correct image
    annos_by_imageid = defaultdict(list)
    for ann in coco_annotations:
        image_id = ann["image_id"]
        annos_by_imageid[image_id].append(ann)

    dataset_dicts = []

    # We’ll still call it new_val_path so you don’t break anything else:
    # new_val_path = Path("path/salience_llava/dataset/split/train")
    new_val_path = Path("path/salience_llava/dataset/vizwiz/vizwiz/vizwiz_test/images")
    

    # 5) Now build each record from the images array
    for idx, anno in enumerate(images_anns):
        record = {}

        # 'anno' here is actually an "image info" dict:
        # {
        #   "file_name": "COCO_val2014_000000000042.jpg",
        #   "height": 480,
        #   "width": 640,
        #   "id": 42
        # }
        filename = str(new_val_path / anno["file_name"])

        # If you want to read the actual image shape from disk:
        #   height, width = cv2.imread(filename).shape[:2]
        # But typically, COCO already provides height/width in the JSON:
        height = anno["height"]
        width = anno["width"]
        img_id = anno["id"]  # same as "image_id" in the old SIFR code

        record["file_name"] = filename
        record["image_id"] = img_id
        record["height"] = height
        record["width"] = width

        objs = []
        ranker_order = []

        # 6) For each annotation belonging to this image
        for obj_anno in annos_by_imageid[img_id]:
            # The old code references "gt_rank", which does NOT exist in standard COCO.
            # If you want to keep the same structure, we can artificially set "gt_rank" to 0.
            # That way your "assert len(ranker_order) == len(record['annotations'])" remains valid.
            ranker_order.append(0)

            obj = {
                "bbox": obj_anno["bbox"],  # in COCO: [x, y, w, h]
                "bbox_mode": BoxMode.XYWH_ABS,
                "segmentation": obj_anno.get("segmentation", []),
                "category_id": 0,  # or use obj_anno["category_id"] if needed
                "gt_rank": 0       # placeholder for your SIFR code
            }
            objs.append(obj)

        record["annotations"] = objs

        # 7) Keep the same assertion so your code flow won’t break
        assert len(ranker_order) == len(record["annotations"])

        dataset_dicts.append(record)

    return dataset_dicts[:]
