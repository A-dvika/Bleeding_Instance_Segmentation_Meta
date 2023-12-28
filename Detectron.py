import os
import json
import cv2
import matplotlib.pyplot as plt
import detectron2
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures import BoxMode
from detectron2 import model_zoo
import numpy as np
setup_logger()

import os
import json
import cv2
from detectron2.structures import BoxMode  # Assuming you're using Detectron2
import os
import json
from detectron2.structures import BoxMode  # Assuming you're using Detectron2

def get_custom_dataset_dicts(img_dir):
    json_file = os.path.join(img_dir, "image_mask_points.json")  # Update the JSON filename as per your dataset
    with open(json_file) as f:
        imgs_anns = json.load(f)

    dataset_dicts = []
    for idx, (filename, v) in enumerate(imgs_anns.items()):
        record = {}
        
        # Check if 'annotations' key exists in the dictionary
        if 'annotations' in v:
            height, width = 0, 0  # Placeholder values
            if 'file_name' in v:
                filename = os.path.join(img_dir, v["file_name"])
                height, width = 480, 640  # Replace with actual image height and width
                
            record["file_name"] = filename
            record["image_id"] = idx
            record["height"] = height
            record["width"] = width
          
            annos = v["annotations"]
            objs = []
            for anno in annos:
                # Adjust to handle the nested segmentation structure
                if isinstance(anno["segmentation"], list) and len(anno["segmentation"]) > 0:
                    px = [x[0] for x in anno["segmentation"][0]]
                    py = [y[1] for y in anno["segmentation"][0]]
                    
                    poly = [(float(x) + 0.5, float(y) + 0.5) for x, y in zip(px, py)]
                    poly = [p for x in poly for p in x]

                    obj = {
                        "bbox": [min(px), min(py), max(px), max(py)],
                        "bbox_mode": BoxMode.XYXY_ABS,
                        "segmentation": [poly],
                        "category_id": 0,  # Update category ID as needed
                    }
                    objs.append(obj)
                else:
                    print(f"Invalid segmentation format for annotation {filename}")
            
            record["annotations"] = objs
            dataset_dicts.append(record)
        else:
            print(f"Missing 'annotations' key in annotation for {filename}")

    return dataset_dicts


# Usage example
dataset_directory = r"C:\Users\HP\Desktop\Ultralytics_RTDeTr\Segmentation\Train"
custom_dataset_dicts = get_custom_dataset_dicts(dataset_directory)
# Now, you can use `custom_dataset_dicts` for further processing or registration with Detectron2 DatasetCatalog
# Define a function to register your custom dataset
def register_custom_dataset(name, data):
    DatasetCatalog.register(name, lambda: data)
    MetadataCatalog.get(name).set(thing_classes=["Bleeding"])  # Update with your class name

# Usage example
register_custom_dataset("custom_train", custom_dataset_dicts)  # Assuming `custom_dataset_dicts` is your processed dataset

# Access the metadata for your dataset
custom_metadata = MetadataCatalog.get("custom_train")

# Configure and train the model
cfg = get_cfg()
cfg.MODEL.DEVICE = "cpu"

cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("custom_train",)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 0
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025
cfg.SOLVER.MAX_ITER = 300
cfg.SOLVER.STEPS = []
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()

# cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained

# cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set a custom testing threshold
# from detectron2.engine import DefaultPredictor
# predictor = DefaultPredictor(cfg)
# from detectron2.utils.visualizer import ColorMode
# dataset_dicts = get_balloon_dicts("bleeding_dataset/bleed/val")
# for d in random.sample(dataset_dicts, 1):    
#     im = cv2.imread(d["file_name"])
#     outputs = predictor(im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
#     v = Visualizer(im[:, :, ::-1],
#                    metadata=balloon_metadata, 
#                    scale=0.5, 
#                    instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels. This option is only available for segmentation models
#     )
#     out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
#     cv2_imshow(out.get_image()[:, :, ::-1])