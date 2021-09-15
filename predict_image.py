# Some basic setup:
# Setup detectron2 logger
import torch, torchvision
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random
import time

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.utils.visualizer import ColorMode

detectron2.data.datasets.register_coco_instances("ALIS_person", {}, "/home/PeterYuan/Coding/dataset/ALIS_dataset/ALIS_train/ALIS_person/ALIS_person_train2021.json", "/home/PeterYuan/Coding/dataset/ALIS_dataset/ALIS_train/ALIS_person/ALIS_person_train2021")

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = "./f_segnet_final.pth"
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.65
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)

cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
cfg.SOLVER.MAX_ITER = 2500    # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
cfg.SOLVER.STEPS = []        # do not decay learning rate
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512   # faster, and good enough for this toy dataset (default: 512)

predictor = DefaultPredictor(cfg)

img = cv2.imread("/home/PeterYuan/Coding/dataset/outcome/000020096.jpg")
outputs = predictor(img)
v = Visualizer(img[:, :, ::-1],
                metadata=MetadataCatalog.get("ALIS_person"),
                scale=1, 
                instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels. This option is only available for segmentation models
)
out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

cv2.imwrite("./predicted.jpg", out.get_image()[:,:,::-1])
