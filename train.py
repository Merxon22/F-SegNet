# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.utils.visualizer import ColorMode
from detectron2.engine import DefaultTrainer
from detectron2.utils.visualizer import ColorMode

from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader

# detectron2.data.datasets.register_coco_instances("ALIS_sample", {}, "/home/PeterYuan/Coding/dataset/ALIS_dataset/ALIS_sample/ALIS_person_sample2021.json", "/home/PeterYuan/Coding/dataset/ALIS_dataset/ALIS_sample/ALIS_person_sample2021")
detectron2.data.datasets.register_coco_instances("ALIS_person", {}, "/home/PeterYuan/Coding/dataset/ALIS_dataset/ALIS_train/ALIS_person/ALIS_person_train2021.json", "/home/PeterYuan/Coding/dataset/ALIS_dataset/ALIS_train/ALIS_person/ALIS_person_train2021")
detectron2.data.datasets.register_coco_instances("ALIS_people", {}, "/home/PeterYuan/Coding/dataset/ALIS_dataset/ALIS_train/ALIS_people/ALIS_people_train2021.json", "/home/PeterYuan/Coding/dataset/ALIS_dataset/ALIS_train/ALIS_people/ALIS_people_train2021")
detectron2.data.datasets.register_coco_instances("ALIS_cat", {}, "/home/PeterYuan/Coding/dataset/ALIS_dataset/ALIS_train/ALIS_cat/ALIS_cat_train2021.json", "/home/PeterYuan/Coding/dataset/ALIS_dataset/ALIS_train/ALIS_cat/ALIS_cat_train2021")
detectron2.data.datasets.register_coco_instances("ALIS_car", {}, "/home/PeterYuan/Coding/dataset/ALIS_dataset/ALIS_train/ALIS_car/ALIS_car_train2021.json", "/home/PeterYuan/Coding/dataset/ALIS_dataset/ALIS_train/ALIS_car/ALIS_car_train2021")
detectron2.data.datasets.register_coco_instances("ALIS_chair", {}, "/home/PeterYuan/Coding/dataset/ALIS_dataset/ALIS_train/ALIS_chair/ALIS_chair_train2021.json", "/home/PeterYuan/Coding/dataset/ALIS_dataset/ALIS_train/ALIS_chair/ALIS_chair_train2021")
#detectron2.data.datasets.register_coco_instances("ALIS_person_val", {}, "/home/PeterYuan/Coding/dataset/ALIS_dataset/ALIS_val/ALIS_person/ALIS_person_val2021.json", "/home/PeterYuan/Coding/dataset/ALIS_dataset/ALIS_val/ALIS_person/ALIS_person_val2021")
detectron2.data.datasets.register_coco_instances("ALIS_sample", {}, "/home/PeterYuan/Coding/dataset/ALIS_dataset/ALIS_sample/ALIS_people_val2021.json", "/home/PeterYuan/Coding/dataset/ALIS_dataset/ALIS_sample/ALIS_person_sample2021")

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("ALIS_person", "ALIS_people", "ALIS_car", "ALIS_cat", "ALIS_chair")
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 2
#cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
#cfg.MODEL.WEIGHTS = "/home/PeterYuan/Coding/detectron/dino_resnet50_pretrain.pth"
cfg.MODEL.WEIGHTS = "/home/PeterYuan/Coding/detectron/mocod.pkl"
#cfg.MODEL.WEIGHTS = ""
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
cfg.SOLVER.MAX_ITER = 500    # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
cfg.SOLVER.STEPS = []        # do not decay learning rate
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512   # faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
# NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.

cfg.OUTPUT_DIR = "/home/PeterYuan/Coding/detectron/train_sample"

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg) 
trainer.resume_or_load(resume=True)
trainer.train()

# #evaluate
# cfg.MODEL.WEIGHTS = "/home/PeterYuan/Coding/detectron/train_pretrained_mask_rcnn/model_final.pth"
# cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
# predictor = DefaultPredictor(cfg)

# evaluator = COCOEvaluator("ALIS_person_val")
# val_loader = build_detection_test_loader(cfg, "ALIS_person_val")
# print(inference_on_dataset(trainer.model, val_loader, evaluator))