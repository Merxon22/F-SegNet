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

import argparse

def main():
    parser = argparse.ArgumentParser(description='Compute normal map of an image')

    parser.add_argument('-d', '--image_directory', type = str)
    parser.add_argument('-an', '--annotation_path', type = str)
    parser.add_argument('-w', '--weights_directory', type = str)
    parser.add_argument('-t', '--score_threshold', type = float, default=0.5)
    args = parser.parse_args()

    # detectron2.data.datasets.register_coco_instances("ALIS_person_val", {}, "/home/PeterYuan/Coding/dataset/ALIS_dataset/ALIS_val/ALIS_person/ALIS_person_val2021.json", "/home/PeterYuan/Coding/dataset/ALIS_dataset/ALIS_val/ALIS_person/ALIS_person_val2021")
    # detectron2.data.datasets.register_coco_instances("ALIS_people_val", {}, "/home/PeterYuan/Coding/dataset/ALIS_dataset/ALIS_val/ALIS_people/ALIS_people_val2021.json", "/home/PeterYuan/Coding/dataset/ALIS_dataset/ALIS_val/ALIS_people/ALIS_people_val2021")
    # detectron2.data.datasets.register_coco_instances("ALIS_car_val", {}, "/home/PeterYuan/Coding/dataset/ALIS_dataset/ALIS_val/ALIS_car/ALIS_car_val2021.json", "/home/PeterYuan/Coding/dataset/ALIS_dataset/ALIS_val/ALIS_car/ALIS_car_val2021")
    # detectron2.data.datasets.register_coco_instances("ALIS_cat_val", {}, "/home/PeterYuan/Coding/dataset/ALIS_dataset/ALIS_val/ALIS_cat/ALIS_cat_val2021.json", "/home/PeterYuan/Coding/dataset/ALIS_dataset/ALIS_val/ALIS_cat/ALIS_cat_val2021")
    # detectron2.data.datasets.register_coco_instances("ALIS_chair_val", {}, "/home/PeterYuan/Coding/dataset/ALIS_dataset/ALIS_val/ALIS_chair/ALIS_chair_val2021.json", "/home/PeterYuan/Coding/dataset/ALIS_dataset/ALIS_val/ALIS_chair/ALIS_chair_val2021")

    detectron2.data.datasets.register_coco_instances(args.image_directory, {}, args.annotation_path, args.image_directory)

    root_dir = args.weights_directory
    weight_path = "model_final.pth"

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = (args.image_directory)
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 2
    #cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
    cfg.SOLVER.MAX_ITER = 40000    # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
    cfg.SOLVER.STEPS = []        # do not decay learning rate
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512   # faster, and good enough for this toy dataset (default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
    # NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.

    cfg.OUTPUT_DIR = root_dir

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg) 
    trainer.resume_or_load(resume=True)
    #trainer.train()

    #evaluate
    cfg.MODEL.WEIGHTS = os.path.join(root_dir, weight_path)
    #cfg.MODEL.WEIGHTS = "./f_segnet_final.pth"
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.score_threshold
    predictor = DefaultPredictor(cfg)

    evaluator = COCOEvaluator(args.image_directory)
    val_loader = build_detection_test_loader(cfg, args.image_directory)
    result = inference_on_dataset(trainer.model, val_loader, evaluator)
    print(result)

    # evaluator1 = COCOEvaluator("ALIS_person_val")
    # val_loader1 = build_detection_test_loader(cfg, "ALIS_person_val")
    # result1 = inference_on_dataset(trainer.model, val_loader1, evaluator1)

    # evaluator2 = COCOEvaluator("ALIS_people_val")
    # val_loader2 = build_detection_test_loader(cfg, "ALIS_people_val")
    # result2 = inference_on_dataset(trainer.model, val_loader2, evaluator2)

    # evaluator3 = COCOEvaluator("ALIS_car_val")
    # val_loader3 = build_detection_test_loader(cfg, "ALIS_car_val")
    # result3 = inference_on_dataset(trainer.model, val_loader3, evaluator3)

    # evaluator4 = COCOEvaluator("ALIS_cat_val")
    # val_loader4 = build_detection_test_loader(cfg, "ALIS_cat_val")
    # result4 = inference_on_dataset(trainer.model, val_loader4, evaluator4)

    # evaluator5 = COCOEvaluator("ALIS_chair_val")
    # val_loader5 = build_detection_test_loader(cfg, "ALIS_chair_val")
    # result5 = inference_on_dataset(trainer.model, val_loader5, evaluator5)

    # print("==================Person==================")
    # print(result1)
    # print("==================People==================")
    # print(result2)
    # print("==================Car==================")
    # print(result3)
    # print("==================Cat==================")
    # print(result4)
    # print("==================Chair==================")
    # print(result5)

if __name__ == "__main__":
    main()
