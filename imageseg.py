import numpy as np
import os, json, cv2, random
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
import pickle, time
import pycocotools.mask as mask_util
from multiprocessing import Pool
from itertools import groupby

start_time = time.time()

image_dir = '/home/PeterYuan/Coding/dataset/coco_test2017'
max_image = 0

def binary_mask_to_rle(binary_mask):
    rle = {'counts': [], 'size': list(binary_mask.shape)}
    counts = rle.get('counts')
    for i, (value, elements) in enumerate(groupby(binary_mask.ravel(order='F'))):
        if i == 0 and value == 1:
            counts.append(0)
        counts.append(len(list(elements)))
    return rle

def instance_seg(img):
    outputs = predictor(img)["instances"]

    pred_classes = outputs.pred_classes

    pred_boxes = outputs.pred_boxes.to("cpu").tensor.numpy()

    pred_masks = np.asfortranarray(outputs.pred_masks.to("cpu").numpy().astype(np.uint8))
    rle_masks = []
    for mask in pred_masks:
        mask = binary_mask_to_rle(mask)
        mask = mask_util.frPyObjects(mask, mask.get("size")[0], mask.get("size")[1])
        rle_masks.append(mask)
        # print(mask_util.decode(mask))
        # print(mask)

    scores = outputs.scores
    return (pred_classes, pred_boxes, rle_masks, scores)

#model
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
predictor = DefaultPredictor(cfg)
seg_result = []

image_processed = 0
for image in os.listdir(image_dir):
    if max_image != 0 and image_processed >= max_image:
        break
    if image.lower().endswith(".jpg") or image.lower().endswith(".jpeg") or image.lower().endswith(".png") or image.lower().endswith(".bmp") or image.lower().endswith(".tif"):
        pred_classes, pred_boxes, pred_masks, scores = instance_seg(cv2.imread(os.path.join(image_dir, image)))
        image_path = os.path.join(image_dir, image)
        seg_result.append((image_path, pred_classes, pred_boxes, pred_masks, scores))
        image_processed += 1

pickle_out = open("seg_result_coco_test2017.pkl", "wb")
pickle.dump(seg_result, pickle_out)
pickle_out.close()

finish_time = time.time()

print(f"Processed {image_processed} images in {round(finish_time - start_time, 2)} secondsn, {round((finish_time - start_time) / image_processed, 2)}s/image")