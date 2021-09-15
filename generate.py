from sys import maxsize
import torch, torchvision
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
import numpy as np
import os, json, cv2, random
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
import time, pickle
from multiprocessing import Pool
from pycocotools import mask as mask_utils

start_time = time.time()

#parameters, source_dir1 should contain files from coco dataset, source_dir2 can be any images
source_dir2 = '/home/PeterYuan/Coding/dataset/coco_train2017'
destination_dir = '/home/PeterYuan/Coding/dataset/ALIS_dataset/ALIS_sample'
seg_results = '/home/PeterYuan/Coding/detectron/seg_result_coco_test2017.pkl'
data_usage = "sample"    #"train" or "val"
existing_img_count = 0
target_object_index = 0    #person = 0, car = 2, cat = 15, dog = 16, chair = 56
available_images = []   #a list of tuple that has structure: : (img_file_name, prediction_box, prediction_mask)
max_result_img = 0  #an int, since looping through dir1 is a time-consuming tasks, 0 = loop through all images in dir1, 500 = loop through the first 500 images in dir1, etc.
max_dir2_img = 500
confidence_requirement = 0.98
minimum_x, minimum_y = 100, 200 #minumum pixel width and height to be considered
object_class_name = {
    0: "person",
    2: "car",
    15: "cat",
    16: "dog",
    56: "chair"
}

#load model
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
predictor = DefaultPredictor(cfg)

#setup output directory
generated_dir = os.path.join(destination_dir, "ALIS_" + object_class_name[target_object_index] + "_" + data_usage + "2021")
mask_dir = os.path.join(destination_dir, "annotations")
if not os.path.exists(generated_dir):
    os.makedirs(generated_dir)
if not os.path.exists(mask_dir):
    os.makedirs(mask_dir)

def instance_seg(img):
    outputs = predictor(img)
    pred_classes = outputs["instances"].pred_classes
    pred_boxes = outputs["instances"].pred_boxes.to("cpu").tensor.numpy()
    pred_masks = outputs["instances"].pred_masks
    scores = outputs["instances"].scores
    return (pred_classes, pred_boxes, pred_masks, scores)

def generate_spliced_image(input):
    #read parameters
    id, path1, path2, box, mask = input
    #read the two images
    img1 = cv2.imread(path1)
    img2 = cv2.imread(path2)
    #crop the obejct out of img1
    #print(img1.shape, mask.shape)
    cropped = cv2.bitwise_and(img1, img1, mask=mask)
    x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
    cropped = cropped[y1:y2, x1:x2]
    #set scale, flip, offset
    max_scale = min((img2.shape[0] / cropped.shape[0]), img2.shape[1] / cropped.shape[1])
    scale = random.uniform(max_scale * 0.3, max_scale * 0.8)
    flip = random.randint(-1, 1)
    #print(img2.shape)
    offset_x = random.randint(0, int(img2.shape[1] - cropped.shape[1] * scale))
    offset_y = random.randint(0, int(img2.shape[0] - cropped.shape[0] * scale))
    #apply cropped area to img2
    mask_inv = np.logical_not(mask).astype(np.uint8)[y1:y2, x1:x2]
    #resize
    mask_inv = cv2.resize(mask_inv, (int((x2 - x1) * scale), int((y2 - y1) * scale)))
    cropped = cv2.resize(cropped, (mask_inv.shape[1], mask_inv.shape[0]))
    #flip
    if random.uniform(0, 1) >= 0.5:
        mask_inv = cv2.flip(mask_inv, flip)
        cropped = cv2.flip(cropped, flip)
    #locate roi
    roi = img2[offset_y:cropped.shape[0] + offset_y, offset_x:cropped.shape[1] + offset_x]
    roi = cv2.bitwise_and(roi, roi, mask = mask_inv)
    roi = cv2.add(cropped, roi)
    img2[offset_y:cropped.shape[0] + offset_y, offset_x:cropped.shape[1] + offset_x] = roi
    truth = np.zeros((img2.shape[0], img2.shape[1]))
    truth[offset_y:cropped.shape[0] + offset_y, offset_x:cropped.shape[1] + offset_x] = (cv2.bitwise_not(mask_inv).astype(np.uint8) - 254) * 255
    mask_truth = np.zeros([img2.shape[0], img2.shape[1], 3])
    for x in range(mask_truth.shape[1]):
        for y in range(mask_truth.shape[0]):
            mask_truth[y, x] = np.array([truth[y, x], truth[y, x], truth[y, x]])
    #save image
    cv2.imwrite(os.path.join(generated_dir, (format(id, "09d") + ".jpg")), img2)
    cv2.imwrite(os.path.join(mask_dir, (format(id, "09d") + "_forged_0" + ".png")), mask_truth)
    #return img2, mask_truth

#search for images with desired objects in dir1
img_looped = 0
start_load_result = time.time()
seg_results = pickle.load(open(seg_results, "rb"))
print(f"Loading seg_result takes {round(time.time() - start_load_result, 2)} seconds")
for result in seg_results:
    if max_result_img != 0 and img_looped >= max_result_img:
        break
    #unpack result
    image_path, pred_classes, pred_boxes, pred_masks, scores = result
    #print(type(image_path), image_path)
    #check if the desired object exists. If so, find the one with highest confidence
    object_index = -1
    for index in range(pred_classes.shape[0]):
        pred_box = pred_boxes[index].astype(np.int_)
        #print(pred_box)
        size_satifise = (pred_box[2] - pred_box[0] >= minimum_x) and (pred_box[3] - pred_box[1] >= minimum_y)
        if pred_classes[index].item() == target_object_index and scores[index].item() >= confidence_requirement and size_satifise:
            object_index = index
            break
    #keep image if it has desires object
    if object_index != -1:
        #print(type(mask_utils.decode(pred_masks[object_index])))
        available_images.append((image_path, pred_boxes[object_index].astype(np.int_), mask_utils.decode(pred_masks[object_index])))
        #cv2.imwrite(os.path.join("/home/PeterYuan/Coding/dataset/selected", image), cv2.imread(os.path.join(source_dir1, image)))
    img_looped += 1

# for image in os.listdir(source_dir1):
#     if max_dir1_img != 0 and img_looped >= max_dir1_img:
#         break
#     #get image output
#     pred_classes, pred_boxes, pred_masks, scores = instance_seg(cv2.imread(os.path.join(source_dir1, image)))
#     #check if the desired object exists. If so, find the one with highest confidence
#     object_index = -1
#     for index in range(pred_classes.shape[0]):
#         pred_box = pred_boxes[index].astype(np.int_)
#         #print(pred_box)
#         size_satifise = (pred_box[2] - pred_box[0] >= minimum_x) and (pred_box[3] - pred_box[1] >= minimum_y)
#         if pred_classes[index].item() == target_object_index and scores[index].item() >= confidence_requirement and size_satifise:
#             object_index = index
#             break
#     #decide whether to keep the image
#     if object_index != -1:
#         available_images.append((image, pred_boxes[object_index].astype(np.int_), pred_masks[object_index].type(torch.int).to("cpu").numpy().astype(np.uint8)))
#         #cv2.imwrite(os.path.join("/home/PeterYuan/Coding/dataset/selected", image), cv2.imread(os.path.join(source_dir1, image)))
#     img_looped += 1

print(f"Available images with '{object_class_name[target_object_index]}' class: {len(available_images)}")

start_generate_iter = time.time()
#create generate_iter
image_itered = 0
generate_iter = []
for image2 in os.listdir(source_dir2):
    if max_dir2_img != 0 and image_itered >= max_dir2_img:
        break
    #execute if the file is an image
    if image2.lower().endswith(".jpg") or image2.lower().endswith(".jpeg") or image2.lower().endswith(".png") or image2.lower().endswith(".bmp") or image2.lower().endswith(".tif"):
        #select random image from dir1
        image1 = random.choice(available_images)
        #generate parameters
        file_id = image_itered + existing_img_count
        image1_path, box, mask = image1
        image2_path = os.path.join(source_dir2, image2)
        generate_iter.append((file_id, image1_path, image2_path, box, mask))
        
        image_itered += 1
print(f"Generate_iter takes {round(time.time() - start_generate_iter, 2)} seconds")

#multiprocess pool
start_generate_time = time.time()
p = Pool(78)
result = p.map(generate_spliced_image, generate_iter)
p.close()
p.join()
print(f"Generate {len(generate_iter)} images takes {round(time.time() - start_generate_time, 2)} seconds, {round((time.time() - start_generate_time) / len(generate_iter), 4)}s/image")

# #generate image
# image_generated = 0

# for image2 in os.listdir(source_dir2):
#     if max_dir2_img != 0 and image_generated >= max_dir2_img:
#         break
#     if image2.lower().endswith(".jpg") or image2.lower().endswith(".jpeg") or image2.lower().endswith(".png") or image2.lower().endswith(".bmp") or image2.lower().endswith(".tif"):
#         image1 = random.choice(available_images)
#         output, mask = generate_spliced_image(os.path.join(source_dir1, image1[0]), os.path.join(source_dir2, image2), image1[1], image1[2])
#         #print(type((os.path.splitext(image2))))
#         cv2.imwrite(os.path.join(generated_dir, (format(image_generated + existing_img_count, "09d") + ".jpg")), output)
#         cv2.imwrite(os.path.join(mask_dir, (format(image_generated+  existing_img_count, "09d") + "_forged_0" + ".png")), mask)
#         image_generated += 1

print(f"Finished in {time.time() - start_time} seconds")