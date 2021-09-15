import datetime
import json
import os
import re
import fnmatch
from PIL import Image
import numpy as np
from pycococreatortools import pycococreatortools
import time
from multiprocessing import Pool

ROOT_DIR = '/home/PeterYuan/Coding/dataset/ALIS_dataset/ALIS_sample'
IMAGE_DIR = os.path.join(ROOT_DIR, "ALIS_person_sample2021")
ANNOTATION_DIR = os.path.join(ROOT_DIR, "annotations")

INFO = {
    "description": "ALIS dataset",
    "url": "",
    "version": "0.1.0",
    "year": 2021,
    "contributor": "Hao Yuan",
    "date_created": datetime.datetime.utcnow().isoformat(' ')
}

LICENSES = [
    {
        "id": 1,
        "name": "Attribution-NonCommercial-ShareAlike License",
        "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/"
    }
]

CATEGORIES = [
    {
        'id': 1,
        'name': 'forged',
        'supercategory': 'forged',
    }
]

def filter_for_jpeg(root, files):
    file_types = ['*.jpeg', '*.jpg']
    file_types = r'|'.join([fnmatch.translate(x) for x in file_types])
    files = [os.path.join(root, f) for f in files]
    files = [f for f in files if re.match(file_types, f)]
    
    return files

def filter_for_annotations(root, files, image_filename):
    file_types = ['*.png']
    file_types = r'|'.join([fnmatch.translate(x) for x in file_types])
    basename_no_extension = os.path.splitext(os.path.basename(image_filename))[0]
    file_name_prefix = basename_no_extension + '.*'
    files = [os.path.join(root, f) for f in files]
    files = [f for f in files if re.match(file_types, f)]
    files = [f for f in files if re.match(file_name_prefix, os.path.splitext(os.path.basename(f))[0])]

    return files

coco_output = {
    "info": INFO,
    "licenses": LICENSES,
    "categories": CATEGORIES,
    "images": [],
    "annotations": []
}

def main():


    image_id = 1
    
    # filter for jpeg images
    for root, _, files in os.walk(IMAGE_DIR):
        image_files = filter_for_jpeg(root, files)

        #generate iter_list
        iter_list = []
        for image in image_files:
            iter_list.append((image, image_id))
            image_id += 1
        
        p = Pool()
        results = p.map(process_image_, iter_list)
        p.close()
        p.join()

        for image, annotation in results:
            coco_output["images"].append(image)
            coco_output["annotations"].append(annotation)

    print(type(coco_output["images"]), len(coco_output["images"]))

    with open('{}/ALIS_people_val2021.json'.format(ROOT_DIR), 'w') as output_json_file:
        json.dump(coco_output, output_json_file)


def process_image_(input):
    return process_image(input[0], input[1])

def process_image(image_filename, image_id):
    image = Image.open(image_filename)
    image_info = pycococreatortools.create_image_info(
        image_id, os.path.basename(image_filename), image.size)
    #print(type(image_info))
    coco_output["images"].append(image_info)
    #print(len(coco_output["images"]))

    # filter for associated png annotations
    for root, _, files in os.walk(ANNOTATION_DIR):
        annotation_files = filter_for_annotations(root, files, image_filename)

        # go through each associated annotation
        for annotation_filename in annotation_files:
            
            #print(annotation_filename)
            class_id = [x['id'] for x in CATEGORIES if x['name'] in annotation_filename][0]

            category_info = {'id': class_id, 'is_crowd': 'crowd' in image_filename}
            binary_mask = np.asarray(Image.open(annotation_filename)
                .convert('1')).astype(np.uint8)
            
            annotation_info = pycococreatortools.create_annotation_info(
                image_id, image_id, category_info, binary_mask,
                image.size, tolerance=1)

            #print(type(annotation_info))
            if annotation_info is not None:
                coco_output["annotations"].append(annotation_info)
    return (image_info, annotation_info)



if __name__ == "__main__":
    start_time = time.time()
    main()
    time_consumed = time.time() - start_time
    print(f"Finished in {time_consumed} seconds")
