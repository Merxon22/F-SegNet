![outcome](https://github.com/Merxon22/F-SegNet/blob/main/sample_image/outcome%201.png)
# About This Project...
This project involves two resources: Auto Labeled Image Splicing (ALIS) dataset and Forgery Segmentation Network (F-SegNet).

## ALIS Dataset Overview
ALIS dataset is an image splicing verification dataset constructed based on ImageNet, COCO, and Crowd Human datasets. It comprises 224,388 spliced images with their corresponding ground truth masks.

![alis](https://github.com/Merxon22/F-SegNet/blob/main/sample_image/ALIS_database.png)

## F-SegNet Overview
F-SegNet can be used to detect image splicing forgeries. It smartly converts an image verification problem into forgery instance segmentation problem. By combining Mask R-CNN, Error Level Analysis, Normal Map Analysis, and MoCo v2 (a self-supervised learning method), F-SegNet can effectively locate spliced instances in a forged image.


![architecture](https://github.com/Merxon22/F-SegNet/blob/main/sample_image/architecture.png)
## Important Resources
ALIS dataset download: [ALIS dataset](https://www.dropbox.com/sh/r94z9f7ov66gj3i/AACLXFgDuogrSK-jiMJPJ9YFa?dl=0)

F-SegNet pretrained weights (after 500,000 iterations of training): [F-SegNet final weights](https://www.dropbox.com/s/zk73svhemd8i7pa/model_final.pth?dl=0).

## Step 1
Install requirements:

`pip install requirements.txt`



## Step 2: Train F-SegNet
To train F-SegNet, run the following code:

`python train.py -d <string> -an <string> -i <int> -c <bool> -w <string> -o <string>`

where: 

`-d` is **path to image directory**

`-an` is **path to COCO annotation file**

`-i` is **training iteration count**

`-c` is **continue training**, which is set to `True` by default

`-w` is **path to pretrained weights**

`-o` is **output directory path**

***Note:***
If **"continue training"** is set to **True**, `-w` will be discarded and F-SegNet will continue training based on the last checkpoin in `-o` **output directory**.

## Step 3: Check Training Log
To check training log, use tensorboard with the following code:

`tensorboard --logdir <str>`

where `--logdir` is the **output directory path**. It should be same with `-o` mentioned in Step 2.

## Step 4: Evaluate F-SegNet
To evalualte F-SegNet, use the following code:

`python evaluate.py -d <string> -an <string> -w <string> -t <float>`

where:

`-d` is **path to image directory**

`-an` is **path to COCO annotation file**

`-w` is **output directory path**, which should be the same as `-o` mentioned in Step 2. F-SegNet will automatically read the `./model_final.pth` in this directory.

`-t` is the **score threshold**, which is set to `0.5` by default

## Acknowledgement
This project is developed based on [Detectron 2](https://github.com/facebookresearch/detectron2) framework.
