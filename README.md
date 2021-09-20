# About This Project...
This project involves two resources: Auto Labeled Image Splicing (ALIS) dataset and Forgery Segmentation Network (F-SegNet).

## ALIS Dataset Overview
ALIS dataset is an image splicing verification dataset constructed based on ImageNet, COCO, and Crowd Human datasets. It comprises 224,388 spliced images with their corresponding ground truth masks. ALIS dataset contains spliced images under five categories: Person, People, Car, Cat, and Chair. However, any researcher can easily expand on these categories with the code provided.

## F-SegNet Overview
F-SegNet can be used to detect image splicing forgeries. It smartly converts an image verification problem into forgery instance segmentation problem. By combining Mask R-CNN, Error Level Analysis, Normal Map Analysis, and MoCo v2 (a self-supervised learning method), F-SegNet can effectively locate spliced instances in a forged image.
[alt text](https://github.com/Merxon22/F-SegNet/blob/main/sample_image/outcome%201.png)
[alt text](https://github.com/Merxon22/F-SegNet/blob/main/sample_image/outcome%202.png)
[alt text](https://github.com/Merxon22/F-SegNet/blob/main/sample_image/outcome%204.png)
## Important Resources
ALIS dataset download: [ALIS dataset](https://www.dropbox.com/sh/r94z9f7ov66gj3i/AACLXFgDuogrSK-jiMJPJ9YFa?dl=0)

F-SegNet pretrained weights (after 500,000 iterations of training): [F-SegNet final weights](https://www.dropbox.com/s/zk73svhemd8i7pa/model_final.pth?dl=0).

## Construction of ALIS dataset
