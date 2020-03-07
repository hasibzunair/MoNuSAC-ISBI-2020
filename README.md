### MoNuSAC-ISBI-2020

##### *PatchEUNet: Fully Convolutional Network for Instance Segmentation*

<p align="center">
<a href="#"><img src="media/resulta.png"></a>
<a href="#"><img src="media/resultb.png"></a>
</p>

Code for Multi-organ Nuclei Segmentation and Classification Challenge organised at ISBI 2020.

### Overview

In this work, we implement an end-to-end deep learning framework for automatic nuclei segmentation and classification from H&E stained whole slide images (WSI) of multiple organs (breast, kidney, lung and prostate). The proposed approach, called *PatchEUNet*, leverages a fully convolutional neural network of the U-Net family by replacing the encoder of the U-Net model with an EfficientNet architecture with weights initialized from ImageNet.

<p align="center">
<a href="#"><img src="media/network.png"></a>
</p>

Since there is a large scale variance in the whole slide images of the MoNuSAC 2020 Challenge, we propose to use a patchwise training scheme to mitigate the problems of multiple scales and limited training data. For the class imbalance problem, we design an objective function defined as a weighted sum of a focal loss and Jaccard distance, resulting in significantly improved performance. During inference, we apply the median filter on the predicted masks in an effort to refine the segmented outputs. Finally for each class mask, we apply watershed algorithm to get the class instances. Details about the problem can be found at [URL](https://monusac-2020.grand-challenge.org/).

### Requirements
* Python: 3.6
* Tensorflow: 2.0.0
* Keras: 2.3.1
* [segmentation_models](https://segmentation-models.readthedocs.io/en/latest/install.html).
* OpenCV

### Environment installations

Run this command to make environment

```
conda env create -f environment.yml
```

OR you can make a new environment by:

```
conda create -n yourenvname python=3.6 anaconda
```

Then install the packages

```
conda install -c anaconda tensorflow-gpu=2.0.0
conda install -c conda-forge keras
conda install -c conda-forge opencv
conda install -c conda-forge tqdm
```

The run `conda activate yourenvname`.

NOTE: `segmentation_models` does not have conda distribution. You can install by running `pip install -U --pre segmentation-models --user` inside your environment. More details at [segmentation_models](https://segmentation-models.readthedocs.io/en/latest/install.html).

### Dataset versions

* `MoNuSAC_images_and_annotations` : original dataset which has patient's whole slide images (WSI) and ground truth. (Given by challenge organizers)
* `MoNuSAC_masks` : contains binary masks generated from `get_mask.ipynb`.
* `Testing Images`: contains test images, without annotations. (Given by challenge organizers)
* `data_processedv0` : contains all raw images and the ground truth masks.
* `data_processedv4` : sliding window patchwise data from original images and masks in `data_processedv0`.
* `data_processedv5` : 80/20 trainval split from `data_processedv4`.

### Getting started

0. Clone repository
1. Make `dataset` folder
2. Put `MoNuSAC_images_and_annotations` in `dataset` folder
3. Run `0_get_masks.ipynb`. You should get the MoNuSAC_masks folder in dataset
4. Run `1_data_process_MoNuSAC.ipynb` to get raw images and their ground truth masks in `data_processedv0`. 
5. Run `2_extract_patches.ipynb` to get patches of images and gt masks from the previous raw version to get `data_processedv4` and the 80/20 split `data_processedv5`.
6. Run `3_train.iynb`. It trains *PatchEUNet* on `data_processedv5`.
7. Put `Testing Images` in `dataset` folder.
8. Run `3c_load_test_data.ipynb`. Outputs `test_images` folder with all test images.
8. Run `4_inference.ipynb` to get final prediction masks from `test_images`. (For visualization)
9. Run `4b_inference.ipynb` to get final prediction masks according to MoNuSAC submission format.

More details coming soon....
