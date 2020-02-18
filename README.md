### MoNuSAC-ISBI-2020


Challenge code for Multi-organ Nuclei Segmentation and Classification Challenge 2020.

### Environment installations

Run this command to make environment

```conda env create -f environment.yml
```

### Dataset

* `MoNuSAC_images_and_annotations` : contains original dataset
* `MoNuSAC_masks` : contains binary masks generated from `get_mask.ipynb`.
* `data_processedv0` : contains all raw images and the ground truth masks in two folders.
* `data_processedv1` : (NOT USING NOW) patchwise data from `data_processedv0`.
* `data_processedv2` : trainval split from `data_processedv0`.
* `data_processedv3` : (NOT USING NOW) trainval split from `data_processedv1`.
* `data_processedv4` : (TRAIN FINAL MODEL ON WHOLE PATCH DATA) sliding window patchwise data from original images and masks in `data_processedv0`.
* `data_processedv5` : trainval split from `data_processedv4`.
* `data_processedvpl` : patchwise and trainval split from `data_processedv2`.


### Getting started

0. Clone repository
1. Make `dataset` folder
2. Put `MoNuSAC_images_and_annotations` in `dataset` folder
3. Run `0_get_masks.ipynb`. You should get the MoNuSAC_masks folder in dataset
4. Run `1_data_process_MoNuSAC.ipynb` to get raw images and their ground truth masks. 
5. Run `2b_extract_patches.ipynb` to get patches of images and gt masks from the previous raw version.
6. Run `5_train.iynb`. It trains on `data_processedv5`.
7. Put `Testing Images` in `dataset` folder.
8. Run `6b_inference.ipynb` to get final prediction masks.

More documentation coming soon.