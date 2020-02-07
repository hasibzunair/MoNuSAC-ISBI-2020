### MoNuSAC-ISBI-2020
ISBI 2020 Challenge

### Environment installations
```
conda env export > <environment-name>.yml    
conda env create -f <environment-name>.yml
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


### Run this

0. Clone repository
1. Make `dataset` folder
2. Put `MoNuSAC_images_and_annotations` in `dataset` folder
3. Run `get_masks.ipynb`. You should get the MoNuSAC_masks folder in dataset
4. Run `data_process_MoNuSAC.ipynb` to get raw images and their ground truth masks. 
5. Run `extract_patches.ipynb` to get patches of images and gt masks from the previous raw version.
6. Run `trainval_split.iynb` to split dataset.
7. ...

More info coming soon.
