### MoNuSAC-ISBI-2020

Code for Multi-organ Nuclei Segmentation and Classification Challenge organised at ISBI 2020.

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

Or you can make a new environment by:

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

NOTE: `segmentation_models` does not have conda distribution. You can install by running `pip install -U --pre segmentation-models --user` inside your environment.

### Dataset versions

* `MoNuSAC_images_and_annotations` : contains original dataset
* `MoNuSAC_masks` : contains binary masks generated from `get_mask.ipynb`.
* `Testing Images`: contains test images, without annotations
* `data_processedv0` : contains all raw images and the ground truth masks in two folders.
* `data_processedv4` : (TRAIN FINAL MODEL ON WHOLE PATCH DATA) sliding window patchwise data from original images and masks in `data_processedv0`.
* `data_processedv5` : trainval split from `data_processedv4`.
* `data_processedvpl` : patchwise and trainval split from `data_processedv2`.


### Getting started

0. Clone repository
1. Make `dataset` folder
2. Put `MoNuSAC_images_and_annotations` in `dataset` folder
3. Run `0_get_masks.ipynb`. You should get the MoNuSAC_masks folder in dataset
4. Run `1_data_process_MoNuSAC.ipynb` to get raw images and their ground truth masks in `data_processedv0`. 
5. Run `2b_extract_patches.ipynb` to get patches of images and gt masks from the previous raw version to get `data_processedv4` and the 70/30 split `data_processedv5`.
6. Run `5_train.iynb`. It trains on `data_processedv5`.
7. Put `Testing Images` in `dataset` folder.
8. Run `6b_inference.ipynb` to get final prediction masks.

More documentation coming soon....
