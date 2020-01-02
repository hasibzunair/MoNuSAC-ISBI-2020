### MoNuSAC-ISBI-2020
ISBI 2020 Challenge

### Environment installations
```
conda env export > <environment-name>.yml    
conda env create -f <environment-name>.yml
```

### Dataset

`MoNuSAC_images_and_annotations` : contains original dataset
`MoNuSAC_masks` : contains binary masks generated from get_mask.ipynb
`data_processedv0` : contains all raw images and the ground truth masks in two folders

### Run this

1. Make `dataset` folder
2. Put `MoNuSAC_images_and_annotations` in `dataset` folder
3. Run `get_masks.ipynb`. You should get the MoNuSAC_masks folder in dataset
4. Run `data_process_MoNuSAC.ipynb`. 
5. ....


More info coming soon.