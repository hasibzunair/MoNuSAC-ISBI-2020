"""
Prediction code for PatchEUnet by the_great_backpropagator


Save final predictions in required format

README.txt for installation instructions.

Major requirements:
Python: 3.6
Tensorflow: 2.0.0
Keras: 2.3.1
segmentation_models
OpenCV

"""

# Import libs
import os
import time
import cv2
from tqdm import tqdm
import numpy as np
import skimage.draw
import random
import keras
import cv2
from glob import glob
import warnings
import random
import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib.colors
from skimage.transform import resize
import efficientnet.tfkeras
from tensorflow.keras.models import load_model


print("All libraries read correctly!!!!")


# Helper for dirtectory creation
def create_directory(directory):
    '''
    Creates a new folder in the specified directory if the folder doesn't exist.
    INPUT
        directory: Folder to be created, called as "folder/".
    OUTPUT
        New folder in the current directory.
    '''
    if not os.path.exists(directory):
        os.makedirs(directory)

        
# Name experiment
experiment_name = "unet-dtvpl96-exp-7"
        
    
# Root directory of the project
ROOT_DIR = os.path.abspath(".")

# Model path
log_path = os.path.join(ROOT_DIR, "logs", experiment_name)

# Test file directory
IMAGES_FOLDER = os.path.join(ROOT_DIR, "dataset", "Testing images/")

# Target destination for storing the prediction masks
PRED_DEST =  os.path.join(ROOT_DIR, "dataset", "the_great_backpropagator_MoNuSAC_test_results")

# Create folder prediction parent folder
create_directory(PRED_DEST)


# Load the model
model = None
model = load_model('{}/{}.h5'.format(log_path, experiment_name), compile=False)
#model.summary()



# Helpers
def pad(img, pad_size=96):
    """
    Load image from a given path and pad it on the sides, so that eash side is divisible by 96 (network requirement)
    if pad = True:
        returns image as numpy.array, tuple with padding in pixels as(x_min_pad, y_min_pad, x_max_pad, y_max_pad)
    else:
        returns image as numpy.array
    """

    if pad_size == 0:
        return img

    height, width = img.shape[:2]

    if height % pad_size == 0:
        y_min_pad = 0
        y_max_pad = 0
    else:
        y_pad = pad_size - height % pad_size
        y_min_pad = int(y_pad / 2)
        y_max_pad = y_pad - y_min_pad

    if width % pad_size == 0:
        x_min_pad = 0
        x_max_pad = 0
    else:
        x_pad = pad_size - width % pad_size
        x_min_pad = int(x_pad / 2)
        x_max_pad = x_pad - x_min_pad

    img = cv2.copyMakeBorder(img, y_min_pad, y_max_pad, x_min_pad, x_max_pad, cv2.BORDER_REFLECT_101)

    return img, (x_min_pad, y_min_pad, x_max_pad, y_max_pad)



def unpad(img, pads):
    """
    img: numpy array of the shape (height, width)
    pads: (x_min_pad, y_min_pad, x_max_pad, y_max_pad)
    @return padded image
    """
    (x_min_pad, y_min_pad, x_max_pad, y_max_pad) = pads
    height, width = img.shape[:2]

    return img[y_min_pad:height - y_max_pad, x_min_pad:width - x_max_pad]



def read_nuclei(path):
    "read raw data"

    # Load 4-channel image
    img = skimage.io.imread(path)
    
    # input image
    if len(img.shape) > 2:
        img = img[:,:,:3]
    # mask
    else:
        # do nothing
        pass
        
    return img



def save_nuclei(path, img):
    "save image"
    skimage.io.imsave(path, img)

    
def sliding_window(image, step, window):
    x_loc = []
    y_loc = []
    cells = []
    
    for y in range(0, image.shape[0], step):
        for x in range(0, image.shape[1], step):
            cells.append(image[y:y + window[1], x:x + window[0]])
            x_loc.append(x)
            y_loc.append(y)
    return x_loc, y_loc, cells


def extract_patches(image, step, patch_size):
    
    patches = []
    
    # Get locations
    x_pos, y_pos, cells = sliding_window(image, step, (patch_size[0], patch_size[1]))

    for (x, y, cell) in zip(x_pos, y_pos, cells):

        # Get patch
        patch = image[y:y + patch_size[0], x:x + patch_size[0]]

        # Get size
        raw_dim = (patch.shape[1], patch.shape[0]) # W, H
        #print(raw_dim)
        #print(patch.shape)


        if raw_dim != (patch_size[0], patch_size[1]):

            # Resize to 64x64
            #patch = cv2.resize(patch, (64, 64), interpolation = cv2.INTER_AREA)
            patch, pad_locs = pad(patch, pad_size=patch_size[0])
            
            
            # Do stuffffff
            patches.append(patch)
        
        else:

            # Do stuffffff
            patches.append(patch)
    
    patches = np.array(patches)
    
    return patches
    
    
    

# Helper function for data visualization
import numpy as np
from skimage.transform import resize

# Helper function for data visualization
def visualize(**images):
    """Plot images in one row."""
    
    norm=plt.Normalize(0,4) # 5 classes including BG
    map_name = matplotlib.colors.LinearSegmentedColormap.from_list("", ["black", "red","yellow","blue", "green"])

    
    n = len(images)
    plt.figure(figsize=(18, 16))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image, cmap=map_name, norm=norm)
    plt.show()
    
    
    
def prep(img):
    img = img.astype('float32')
    img = (img > 0.5).astype(np.uint8)  # threshold
    img = resize(img, (image_cols, image_rows), preserve_range=True)
    return img




def visualize_results(image, mask):
    
    f, axarr = plt.subplots(1,2, figsize=(16, 16))
    
    norm=plt.Normalize(0,4) # 5 classes including BG
    map_name = matplotlib.colors.LinearSegmentedColormap.from_list("", ["black", "red","yellow","blue", "green"])

    axarr[0].imshow(image)
    axarr[1].imshow(mask, cmap=map_name, norm=norm)


    
def vis_gray(image, mask):
    
    f, axarr = plt.subplots(1,2, figsize=(16, 16))
    
    axarr[0].imshow(image)
    axarr[1].imshow(mask, cmap='gray')



def predict(im):
    """Predict on patch"""
    
    im = np.expand_dims(im, axis=0)
    
    im = model.predict(im)
    im = np.argmax(im.squeeze(), axis=-1)
 
    #assert im.shape == (96, 96), "Wrong shape, {}!".format(im.shape)
    
    return im



def whole_slide_predict(whole_image):
    
    #import pdb; pdb.set_trace()
    
    # If input image less than patch, infer on whole image
    if whole_image.shape[0] < 96 or whole_image.shape[1] < 96:
        
        # Get size
        raw_dim = (whole_image.shape[1], whole_image.shape[0]) # W, H
        
        # Resize to 64x64 for prediction
        #whole_image_rs = cv2.resize(whole_image, (64, 64), interpolation = cv2.INTER_AREA)
        whole_image_rs, pad_locs = pad(whole_image, pad_size=96)
        
        
        # Infer
        pred = predict(whole_image_rs)
        
        
        # Resize back to original shape
        #pred = cv2.resize(pred, raw_dim, interpolation = cv2.INTER_AREA)
        pred = unpad(pred, pad_locs)
        
        # Change dtype for resizing back to original shape
        pred = pred.astype(np.uint8)
        
      
    else:
        
        # Get patch locations
        x_pos, y_pos, cells = sliding_window(whole_image, 96, (96, 96)) 

        # Array for storing predictions
        pred = np.zeros((whole_image.shape[0], whole_image.shape[1])).astype(np.uint8)

        # Slide over each patch
        for (x, y, cell) in zip(x_pos, y_pos, cells):

            # Get patch
            patch = whole_image[y:y + 96, x:x + 96]

            # Get size
            raw_dim = (patch.shape[1], patch.shape[0]) # W, H

            # If less than patch size, resize and then run prediction
            if raw_dim != (96, 96):


                # Resize to 64x64
                #patch_rs = cv2.resize(patch, (64, 64), interpolation = cv2.INTER_AREA)
                patch_rs, pad_locs = pad(patch, pad_size=96)
                
                #print(patch.dtype, processed.dtype)
                
                assert patch.dtype == patch_rs.dtype, "Wrong data type after resizing!"

                
                # Infer
                processed = predict(patch_rs)
                
                # Resize back to original shape
                #processed = cv2.resize(processed, raw_dim, interpolation = cv2.INTER_AREA)
                processed = unpad(processed, pad_locs)
                
                # Change dtype 
                processed = processed.astype(np.uint8)
                
                assert patch.shape[:2] == processed.shape, "Wrong shape!"
                assert patch.dtype == processed.dtype, "Wrong data type in prediction!"

            else:

                
                # Infer
                processed = predict(patch)
                
                # Change dtype
                processed = processed.astype(np.uint8)

                #print(patch.dtype, processed.dtype)
                

                assert patch.shape[:2] == processed.shape, "Wrong shape!"
                assert patch.dtype == processed.dtype, "Wrong data type in prediction!"


            # Add in image variable
            pred[y:y + 96, x:x + 96] = processed 
            processed = None

    return pred




# Save predictions
# label_map = {'Epithelial':1, 'Lymphocyte':2, 'Macrophage':4, 'Neutrophil':3, }

# Read patient folders
IMAGES_SUB_FOLDER = [os.path.join(IMAGES_FOLDER, i) for i in sorted(next(os.walk(IMAGES_FOLDER))[1])]


# Test image path
image_paths = []

# Iterate over patient folders
raw_ct = 0
for ct in tqdm(range(len(IMAGES_SUB_FOLDER[:]))):
    
    #print(ct)
    
    # Read all raw images in patient sub folder
    all_imgs = sorted(glob(IMAGES_SUB_FOLDER[ct] + '/*.tif'))
    
    # Get patient ID
    #pn = IMAGES_SUB_FOLDER[ct].split('.')[0][-23:]
    pn = IMAGES_SUB_FOLDER[ct].split('/')[-1]
    
    print("Patient ID ----------------------> : ", pn)
    
    # Create patient folder with ID
    pn_folder = os.path.join(PRED_DEST, pn)
    create_directory(pn_folder)
    
    # Get all images in the patient folder
    paths = [s.split('/')[-1][:-4] for s in all_imgs]
    
    print(paths)
    
    
    
    # Iterate over the images of a patient
    for i in range(len(all_imgs)):
        
        # Read patient image
        image_paths.append(all_imgs[i])
        img = read_nuclei(all_imgs[i])
        
        print("Patient image: ", paths[i])
        
        # Create sub folder -> each sub image for a patient
        patient_sub_folder = os.path.join(pn_folder, paths[i])
        create_directory(patient_sub_folder)
        
        # For each sub folder(sub image) make sub-sub folder for each cell type
        epi_path = os.path.join(patient_sub_folder, "Epithelial")
        lym_path = os.path.join(patient_sub_folder, "Lymphocyte")
        neu_path = os.path.join(patient_sub_folder, "Neutrophil")
        macro_path = os.path.join(patient_sub_folder, "Macrophage")
        
        # Creat sub-sub folders for each sub image
        create_directory(epi_path)
        create_directory(lym_path)
        create_directory(neu_path)
        create_directory(macro_path)
        
        
        # Predict whole slide sub image of a patient
        pred = whole_slide_predict(img)
        
        # Post processing to refine predictions
        pred_filt = cv2.medianBlur(pred.astype(np.uint8), 5)

        #print(img.shape, pred.shape)
        #print("Uniques predicted", np.unique(pred_filt))

        # Make dummy mask
        zero_mask = np.zeros((pred_filt.shape[0], pred_filt.shape[1])).astype(np.uint8)

        # Overlay target class outputs
        epi_mask = np.where(pred_filt != 1, zero_mask, 1)
        lym_mask = np.where(pred_filt != 2, zero_mask, 2)
        neu_mask = np.where(pred_filt != 3, zero_mask, 3)
        macro_mask = np.where(pred_filt != 4, zero_mask, 4)
        
        # Save masks
        # Check if last number of uniques is not zero, if it is not then save this mask.
        # If it zero, it means the mask is empty, so skip this
        if np.unique(epi_mask)[-1] != 0:
                #np.save("{}/{}.npy".format(epi_path, raw_ct), epi_mask)
                sio.savemat("{}/{}.mat".format(epi_path, raw_ct), {'epi_mask':epi_mask})
        
        raw_ct+=1

        if np.unique(lym_mask)[-1] != 0:
                #np.save("{}/{}.npy".format(lym_path, raw_ct), lym_mask)
                sio.savemat("{}/{}.mat".format(lym_path, raw_ct), {'lym_mask':lym_mask})
                
        raw_ct+=1
        
        if np.unique(neu_mask)[-1] != 0:
                #np.save("{}/{}.npy".format(neu_path, raw_ct), neu_mask)
                sio.savemat("{}/{}.mat".format(neu_path, raw_ct), {'neu_mask':neu_mask})
                
        raw_ct+=1
        
        if np.unique(macro_mask)[-1] != 0:
                #np.save("{}/{}.npy".format(macro_path, raw_ct), macro_mask)
                sio.savemat("{}/{}.mat".format(macro_path, raw_ct), {'macro_mask':macro_mask})
                
        raw_ct+=1
    

print("Done!")
