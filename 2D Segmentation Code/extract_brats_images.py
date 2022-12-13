import h5py
import imageio
from PIL import Image
import numpy as np
import matplotlib.image as mpimg
from tqdm import tqdm
import os
import shutil


def make_folder(target_folder):
    if not (os.path.isdir(target_folder)):
        print(f'Creating {target_folder} folder')
        os.mkdir(target_folder)


def get_image_data(filename, path):
    path = os.path.join(path, filename)
    file = h5py.File(path, 'r')
    data = dict()
    data['image'] = np.array(file.get('image')) 
    data['mask'] = np.array(file.get('mask')) 
    return data

def save_image_data(filename, path, tmp_folder, data):
    path_image = os.path.join(path, filename+'.png')
    path_mask = os.path.join(path, filename+'_mask.png')
    
    tmp = os.path.join(tmp_folder, "tmp.png")
    imageio.imwrite(tmp, data['image'])
    img = Image.open(tmp).convert('L')
    img.save(path_image)

    imageio.imwrite(tmp, data['mask'])
    img = Image.open(tmp).convert('L')
    img.save(path_mask)

def clean(path):
    shutil.rmtree(path)

def delete_blank_images(path, filenames):
    count = 0
    for file in filenames:
        if(".png" in file):
            fullpath = os.path.join(path, file)
            with Image.open(fullpath) as img:
                # check if image is blank
                if not img.getbbox():
                    os.remove(fullpath)
                    count += 1
        else:
            print(f'{file} is not a png...')
    
    filenames = os.listdir(path)
    print(f'There were {count} blank images found...')
    print(f'There are now {len(filenames)} files in the directory...')

def delete_nonpairs(path):
    filenames = os.listdir(path)
    non_pairs = []
    for file in filenames:
        """ Check that all the MASK's have a corresponding IMAGE """
        if "mask" in file: 
            filename = file.replace("_mask.png", ".png")
            if filename not in filenames:
                non_pairs.append(file)
        """ Check that all the IMAGES have a corresponding MASK """
        if "mask" not in file:  
            filename = file.replace(".png", "_mask.png")
            if filename not in filenames:
                non_pairs.append(file)
    print(f'{len(non_pairs)} files dont have a corresponding image...')

    # delete non-pairs from directory from directory
    count = 0
    for file in non_pairs:
        fullpath = os.path.join(path, file)
        os.remove(fullpath)
        count += 1

    print(f'{count} files have been deleted from png_dataset...')
        
def reorder_dataset(path):
    filenames = os.listdir(path)
    sorted_filenames = []
    num_images = int((len(filenames) - 1) / 2)
    print(f'images/mask names being reorder: {num_images}' )
    for x in range(num_images):
        sorted_filenames.append(str(x)+".png")

    """ Reordering """
    count = 0
    for file in filenames:
        if count == num_images:
            break
        if ".png" in file and "mask" not in file:
            # Get new and old file names
            oldpath_image = os.path.join(path, file) 
            oldpath_mask = os.path.join(path, file.replace(".png", "_mask.png"))
            make_folder("png")
            newpath_image = os.path.join("png", sorted_filenames[count])
            newpath_mask = os.path.join("png", sorted_filenames[count].replace(".png", "_mask.png"))
            # rename images and corresponding mask files
            os.rename(oldpath_image, newpath_image)
            os.rename(oldpath_mask, newpath_mask)
            print(f'old path: {oldpath_image} --> new path: {newpath_image}')
            count += 1

def filter_brats(path):
    # Get all filenames
    filenames = os.listdir(path)
    # delete blank images
    print("Deleting blank images...")
    delete_blank_images(path, filenames)
    # delete images/mask that do not have corresponding mask/image
    print("\nDeleting non-pairs...")
    delete_nonpairs(path)
    # reorder dataset so that names follow consecutive sequence (i.e. 0.png, 0.mask, 1.png, 1.mask)
    print("Reordering png_dataset...")
    reorder_dataset(path)

    # new dataset located in png
    print("Complete!...")
    new_path = "/Users/jiezy/Desktop/Rice Fall 2022/COMP 576 Deep Learning/ju6-project/GliomaSegmentation/my_bts/png"
    print(f'New dataset located in ./png and contains {len(os.listdir(new_path))} total images...')



def main():
    """ Hyperparameters """
    EXTRACT_DATA = False
    FILTER_DATA = True
    PATH = "/Users/jiezy/Desktop/Rice Fall 2022/COMP 576 Deep Learning/ju6-project/GliomaSegmentation/my_bts/png_dataset"
    """ Hyperparameters """

    if EXTRACT_DATA:
        data_read_path = "/Users/jiezy/Desktop/Rice Fall 2022/data"
        folder = os.listdir(data_read_path)
        total_images = len(folder)
        data_save_path = "png_dataset"
        tmp_save_path = "tmp"

        print("Creating necessary folders...")
        make_folder(data_save_path)
        make_folder(tmp_save_path)
        print(f'Starting to read images from {data_read_path}...')
        
        files = os.listdir(data_read_path)
        for count in tqdm(range(1, total_images+1)):
            if(".h5" in files[count]):
                save_filename = str(count)
                data = get_image_data(files[count], data_read_path)
                save_image_data(str(int(save_filename)-1), data_save_path, tmp_save_path, data)

        """clean up when done"""
        clean(tmp_save_path)
    if FILTER_DATA:
        filter_brats(PATH)
   

if __name__ == "__main__":
    main()