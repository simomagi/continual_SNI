 
from __future__ import annotations
import numpy as np 
from copy import deepcopy
import os 
import sys 
from PIL import Image
import argparse
from tqdm import tqdm 
from joblib import Parallel, delayed
import multiprocessing 
import pickle
from os.path import join 
import sys 
from pathlib import Path
sys.path.append(str(Path('..').absolute().parent))
from path_variables import *



def crop(sample, patch_size):

 
    sample_h, sample_w, sample_ch = sample.shape[0], sample.shape[1], sample.shape[2]
  
    # clip row out of the patch size 
    sample = sample[:int(patch_size * np.floor(sample_h/patch_size)), :int(patch_size * np.floor(sample_w/patch_size)), :]

    tiled_sample =  sample.reshape(int(sample_h // patch_size), patch_size,  int(sample_w // patch_size),  patch_size, sample_ch)
    tiled_sample = tiled_sample.swapaxes(1, 2)
    tiled_sample = tiled_sample.reshape(-1, patch_size, patch_size,  sample_ch)
    
 
    return [tiled_sample] 

def crop_image(src_path, this_img,  patch_size, dst_folder, this_class, this_id):
   
  
    img = np.asarray(Image.open(os.path.join(src_path, this_img)).convert('RGB'))
    full_img_name = deepcopy(this_img).replace(".jpg", "")
    crops_path = os.path.join(dst_folder,full_img_name+"_crops")
    if not os.path.exists(crops_path):
        os.mkdir(crops_path)
        
    tiles = crop(img, patch_size=patch_size)
    count = 0
    crop_names = []
    classes = []
    ids = []
    for tile in tiles:
        for  img in tile:
            pil_image = Image.fromarray(img)

            crop_img_name = "{}_{}.png".format(full_img_name, count)
            pil_image.save(os.path.join(dst_folder, crops_path, crop_img_name))
            crop_names.append(crop_img_name)
            classes.append(this_class)
            ids.append(this_id)
            count += 1


    return this_img, crop_names, classes, ids, count
    

 
def main(args):
    INFO_PATH = os.path.join(FACIL_BOLOGNA_PATH,"Info")
    IMAGES_PATH = os.path.join(FACIL_BOLOGNA_PATH, "Images")
    dst_folder = os.path.join(FACIL_BOLOGNA_PATH, "Patches_{}".format(args.patch_size))
    
    
    if not os.path.exists(dst_folder):
        os.mkdir(dst_folder)

    trn_lines = np.loadtxt(os.path.join(INFO_PATH,'train.txt'), dtype=str)
    val_lines =  np.loadtxt(os.path.join(INFO_PATH,'val.txt'), dtype=str)
    tst_lines = np.loadtxt(os.path.join(INFO_PATH, 'test.txt'), dtype=str)
    
    datasets = {"train": trn_lines, "val":val_lines, "test":tst_lines}
    
    for _set, lines in datasets.items():
        result = Parallel(n_jobs=multiprocessing.cpu_count())(delayed(crop_image)(IMAGES_PATH, str(this_img),  args.patch_size,  dst_folder, str(this_class), str(this_id))   for this_img, this_class, this_id  in tqdm(lines))
        
        image_names = [img_crop for res in result for img_crop in res[1]] #iterate over lines
        image_classes = [crop_class for res in result for crop_class in res[2]]
        image_ids = [crop_id for res in result for crop_id in res[3]] 
        
        original_images = [res[0] for res in result]
        count_images = [int(res[4]) for res in result]
        count_dict = {k:v for k, v in zip(original_images, count_images)}
        

        with open(join(INFO_PATH, '{}_crops.txt'.format(_set)),'w') as  annotations_file:
            for img_name, img_c, img_id in zip(image_names, image_classes, image_ids):
                annotations_file.write(' '.join([img_name, img_c, img_id])+'\n')
        
        with open(join(INFO_PATH, '{}_crop_count.pickle'.format(_set)), 'wb') as handle:
            pickle.dump(count_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--patch_size", type=int,default=256)
    
    args = parser.parse_args()
    main(args)
 
