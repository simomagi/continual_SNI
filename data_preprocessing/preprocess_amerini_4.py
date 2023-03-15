from PIL import Image
from preprocess_utils import noise_extract, get_histogram_dct, rgb2gray
import numpy as np 
import os 
from joblib import Parallel, delayed
from multiprocessing import cpu_count
from tqdm import tqdm 
import sys 
from pathlib import Path
sys.path.append(str(Path('..').absolute().parent))
from path_variables import *
import argparse
from copy import deepcopy
 
def preprocess_image(img_path, hist_path, res_path, patch_size=256):
    # compute quantization table 
    
    full_img_name = img_path.split("/")[-1].replace(".jpg","")
    hist_path = os.path.join(hist_path, full_img_name+"_crops")
    res_path = os.path.join(res_path, full_img_name+"_crops")
    
    if not os.path.exists(hist_path):
        os.mkdir(hist_path)
    
    if not os.path.exists(res_path):
        os.mkdir(res_path)
        
    patch_ids, hist_dct = get_histogram_dct(img_path,  patch_size)
    
    img = np.asarray(Image.open(img_path)).astype(np.uint8)
    # compute Noise Residual 
    W = noise_extract(img, levels=4, sigma=5)
    W = rgb2gray(W)
    W = (W - W.min()) /(W.max() - W.min())
    h, w = W.shape[0], W.shape[1]

    W = W[:int(patch_size * np.floor(h/patch_size)), :int(patch_size * np.floor(w/patch_size))]
    new_h, new_w = W.shape[0], W.shape[1]
    tiled_W = W.reshape(int(new_h // patch_size), patch_size,  int(new_w // patch_size),  patch_size)
    tiled_W = tiled_W.swapaxes(1, 2)
    tiled_W = tiled_W.reshape(-1, patch_size, patch_size)
    
    W_patch_ids = [i for i in range(0, tiled_W.shape[0])]
 
    assert W_patch_ids == patch_ids
    
    for i in patch_ids:
        name = img_path.split("/")[-1]
        residual_name =  name.replace(".jpg", "_{}_res.npy".format(i))
        hist_name = name.replace(".jpg", "_{}_hist.npy".format(i))
        current_hist = hist_dct[i]
        current_residual = tiled_W[i]
                
        np.save(os.path.join(hist_path, hist_name), current_hist)
        np.save(os.path.join(res_path, residual_name), current_residual)


def main(args):
    IMAGES_PATH =   os.path.join(FACIL_BOLOGNA_PATH, "Images")
    HIST_PATH = os.path.join(FACIL_BOLOGNA_PATH, "Histograms")
    RESIDUAL_PATH = os.path.join(FACIL_BOLOGNA_PATH, "PrnuResidual")

    if not os.path.exists(HIST_PATH):
        os.mkdir(HIST_PATH)

    if not os.path.exists(RESIDUAL_PATH):
        os.mkdir(RESIDUAL_PATH)

    images = os.listdir(IMAGES_PATH)

    images_path = [os.path.join(IMAGES_PATH, img) for img in images]

    Parallel(n_jobs=cpu_count())(delayed(preprocess_image)(img_path, HIST_PATH, RESIDUAL_PATH, args.patch_size) for img_path in tqdm(images_path))  



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--patch_size", type=int,default=256)
    
    args = parser.parse_args()
    main(args)
 
