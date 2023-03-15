 
from tqdm import tqdm
import torch 
import pandas as pd 
import os 
import sys 
from PIL import Image 
import numpy as np
from pathlib import Path
from joblib import Parallel, delayed
from multiprocessing import cpu_count
from torchvision import transforms 
sys.path.append(str(Path('..').absolute().parent))
from path_variables import *
from tqdm import tqdm 
from src_incremental.datasets.base_dataset import hpf

def read_rgb(img_path):
    transform = transforms.Compose([transforms.ToTensor()])
    return transform(Image.open(img_path))

def read_gray(img_path):
    transform = transforms.Compose([transforms.ToTensor()])
    return  transform(Image.open(img_path).convert('L'))

def read_hist(hist_path):
    return  np.load(hist_path)

def read_hpf(img_path):
    pil_img = Image.open(img_path).convert('L')
    x, _ = hpf(pil_img)
    x = np.expand_dims(x, axis=2)
    transform = transforms.Compose([transforms.ToTensor()])
    
    return transform(x) 


def read_patches(img_patch_path, type="rgb"):
 
    patch_list = os.listdir(img_patch_path)
    
    patches = []
    for p in patch_list:
        if type == "rgb":
            patch = read_rgb(os.path.join(img_patch_path, p))
        elif type == "hpf":
            patch = read_hpf(os.path.join(img_patch_path, p))    
        elif type == "gray":
            patch = read_gray(os.path.join(img_patch_path, p))    
             
        patches.append(patch)

    patches = torch.stack(patches, dim=0)
    # std = sqrt(E[X^2] - (E[X])^2)
    channels_sum = torch.sum(torch.mean(patches , dim=[2,3]), dim=0)
    channels_squared_sum = torch.sum(torch.mean(patches**2, dim=[2,3]), dim=0)
    num_patches = patches.shape[0]
        
    return channels_sum, channels_squared_sum, num_patches

       
        
def read_histogram_patches(img_hist_path):
    
    patch_list = os.listdir(img_hist_path)
    histogram_list = []
    
    for p in patch_list:
        hist_patch = read_hist(os.path.join(img_hist_path, p))
        histogram_list.append(hist_patch)
    
    histogram_list = np.stack(histogram_list, axis=0)
      
    return histogram_list

if __name__ == "__main__":
    
    
    histogram_path = os.path.join(FACIL_BOLOGNA_PATH, "Histograms")
    patches_path = os.path.join(FACIL_BOLOGNA_PATH, "Patches_256")
    info_path = os.path.join(FACIL_BOLOGNA_PATH, "Info")
    
    socials = ["Facebook", 
                "Google+" , 
                "GPhoto" ,
                "Instagram",
                "Pinterest", 
                "QQ", 
                "Telegram",
                "Tumblr",
                "Twitter",
                "Viber",
                "VK",
                "WeChat",
                "WhatsApp",
                "Wordpress"]
    
    train_images_file = open(os.path.join(info_path, "train.txt"),"r")
    lines = train_images_file.readlines()
    train_images = []
    for x in lines:
        train_images.append(x.split(' ')[0].replace(".jpg", ""))
        
        

    
    """
    Compute Max Min Histogram For Normalization
    """
    print("Computing Histogram Normalization Factor")
    histograms = Parallel(n_jobs=cpu_count())(delayed(read_histogram_patches) (os.path.join( histogram_path , img+"_crops"))
                                for img in tqdm(train_images)) 
    
    histograms = np.concatenate(histograms)
    histograms = histograms.reshape(histograms.shape[0], -1)
    max_hist = np.max(histograms, axis=0) 
    min_hist = np.min(histograms, axis=0)
    np.save(os.path.join(FACIL_BOLOGNA_PATH,"Info", "max_hist.npy"), max_hist)
    np.save(os.path.join(FACIL_BOLOGNA_PATH,"Info", "min_hist.npy"), min_hist)
    
    """
    Compute HPF Mean STD for standardization
    """
    print("Computing HPF Mean STD")
    result = Parallel(n_jobs=cpu_count())(delayed(read_patches) (os.path.join(patches_path, img+"_crops"), type="hpf")
                                for img in tqdm(train_images))  
    mean = torch.stack([r[0] for r in result], dim=0)
    squared_mean = torch.stack([r[1] for r in result], dim=0)
    n_patches = np.sum([r[2] for r in result])
    
 
    hpf_mean = torch.sum(mean,dim=0) / n_patches
    hpf_std = (torch.sum(squared_mean) / n_patches - hpf_mean ** 2) ** 0.5

    print("HPF Mean {}, HPF STD {}".format(hpf_mean, hpf_std))

    
    """
    Compute RGB Mean STD for standardization
    """
    print("Computing RGB Mean STD")
    result = Parallel(n_jobs=cpu_count())(delayed(read_patches) (os.path.join(patches_path, img+"_crops"))
                                for img in tqdm(train_images))  
    mean = torch.stack([r[0] for r in result], dim=0)
    squared_mean = torch.stack([r[1] for r in result], dim=0)
    n_patches = np.sum([r[2] for r in result])
    
 
    rgb_mean = torch.sum(mean,dim=0) / n_patches
    rgb_std = (torch.sum(squared_mean) / n_patches - rgb_mean ** 2) ** 0.5

    print("RGB Mean {}, RGB STD {}".format(rgb_mean, rgb_std))
    
        
    """
    Compute Gray Mean STD for standardization
    """
    print("Computing Gray Mean STD")
    result = Parallel(n_jobs=cpu_count())(delayed(read_patches) (os.path.join(patches_path, img+"_crops"),type="gray")
                                for img in tqdm(train_images))  
    mean = torch.stack([r[0] for r in result], dim=0)
    squared_mean = torch.stack([r[1] for r in result], dim=0)
    n_patches = np.sum([r[2] for r in result])
    
 
    gray_mean = torch.sum(mean,dim=0) / n_patches
    gray_std = (torch.sum(squared_mean) / n_patches - gray_mean ** 2) ** 0.5

    print("Gray Mean {}, Gray STD {}".format(gray_mean, gray_std))

    
    
    
    
    



