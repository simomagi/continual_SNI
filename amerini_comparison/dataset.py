from torch.utils.data import Dataset
import os 
from os.path import join 
import pickle 
import random 
import numpy as np 
import torch 
import os
import random
import numpy as np
from torch.utils.data import Dataset
import scipy.fftpack as fp
import torch 
import pickle
from os.path import join
import cv2 
from scipy.fft import  dctn, idctn
from copy import deepcopy
from torchvision import transforms
from PIL import Image


def hpf(im,  r=50):
 
    im = np.asarray(im).astype(np.int32) - 128 
 
    T = dctn((im).astype(float))
 
    T_mask = deepcopy(T)

    # mask the highest frequencies according radius r and mask type
    mask = np.zeros_like(im).astype(np.uint8)

    points = np.array([[0,0],[r,0], [0,r]], np.int32)
    cv2.fillPoly(mask, [points], 255) #, thickness=1 True,

    mask = 255 - mask # black shape (0), white background (255) 

    # apply the masking
    T_mask = np.multiply(T_mask, mask/255)

    # apply inverse transform to come back in the image space
    im_filtered = (idctn(T_mask) + 128).astype(np.uint8) 
 
 
    im_filtered = im_filtered.clip(0,255)
 
    return im_filtered, T

def get_data(path):
    print("Hello")
    
    # read filenames and labels
    trn_lines = np.loadtxt(os.path.join(path,"Info",'train.txt'), dtype=str)

    tst_lines = np.loadtxt(os.path.join(path,"Info", 'test_crops.txt'), dtype=str)

    # dataset for exemplar and ewc based approach
    trn_lines_crops = np.loadtxt(os.path.join(path,"Info","train_crops.txt"),dtype=str)

    #ADD read validation lines  
    val_lines = np.loadtxt(os.path.join(path, "Info", 'val_crops.txt'), dtype=str)
    
    data = {}
    data ['name'] = 'task-0'  
    data ['trn'] = {'x': [], 'y': [], 'id':[]}
    data ['val'] = {'x': [], 'y': [], 'id':[]}
    data ['tst'] = {'x': [], 'y': [], 'id':[]}
    data['trn_crops'] = {'x': [], 'y': [], 'id':[]}

    # ALL OR TRAIN
    for this_image, this_label, this_id in trn_lines:
        data ['trn']['x'].append(str(this_image))
        data['trn']['y'].append(this_label)
        data['trn']['id'].append(this_id)
    
    # ALL OR TEST
    for this_image, this_label, this_id in tst_lines:
        data ['tst']['x'].append(str(this_image))
        data ['tst']['y'].append(this_label)
        data ['tst']['id'].append(this_id)
    
    # ALL OR VAL  (ADDED)

    for this_image, this_label, this_id in val_lines:
        data ['val']['x'].append(str(this_image))
        data ['val']['y'].append(this_label)
        data ['val']['id'].append(this_id)
    
    # Train Crops
    for this_image, this_label, this_id in trn_lines_crops:
        data['trn_crops']['x'].append(str(this_image))
        data['trn_crops']['y'].append(this_label)
        data['trn_crops']['id'].append(this_id)
    
    return data


class AmeriniDataset(Dataset):
    def  __init__(self, data, dataset_path, train) -> None:
        self.labels = data['y']
        self.images = data['x']
        self.ids = data['id']
        self.train = train
        self.info_path = os.path.join(dataset_path,"Info")
        self.hist_path = os.path.join(dataset_path, "Histograms")
        self.residual_path =  os.path.join(dataset_path, "PrnuResidual")

        
        if self.train:
            with open(join(self.info_path, 'train_crop_count.pickle'), 'rb') as handle:
               self.count_crops = pickle.load(handle)
        else:
            self.count_crops = None 
            
    def  __len__(self):
        return len(self.images)

    def __getitem__(self, index)  :
        if self.train:
            current_img_name = self.images[index]
            random_crop_id = random.randint(0, self.count_crops[current_img_name]-1)
            current_hist_name = current_img_name.replace(".jpg","_{}_hist.npy".format(random_crop_id))
            current_res_name = current_img_name.replace(".jpg", "_{}_res.npy".format(random_crop_id))
            img_name = current_img_name.replace(".jpg","")
            
            
        else:
            current_img_name = self.images[index]
            img_name = current_img_name.replace(".png","")
            img_name = img_name.split("_")[:-1]
            img_name = "_".join(img_name)

            current_hist_name = current_img_name.replace(".png","_hist.npy")
            current_res_name = current_img_name.replace(".png", "_res.npy")

        id = int(self.ids[index])
        label = int(self.labels[index])  
        
      
        residual = np.load(os.path.join(self.residual_path, img_name+"_crops", current_res_name))
        residual = torch.tensor(residual).unsqueeze(0)
        histogram =  np.load(os.path.join(self.hist_path, img_name+"_crops", current_hist_name))
   
        
        histogram = torch.tensor(histogram, dtype=torch.float).flatten().unsqueeze(0)
        
        return residual, histogram, label, id 
    


class CustomDataset(Dataset):
    def  __init__(self, data, dataset_path, train) -> None:
        self.labels = data['y']
        self.images = data['x']
        self.ids = data['id']
        self.train = train
        self.info_path = os.path.join(dataset_path,"Info")
        
        self.hist_path = os.path.join(dataset_path, "Histograms")
        self.patch_path =  os.path.join(dataset_path, "Patches_256")

        self.max_hist = torch.tensor(np.load(os.path.join(self.info_path,"max_hist.npy")))
        self.min_hist = torch.tensor(np.load(os.path.join(self.info_path,"min_hist.npy")))
        
        self.transform = transforms.Compose([transforms.ToTensor(), 
                                                transforms.Normalize(mean=0.4999687373638153, 
                                                                    std=0.0446637235581874)])
                
        if self.train:
            with open(join(self.info_path, 'train_crop_count.pickle'), 'rb') as handle:
               self.count_crops = pickle.load(handle)
        else:
            self.count_crops = None 
            
    def  __len__(self):
        return len(self.images)

    def __getitem__(self, index)  :
        if self.train:
            current_img_name = self.images[index]
            random_crop_id = random.randint(0, self.count_crops[current_img_name]-1)
            current_hist_name = current_img_name.replace(".jpg","_{}_hist.npy".format(random_crop_id))
            current_patch_name = current_img_name.replace(".jpg", "_{}.png".format(random_crop_id))
            img_name = current_img_name.replace(".jpg","")
        else:
            current_img_name = self.images[index]
            img_name = current_img_name.replace(".png","")
            img_name = img_name.split("_")[:-1]
            img_name = "_".join(img_name)
            
            current_hist_name = current_img_name.replace(".png","_hist.npy")
            current_patch_name = current_img_name

        
        id = int(self.ids[index])
        label = int(self.labels[index])  
        
        x = Image.open(os.path.join(self.patch_path, img_name+'_crops', current_patch_name)).convert('L') 
        
        out_hpf, _ = hpf(x)
        out_hpf = np.expand_dims(out_hpf, axis=2)
        out_hpf = self.transform(out_hpf)
        
        histogram =  np.load(os.path.join(self.hist_path, img_name+"_crops", current_hist_name))
        histogram = torch.tensor(histogram, dtype=torch.float).flatten() 
        histogram = torch.divide((histogram-self.min_hist), (self.max_hist-self.min_hist))
        
        
        return out_hpf, histogram, label, id 
