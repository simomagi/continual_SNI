import os
import random
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torch 
import pickle
from torchvision import transforms
from os.path import join
import cv2 
from scipy.fft import  dctn, idctn
from copy import deepcopy
 


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


 

def preprocess_data(img_name, crop_id, dataset_type, 
                    crops_path, hist_path,  rgb_transform,
                    gray_transform,  residual_transform, 
                    max_hist,  min_hist):
        
    
    crops_set = img_name +"_crops"

    if dataset_type == "rgb":
        x = Image.open(os.path.join(crops_path, crops_set, img_name+"_{}.png".format(crop_id))).convert('RGB')
        final_data =  [rgb_transform(x)]
        
    elif dataset_type  == "gray":
        x = Image.open(os.path.join(crops_path, crops_set, img_name+"_{}.png".format(crop_id))).convert('L') 
        final_data = [gray_transform(x)]
 
    elif dataset_type == "residual":
        x = Image.open(os.path.join(crops_path, crops_set, img_name+"_{}.png".format(crop_id))).convert('L') 
        x, _ = hpf(x)
        x = np.expand_dims(x, axis=2)
        x = residual_transform(x) 
        final_data = [x]
        
    elif dataset_type == "residual_hist":
        x = Image.open(os.path.join(crops_path, crops_set, img_name+"_{}.png".format(crop_id))).convert('L') 
        x, _ = hpf(x)
        x = np.expand_dims(x, axis=2)
        x = residual_transform(x) 
        
        hist =  np.load(os.path.join(hist_path, crops_set, img_name+"_{}_hist.npy".format(crop_id)))
        hist = torch.tensor(hist, dtype=torch.float).flatten() 
        hist = torch.divide((hist-min_hist), ( max_hist- min_hist))
        final_data = [x, hist]

    return final_data

 

class BaseDataset(Dataset):
    """Characterizes a dataset for PyTorch -- this dataset pre-loads all paths in memory"""

    def __init__(self, data,  dataset_type, dataset_path,  
                 train=True, class_indices=None, return_image_path=False):
        """Initialization"""
        self.labels = data['y']
        self.images = data['x']
        self.ids = data['id']
        self.class_indices = class_indices
        self.train = train 
 
        
        self.dataset_type = dataset_type
        # path of the dataset
        self.dataset_path = dataset_path
        self.images_path = os.path.join(dataset_path, "Images")
        self.hist_path = os.path.join(dataset_path,"Histograms")
        self.info_path = os.path.join(dataset_path, "Info")
        self.crops_path = os.path.join(dataset_path, "Patches_256")
        self.prnu_path = os.path.join(dataset_path, "PrnuResidual")
        self.return_image_path = return_image_path
        
        # 2D preprocessings
        self.rgb_mean, self.rgb_std = [0.4691, 0.4504, 0.4212], [0.7639, 0.7751, 0.7913] #[0.2615, 0.2554, 0.2640] 
        self.gray_mean, self.gray_std =  [0.4526], [0.2530] 
        self.residual_mean, self.residual_std = [0.4999687373638153], [0.0446637235581874]
        self.rgb_transform = [transforms.ToTensor(), 
                              transforms.Normalize(mean=self.rgb_mean, 
                                                   std=self.rgb_std)]
        self.gray_transform = [transforms.ToTensor(), 
                               transforms.Normalize(mean=self.gray_mean, 
                                                   std=self.gray_std)]
        self.residual_transform = [transforms.ToTensor(), 
                                  transforms.Normalize(mean= self.residual_mean, 
                                                      std=self.residual_std)]
        # 1D preprocessings 
        self.max_hist = torch.tensor(np.load(os.path.join(self.info_path, "max_hist.npy")))
        self.min_hist = torch.tensor(np.load(os.path.join(self.info_path, "min_hist.npy")))
        
        self.transform = None 

        if self.train:
            with open(join(self.info_path,'train_crop_count.pickle'), 'rb') as handle:
               self.count_crops = pickle.load(handle)
        else: 
            self.count_crops = None
 
 
        self.rgb_transform = transforms.Compose([transforms.ToTensor(), 
                            transforms.Normalize(mean=self.rgb_mean, 
                                                std=self.rgb_std)])
        self.gray_transform = transforms.Compose([transforms.ToTensor(), 
                            transforms.Normalize(mean=self.gray_mean, 
                                                std=self.gray_std)])
        
        self.residual_transform = transforms.Compose([transforms.ToTensor(), 
                                transforms.Normalize(mean=self.residual_mean, 
                                                    std=self.residual_std)])
 
 

    def __len__(self):
        """Denotes the total number of samples"""
        return len(self.images)

    def __getitem__(self, index):
        
        if self.train:
            orig_img  = self.images[index] # with jpeg extension (original smart data)
            # generated_random during training
            crop_id = random.randint(0, self.count_crops[orig_img]-1)
            img_name = orig_img.replace(".jpg","") # name of the image without jpg extension
        else:
            orig_img = self.images[index] # with png extension (crop)
            crop_id =  orig_img.replace(".png","").split("_")[-1]
            img_name = "_".join(orig_img.replace(".png","").split("_")[:-1]) # name of the image without crop id and png ext.
            
            
        id = int(self.ids[index])
        label = int(self.labels[index])

        x = preprocess_data(img_name, crop_id, self.dataset_type, self.crops_path, 
                            self.hist_path, self.rgb_transform, self.gray_transform, self.residual_transform, 
                            self.max_hist, self.min_hist)
        

        if self.return_image_path:
            
            return x, label, id, orig_img
        else:
            
            return x, label, id

    



        


def get_data(path, num_tasks, nc_first_task, validation, shuffle_classes, class_order=None):
    """Prepare data: dataset splits, task partition, class order"""

    data = {}
    taskcla = []

    # read filenames and labels
    trn_lines = np.loadtxt(os.path.join(path,"Info",'train.txt'), dtype=str)

    tst_lines = np.loadtxt(os.path.join(path,"Info", 'test_crops.txt'), dtype=str)

    # dataset for exemplar and ewc based approach
    trn_lines_crops = np.loadtxt(os.path.join(path,"Info","train_crops.txt"),dtype=str)

    #ADD read validation lines  
    val_lines = np.loadtxt(os.path.join(path, "Info", 'val_crops.txt'), dtype=str)

    if class_order is None:
        num_classes = len(np.unique(trn_lines[:, 1]))
        class_order = list(range(num_classes))
    else:
        num_classes = len(class_order)
        class_order = class_order.copy()
    if shuffle_classes:
        np.random.shuffle(class_order)

    # compute classes per task and num_tasks
    if nc_first_task is None:
        cpertask = np.array([num_classes // num_tasks] * num_tasks)
        for i in range(num_classes % num_tasks):
            cpertask[i] += 1
    else:
        assert nc_first_task < num_classes, "first task wants more classes than exist"
        remaining_classes = num_classes - nc_first_task
        assert remaining_classes >= (num_tasks - 1), "at least one class is needed per task"  # better minimum 2
        cpertask = np.array([nc_first_task] + [remaining_classes // (num_tasks - 1)] * (num_tasks - 1))
        for i in range(remaining_classes % (num_tasks - 1)):
            cpertask[i + 1] += 1

    assert num_classes == cpertask.sum(), "something went wrong, the split does not match num classes"
    cpertask_cumsum = np.cumsum(cpertask)
    init_class = np.concatenate(([0], cpertask_cumsum[:-1]))

    # initialize data structure
    for tt in range(num_tasks):
        data[tt] = {}
        data[tt]['name'] = 'task-' + str(tt)
        data[tt]['trn'] = {'x': [], 'y': [], 'id':[]}
        data[tt]['val'] = {'x': [], 'y': [], 'id':[]}
        data[tt]['tst'] = {'x': [], 'y': [], 'id':[]}
        data[tt]['trn_crops'] = {'x': [], 'y': [], 'id':[]}

    # ALL OR TRAIN
    for this_image, this_label, this_id in trn_lines:
        #if not os.path.isabs(this_image):
        #    this_image = os.path.join(path, this_image)
        this_label = int(this_label)
        if this_label not in class_order:
            continue
        # If shuffling is false, it won't change the class number
        this_label = class_order.index(this_label)

        # add it to the corresponding split
        this_task = (this_label >= cpertask_cumsum).sum()
        data[this_task]['trn']['x'].append(str(this_image))
        data[this_task]['trn']['y'].append(this_label - init_class[this_task])
        data[this_task]['trn']['id'].append(this_id)
    
    # ALL OR TEST
    for this_image, this_label, this_id in tst_lines:
        #if not os.path.isabs(this_image):
        #    this_image = os.path.join(path, this_image)
        this_label = int(this_label)
        if this_label not in class_order:
            continue
        # If shuffling is false, it won't change the class number
        this_label = class_order.index(this_label)

        # add it to the corresponding split
        this_task = (this_label >= cpertask_cumsum).sum()
        data[this_task]['tst']['x'].append(str(this_image))
        data[this_task]['tst']['y'].append(this_label - init_class[this_task])
        data[this_task]['tst']['id'].append(this_id)
    
    # ALL OR VAL  (ADDED)

    for this_image, this_label, this_id in val_lines:
        #if not os.path.isabs(this_image):
        #    this_image = os.path.join(path, this_image)
        this_label = int(this_label)
        if this_label not in class_order:
            continue
        # If shuffling is false, it won't change the class number
        this_label = class_order.index(this_label)

        # add it to the corresponding split
        this_task = (this_label >= cpertask_cumsum).sum()
        data[this_task]['val']['x'].append(str(this_image))
        data[this_task]['val']['y'].append(this_label - init_class[this_task])
        data[this_task]['val']['id'].append(this_id)
    
    # Train Crops For Exemplar AND EWC BASED APPROACH

    for this_image, this_label, this_id in trn_lines_crops:
        #if not os.path.isabs(this_image):
        #    this_image = os.path.join(path, this_image)
        this_label = int(this_label)
        if this_label not in class_order:
            continue
        # If shuffling is false, it won't change the class number
        this_label = class_order.index(this_label)

        # add it to the corresponding split
        this_task = (this_label >= cpertask_cumsum).sum()
        data[this_task]['trn_crops']['x'].append(str(this_image))
        data[this_task]['trn_crops']['y'].append(this_label - init_class[this_task])
        data[this_task]['trn_crops']['id'].append(this_id)
    


    # check classes
    for tt in range(num_tasks):
        data[tt]['ncla'] = len(np.unique(data[tt]['trn']['y']))
        assert data[tt]['ncla'] == cpertask[tt], "something went wrong splitting classes"
    

    # other
    n = 0
    for t in data.keys():
        taskcla.append((t, data[t]['ncla']))
        n += data[t]['ncla']
    data['ncla'] = n

    return data, taskcla, class_order
