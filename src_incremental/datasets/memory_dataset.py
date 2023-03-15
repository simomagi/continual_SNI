import random
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from .base_dataset import preprocess_data 
import os 
import torch 

class MemoryDataset(Dataset):
    """Characterizes a dataset for PyTorch -- this dataset pre-loads all images in memory"""

    def __init__(self, data,  dataset_type, dataset_path, class_indices=None):
        """Initialization"""
        self.labels = data['y']
        self.images = data['x']
        self.ids = data['ids']
        self.class_indices = class_indices
 
        self.dataset_type = dataset_type
        
        # path of the dataset
        self.dataset_path = dataset_path
        self.images_path = os.path.join(dataset_path, "Images")
        self.hist_path = os.path.join(dataset_path,"Histograms")
        self.info_path = os.path.join(dataset_path, "Info")
        self.crops_path = os.path.join(dataset_path, "Patches_256")
        self.prnu_path = os.path.join(dataset_path, "PrnuResidual")
        
        # 2D preprocessings
        self.rgb_mean, self.rgb_std = [0.4691, 0.4504, 0.4212], [0.2615, 0.2554, 0.2640] 
        self.gray_mean, self.gray_std =  [0.4526], [0.2530] 
        self.residual_mean, self.residual_std = [0.4999687373638153], [0.0446637235581874]
        
        self.transform = None 
           
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
        
 
 
        self.rgb_transform = transforms.Compose([transforms.ToTensor(), 
                            transforms.Normalize(mean=self.rgb_mean, 
                                                std=self.rgb_std)])
        self.gray_transform = transforms.Compose([transforms.ToTensor(), 
                            transforms.Normalize(mean=self.gray_mean, 
                                                std=self.gray_std)])
        
        self.residual_transform = transforms.Compose([transforms.ToTensor(), 
                                transforms.Normalize(mean= self.residual_mean, 
                                                    std=self.residual_std)])
             
 

    def __len__(self):
        """Denotes the total number of samples"""
        return len(self.images)

    def __getitem__(self, index):
        orig_img = self.images[index] # with png extension (crop)
        crop_id =  orig_img.replace(".png","").split("_")[-1]
        img_name = "_".join(orig_img.replace(".png","").split("_")[:-1]) # name of the image without crop id and png ext.
            
        id = int(self.ids[index])
        label = int(self.labels[index])

        x = preprocess_data(img_name, crop_id, self.dataset_type, self.crops_path, 
                        self.hist_path, self.rgb_transform, self.gray_transform, self.residual_transform, 
                        self.max_hist, self.min_hist)
        

        return x, label, id  

    


def get_data(trn_data, tst_data, num_tasks, nc_first_task, validation, shuffle_classes, class_order=None):
    """Prepare data: dataset splits, task partition, class order"""

    data = {}
    taskcla = []
    if class_order is None:
        num_classes = len(np.unique(trn_data['y']))
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
        data[tt]['trn'] = {'x': [], 'y': []}
        data[tt]['val'] = {'x': [], 'y': []}
        data[tt]['tst'] = {'x': [], 'y': []}

    # ALL OR TRAIN
    filtering = np.isin(trn_data['y'], class_order)
    if filtering.sum() != len(trn_data['y']):
        trn_data['x'] = trn_data['x'][filtering]
        trn_data['y'] = np.array(trn_data['y'])[filtering]
    for this_image, this_label in zip(trn_data['x'], trn_data['y']):
        # If shuffling is false, it won't change the class number
        this_label = class_order.index(this_label)
        # add it to the corresponding split
        this_task = (this_label >= cpertask_cumsum).sum()
        data[this_task]['trn']['x'].append(this_image)
        data[this_task]['trn']['y'].append(this_label - init_class[this_task])

    # ALL OR TEST
    filtering = np.isin(tst_data['y'], class_order)
    if filtering.sum() != len(tst_data['y']):
        tst_data['x'] = tst_data['x'][filtering]
        tst_data['y'] = tst_data['y'][filtering]
    for this_image, this_label in zip(tst_data['x'], tst_data['y']):
        # If shuffling is false, it won't change the class number
        this_label = class_order.index(this_label)
        # add it to the corresponding split
        this_task = (this_label >= cpertask_cumsum).sum()
        data[this_task]['tst']['x'].append(this_image)
        data[this_task]['tst']['y'].append(this_label - init_class[this_task])

    # check classes
    for tt in range(num_tasks):
        data[tt]['ncla'] = len(np.unique(data[tt]['trn']['y']))
        assert data[tt]['ncla'] == cpertask[tt], "something went wrong splitting classes"

    # validation
    if validation > 0.0:
        for tt in data.keys():
            for cc in range(data[tt]['ncla']):
                cls_idx = list(np.where(np.asarray(data[tt]['trn']['y']) == cc)[0])
                rnd_img = random.sample(cls_idx, int(np.round(len(cls_idx) * validation)))
                rnd_img.sort(reverse=True)
                for ii in range(len(rnd_img)):
                    data[tt]['val']['x'].append(data[tt]['trn']['x'][rnd_img[ii]])
                    data[tt]['val']['y'].append(data[tt]['trn']['y'][rnd_img[ii]])
                    data[tt]['trn']['x'].pop(rnd_img[ii])
                    data[tt]['trn']['y'].pop(rnd_img[ii])

    # convert them to numpy arrays
    for tt in data.keys():
        for split in ['trn', 'val', 'tst']:
            data[tt][split]['x'] = np.asarray(data[tt][split]['x'])

    # other
    n = 0
    for t in data.keys():
        taskcla.append((t, data[t]['ncla']))
        n += data[t]['ncla']
    data['ncla'] = n

    return data, taskcla, class_order