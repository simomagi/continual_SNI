from torch.utils import data
from . import base_dataset as basedat
from .dataset_config import dataset_config


 

def get_loaders(datasets, num_tasks, nc_first_task, batch_size,  num_workers, pin_memory,
                dataset_type, return_image_path=False, validation=.1):
    """Apply transformations to Datasets and create the DataLoaders for each task"""

    trn_load, val_load, tst_load = [], [], []
    
    trn_crops_load = [] # data loader for exemplar and ewc consolidation based approach

    taskcla = []
    dataset_offset = 0
    for idx_dataset, cur_dataset in enumerate(datasets, 0):
        # get configuration for current dataset
        dc = dataset_config[cur_dataset]

        # datasets
        trn_dset, val_dset, tst_dset, trn_crops_dset,  curtaskcla = get_datasets(cur_dataset, dc['path'], num_tasks, nc_first_task,
                                                                validation=validation,
                                                            
                                                                dataset_type=dataset_type,
                                                                dataset_path=dc["path"],
                                                                return_image_path=return_image_path,
                                                                class_order=dc['class_order']
                                                                )

        # apply offsets in case of multiple datasets
        if idx_dataset > 0:
            for tt in range(num_tasks):
                trn_dset[tt].labels = [elem + dataset_offset for elem in trn_dset[tt].labels]
                val_dset[tt].labels = [elem + dataset_offset for elem in val_dset[tt].labels]
                tst_dset[tt].labels = [elem + dataset_offset for elem in tst_dset[tt].labels]
                trn_crops_dset[tt].labels = [elem + dataset_offset for elem in trn_crops_dset[tt].labels]

        dataset_offset = dataset_offset + sum([tc[1] for tc in curtaskcla])

        # reassign class idx for multiple dataset case
        curtaskcla = [(tc[0] + idx_dataset * num_tasks, tc[1]) for tc in curtaskcla]

        # extend final taskcla list
        taskcla.extend(curtaskcla)

        # loaders
        for tt in range(num_tasks):
            trn_load.append(data.DataLoader(trn_dset[tt], batch_size=batch_size, shuffle=True, num_workers=num_workers,
                                            pin_memory=pin_memory))
            val_load.append(data.DataLoader(val_dset[tt], batch_size=batch_size*2, shuffle=False, num_workers=num_workers ,
                                            pin_memory=pin_memory))
            tst_load.append(data.DataLoader(tst_dset[tt], batch_size=batch_size*2, shuffle=False, num_workers=num_workers,
                                            pin_memory=pin_memory))

            trn_crops_load.append(data.DataLoader(trn_crops_dset[tt], batch_size=batch_size, shuffle=False, num_workers=num_workers,
                                 pin_memory=pin_memory))

    return trn_load, val_load, tst_load, trn_crops_load, taskcla


def get_datasets(dataset, path, num_tasks, nc_first_task, validation, 
                 dataset_type, dataset_path,  return_image_path=False, class_order=None):

    """Extract datasets and create Dataset class"""

    trn_dset, val_dset, tst_dset = [], [], []

    trn_crops_dset = [] # dataset for exemplar and ewc based approach 

    # read data paths and compute splits -- path needs to have a train.txt and a test.txt with image-label pairs
    all_data, taskcla, class_indices = basedat.get_data(path,  num_tasks=num_tasks, nc_first_task=nc_first_task,
                                                        validation=validation, shuffle_classes=class_order is None,
                                                        class_order=class_order)
    # set dataset type
    Dataset = basedat.BaseDataset

    # get datasets, apply correct label offsets for each task
    offset = 0
    for task in range(num_tasks):
        all_data[task]['trn']['y'] = [label + offset for label in all_data[task]['trn']['y']]
        all_data[task]['val']['y'] = [label + offset for label in all_data[task]['val']['y']]
        all_data[task]['tst']['y'] = [label + offset for label in all_data[task]['tst']['y']]

        all_data[task]['trn_crops']['y'] = [label + offset for label in all_data[task]['trn_crops']['y']]

        trn_dset.append(Dataset(all_data[task]['trn'], dataset_type, dataset_path,  train=True, class_indices=class_indices, 
                                return_image_path=return_image_path))
        
        trn_crops_dset.append(Dataset(all_data[task]['trn_crops'],  dataset_type, dataset_path,     train=False, class_indices=class_indices, 
                                   return_image_path=return_image_path))
  
        val_dset.append(Dataset(all_data[task]['val'],  dataset_type, dataset_path,   train=False, class_indices=class_indices, 
                                 return_image_path=return_image_path))
        tst_dset.append(Dataset(all_data[task]['tst'], dataset_type, dataset_path,  train=False, class_indices=class_indices, 
                        return_image_path=return_image_path))
        
        offset += taskcla[task][1]

    return trn_dset, val_dset, tst_dset, trn_crops_dset,  taskcla


 
