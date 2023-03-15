import random
import time
from contextlib import contextmanager
from typing import Iterable
import numpy as np
from torch.utils.data import DataLoader, ConcatDataset 
from datasets.exemplars_dataset import ExemplarsDataset
from networks.network import LLL_Net

class ExemplarsSelector:
    """Exemplar selector for approaches with an interface of Dataset"""

    def __init__(self, exemplars_dataset: ExemplarsDataset):
        self.exemplars_dataset = exemplars_dataset

    def __call__(self, model: LLL_Net, trn_loader: DataLoader, transform):
        clock0 = time.time()
        exemplars_per_class = self._exemplars_per_class_num(model)
        ds_for_selection = trn_loader.dataset 
        sel_loader = DataLoader(ds_for_selection, batch_size=trn_loader.batch_size, shuffle=False,
                                num_workers=trn_loader.num_workers, pin_memory=trn_loader.pin_memory)
        x, y, ids,  dataset_type,  dataset_path  = self._select_exemplars(model, sel_loader, exemplars_per_class)
 
 
        clock1 = time.time()
        print('| Selected {:d} train exemplars, time={:5.1f}s'.format(len(x), clock1 - clock0))
        return x, y, ids,  dataset_type,  dataset_path 
    

    def _exemplars_per_class_num(self, model: LLL_Net):
        
        num_cls = model.task_cls.sum().item()
        num_exemplars = self.exemplars_dataset.max_num_exemplars
        exemplars_per_class = int(np.ceil(num_exemplars / num_cls))
        assert exemplars_per_class > 0, \
            "Not enough exemplars to cover all classes!\n" \
            "Number of classes so far: {}. " \
            "Limit of exemplars: {}".format(num_cls,
                                            num_exemplars)
        return exemplars_per_class

    def _select_exemplars(self, model: LLL_Net, sel_loader: DataLoader, exemplars_per_class: int) -> Iterable:
        pass


class RandomExemplarsSelector(ExemplarsSelector):
    """Selection of new samples. This is based on random selection, which produces a random list of samples."""

    def __init__(self, exemplars_dataset):
        super().__init__(exemplars_dataset)

    def _select_exemplars(self, model: LLL_Net, sel_loader: DataLoader, exemplars_per_class: int) -> Iterable:
        num_cls = sum(model.task_cls)
        result = []
        labels = self._get_labels(sel_loader)
        images, dataset_type,  dataset_path = self._get_image_paths(sel_loader)
        ids = self._get_image_ids(sel_loader)

        for curr_cls in range(num_cls):
            # get all indices from current class -- check if there are exemplars from previous task in the loader
            cls_ind = np.where(labels == curr_cls)[0]
            assert (len(cls_ind) > 0), "No samples to choose from for class {:d}".format(curr_cls)
            assert (exemplars_per_class <= len(cls_ind)), "Not enough samples to store"
            # select the exemplars randomly
            result.extend(random.sample(list(cls_ind), exemplars_per_class))
        
        x = [images[idx] for idx in result]
        y = [labels[idx] for idx in result]
        ids = [ids[idx] for idx in result]
        return x, y, ids, dataset_type,  dataset_path 

    def _get_image_paths(self, sel_loader):
        if hasattr(sel_loader.dataset, 'images'):
            images = sel_loader.dataset.images
            
            dataset_type = sel_loader.dataset.dataset_type
            dataset_path = sel_loader.dataset.dataset_path
 
            
        elif isinstance(sel_loader.dataset, ConcatDataset):
            images = []
            for ds in sel_loader.dataset.datasets:
                images.extend(ds.images)
                dataset_type = ds.dataset_type
                dataset_path = ds.dataset_path
 
        else:
            raise RuntimeError("Unsupported dataset: {}".format(sel_loader.dataset.__class__.__name__))

        return images, dataset_type,  dataset_path 
    
    def _get_image_ids(self, sel_loader):
        if hasattr(sel_loader.dataset, 'ids'):
            ids = sel_loader.dataset.ids
        elif isinstance(sel_loader.dataset, ConcatDataset):
            ids = []
            for ds in sel_loader.dataset.datasets:
                ids.extend(ds.ids)
        else:
            raise RuntimeError("Unsupported dataset: {}".format(sel_loader.dataset.__class__.__name__))

        return ids
    

    def _get_labels(self, sel_loader):
        if hasattr(sel_loader.dataset, 'labels'):  # BaseDataset, MemoryDataset
            labels = np.asarray(sel_loader.dataset.labels)
        elif isinstance(sel_loader.dataset, ConcatDataset):
            labels = []
            for ds in sel_loader.dataset.datasets:
                labels.extend(ds.labels)
            labels = np.array(labels)
        else:
            raise RuntimeError("Unsupported dataset: {}".format(sel_loader.dataset.__class__.__name__))
        return labels



def dataset_transforms(dataset, transform_to_change):
    if isinstance(dataset, ConcatDataset):
        r = []
        for ds in dataset.datasets:
            r += dataset_transforms(ds, transform_to_change)
        return r
    else:
        old_transform = dataset.transform
        dataset.transform = transform_to_change
        return [(dataset, old_transform)]


@contextmanager
def override_dataset_transform(dataset, transform):
    try:
        datasets_with_orig_transform = dataset_transforms(dataset, transform)
        yield dataset
    finally:
        # get bac original transformations
        for ds, orig_transform in datasets_with_orig_transform:
            ds.transform = orig_transform
