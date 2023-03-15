import torch
from argparse import ArgumentParser
from torch.utils.data import DataLoader, Dataset
from .incremental_learning import Inc_Learning_Appr
from datasets.exemplars_dataset import ExemplarsDataset
from utils import list_to_device

class Appr(Inc_Learning_Appr):
    """Class implementing the joint baseline"""

    def __init__(self, model, device, nepochs=100, optimizer_type="adam",  lr=0.05, lr_min=1e-4, lr_factor=3, lr_patience=5, clipgrad=10000,
                 momentum=0, wd=0,   logger=None, exemplars_dataset=None):
        super(Appr, self).__init__(model, device, nepochs, optimizer_type, lr, lr_min, lr_factor, lr_patience, clipgrad, momentum, wd,
                                   logger, exemplars_dataset)
        
        self.trn_datasets = []
        self.val_datasets = []

        have_exemplars = self.exemplars_dataset.max_num_exemplars  
        assert (have_exemplars == 0), 'Warning: Joint does not use exemplars. Comment this line to force it.'

    @staticmethod
    def exemplars_dataset_class():
        return ExemplarsDataset
        
    @staticmethod
    def extra_parser(args):
        """Returns a parser containing the approach specific parameters"""
        parser = ArgumentParser()
 
        return parser.parse_known_args(args)

    def post_train_process(self, t, trn_loader, trn_loader_crops):
        pass 

    def train_loop(self, t, trn_loader, trn_loader_crops, val_loader):
        """Contains the epochs loop"""

        # add new datasets to existing cumulative ones
        self.trn_datasets.append(trn_loader.dataset)
        self.val_datasets.append(val_loader.dataset)
        trn_dset = JointDataset(self.trn_datasets)
        val_dset = JointDataset(self.val_datasets)
        trn_loader = DataLoader(trn_dset,
                                batch_size=trn_loader.batch_size,
                                shuffle=True,
                                num_workers=trn_loader.num_workers,
                                pin_memory=trn_loader.pin_memory)
        val_loader = DataLoader(val_dset,
                                batch_size=val_loader.batch_size,
                                shuffle=False,
                                num_workers=val_loader.num_workers,
                                pin_memory=val_loader.pin_memory)
        # continue training as usual
        super().train_loop(t, trn_loader, val_loader)

    def train_epoch(self, t, trn_loader):
        """Runs a single epoch"""  
        self.model.train()
        for x, targets, _  in  trn_loader:
            x = list_to_device(x, self.device)
            outputs = self.model(x)
                
            loss = self.criterion(t, outputs, targets.to(self.device))
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clipgrad)
            self.optimizer.step()

    def criterion(self, t, outputs, targets):
        """Returns the loss value"""
        return torch.nn.functional.cross_entropy(torch.cat(outputs, dim=1), targets)


class JointDataset(Dataset):
    """Characterizes a dataset for PyTorch -- this dataset accumulates each task dataset incrementally"""

    def __init__(self, datasets):
        self.datasets = datasets
        self._len = sum([len(d) for d in self.datasets])

    def __len__(self):
        'Denotes the total number of samples'
        return self._len

    def __getitem__(self, index):
        for d in self.datasets:
            if len(d) <= index:
                index -= len(d)
            else:
                x, y, id  = d[index]
                return x, y, id 
