import torch
from argparse import ArgumentParser
from .incremental_learning import Inc_Learning_Appr
from datasets.exemplars_dataset import ExemplarsDataset
 

class Appr(Inc_Learning_Appr):
    """Class implementing the finetuning baseline"""

    def __init__(self, model, device,  nepochs=100, optimizer_type="adam", lr=0.05, lr_min=1e-4, lr_factor=3, lr_patience=5, clipgrad=10000,
                 momentum=0.9, wd=0.0002, 
                 logger=None, exemplars_dataset=None ):
        super(Appr, self).__init__(model, device,  nepochs, optimizer_type, lr, lr_min, lr_factor, lr_patience, clipgrad, momentum, wd,  logger,
                                   exemplars_dataset)
   

    @staticmethod
    def exemplars_dataset_class():
        return ExemplarsDataset

    @staticmethod
    def extra_parser(args):
        """Returns a parser containing the approach specific parameters"""
        parser = ArgumentParser()
        return parser.parse_known_args(args)

    def _get_optimizer(self):
        """Returns the optimizer"""

        if len(self.exemplars_dataset) == 0 and len(self.model.heads) > 1:
            # if there are no exemplars, previous heads are not modified
            params = list(self.model.model.parameters()) + list(self.model.heads[-1].parameters())
        else:
            params = self.model.parameters()

        if self.optimizer_type == "adam":
            print("Using adam optimizer")
            return torch.optim.Adam(params, lr=self.lr, weight_decay=self.wd)
        else:
            print("Using SGD")
            return torch.optim.SGD(params, lr=self.lr, weight_decay=self.wd, momentum=self.momentum)

    def train_loop(self, t, trn_loader, trn_loader_crops, val_loader):
        """Contains the epochs loop"""

        # add exemplars to train_loader
        if len(self.exemplars_dataset) > 0 and t > 0:
            trn_loader = torch.utils.data.DataLoader(trn_loader.dataset + self.exemplars_dataset,
                                                     batch_size=trn_loader.batch_size,
                                                     shuffle=True,
                                                     num_workers=trn_loader.num_workers,
                                                     pin_memory=trn_loader.pin_memory)

        # FINETUNING TRAINING -- contains the epochs loop
        super().train_loop(t, trn_loader, val_loader)

        # EXEMPLAR MANAGEMENT -- select training subset
        if t == 0:
            self.exemplars_dataset.collect_exemplars(self.model, trn_loader_crops, val_loader.dataset.transform)
        else:
            trn_loader_crops = torch.utils.data.DataLoader(trn_loader_crops.dataset + self.exemplars_dataset,
                                                    batch_size=trn_loader.batch_size,
                                                    shuffle=True,
                                                    num_workers=trn_loader.num_workers,
                                                    pin_memory=trn_loader.pin_memory)
                                                    
            self.exemplars_dataset.collect_exemplars(self.model, trn_loader_crops, val_loader.dataset.transform)

    
    def criterion(self, t, outputs, targets):
        """Returns the loss value"""
        if len(self.exemplars_dataset) > 0:
            return torch.nn.functional.cross_entropy(torch.cat(outputs, dim=1), targets)
        return torch.nn.functional.cross_entropy(outputs[t], targets - self.model.task_offset[t])
    
    
 