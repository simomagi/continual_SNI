import torch
import warnings
import numpy as np
from copy import deepcopy
from argparse import ArgumentParser
from torch.utils.data import DataLoader
import os 
from .incremental_learning import Inc_Learning_Appr
from datasets.exemplars_dataset import ExemplarsDataset
from datasets.exemplars_selection import override_dataset_transform
from utils import list_to_device

class Appr(Inc_Learning_Appr):
    """Class implementing the Incremental Classifier and Representation Learning (iCaRL) approach
    described in https://arxiv.org/abs/1611.07725
    Original code available at https://github.com/srebuffi/iCaRL
    """

    def __init__(self, model, device, nepochs=60, optimizer_type="adam", lr=0.5, lr_min=1e-4, lr_factor=3, lr_patience=5, clipgrad=10000,
                 momentum=0.9, wd=1e-5, logger=None, exemplars_dataset=None, lamb=1):
        
        super(Appr, self).__init__(model, device, nepochs,optimizer_type, lr, lr_min, lr_factor, lr_patience, clipgrad, momentum, wd,
                                   logger, exemplars_dataset)
        
        self.model_old = None
        self.lamb = lamb

        # iCaRL is expected to be used with exemplars. If needed to be used without exemplars, overwrite here the
        # `_get_optimizer` function with the one in LwF and update the criterion
        have_exemplars = self.exemplars_dataset.max_num_exemplars  
        if not have_exemplars:
            warnings.warn("Warning: iCaRL is expected to use exemplars. Check documentation.")

    @staticmethod
    def exemplars_dataset_class():
        return ExemplarsDataset

    @staticmethod
    def extra_parser(args):
        """Returns a parser containing the approach specific parameters"""
        parser = ArgumentParser()
        # Sec. 4. " allowing iCaRL to balance between CE and distillation loss."
        parser.add_argument('--lamb', default=1, type=float, required=False,
                            help='Forgetting-intransigence trade-off (default=%(default)s)')
        return parser.parse_known_args(args)

    # Algorithm 1: iCaRL NCM Classify
    def classify(self, task, features, targets):
        # expand means to all batch images
        means = torch.stack(self.exemplar_means)
        means = torch.stack([means] * features.shape[0])
        means = means.transpose(1, 2)
        # expand all features to all classes
        features = features / features.norm(dim=1).view(-1, 1)
        features = features.unsqueeze(2)
        features = features.expand_as(means)
        # get distances for all images to all exemplar class means -- nearest prototype
        dists = (features - means).pow(2).sum(1).squeeze()
        # Task-Aware Multi-Head
        num_cls = self.model.task_cls[task]
        offset = self.model.task_offset[task]
        pred = dists[:, offset:offset + num_cls].argmin(1)
        hits_taw = (pred + offset == targets.to(self.device)).float()
        # Task-Agnostic Multi-Head
        pred = dists.argmin(1)
        hits_tag = (pred == targets.to(self.device)).float()
        return hits_taw, hits_tag 

    def classify_images(self, task, features, targets):
        # expand means to all batch images
 
        n_images = targets.shape[0]
        means = torch.stack(self.exemplar_means)
        means = torch.stack([means] * features.shape[0])
        means = means.transpose(1, 2)
        # a single target for an image
        targets = targets[0].unsqueeze(0) 
        
        # expand all features to all classes
        features = features / features.norm(dim=1).view(-1, 1)
        features = features.unsqueeze(2)
        features = features.expand_as(means)
        # get distances for all the patches of an image to all exemplar class means -- nearest prototype
        dists = (features - means).pow(2).sum(1).squeeze()
        # get avg distance on each class mean 
        if n_images > 1:
            dists = torch.mean(dists, dim=0).unsqueeze(0) 
        else:
            dists = dists.unsqueeze(0)
 
        # Task-Aware Multi-Head
        num_cls = self.model.task_cls[task]
        offset = self.model.task_offset[task]
        pred = dists[:, offset:offset + num_cls].argmin(1)
        hits_taw = (pred + offset == targets.to(self.device)).float()
        # Task-Agnostic Multi-Head
        pred = dists.argmin(1)
        hits_tag = (pred == targets.to(self.device)).float()
        return hits_taw, hits_tag 



    def compute_mean_of_exemplars(self, trn_loader, transform):
        # change transforms to evaluation for this calculation
        with override_dataset_transform(self.exemplars_dataset, transform) as _ds:
            # change dataloader so it can be fixed to go sequentially (shuffle=False), this allows to keep same order
            icarl_loader = DataLoader(_ds, batch_size=trn_loader.batch_size, shuffle=False,
                                      num_workers=trn_loader.num_workers, pin_memory=trn_loader.pin_memory)
            # extract features from the model for all train samples
            # Page 2: "All feature vectors are L2-normalized, and the results of any operation on feature vectors,
            # e.g. averages are also re-normalized, which we do not write explicitly to avoid a cluttered notation."
            extracted_features = []
            extracted_targets = []
            with torch.no_grad():
                self.model.eval()
                for x, targets, ids  in icarl_loader:
                    x = list_to_device(x, self.device)
                    feats = self.model(x, return_features=True)[1]
                    # normalize
                    extracted_features.append(feats / feats.norm(dim=1).view(-1, 1))
                    extracted_targets.extend(targets)
            extracted_features = torch.cat(extracted_features)
            extracted_targets = np.array(extracted_targets)
            for curr_cls in np.unique(extracted_targets):
                # get all indices from current class
                cls_ind = np.where(extracted_targets == curr_cls)[0]
                # get all extracted features for current class
                cls_feats = extracted_features[cls_ind]
                # add the exemplars to the set and normalize
                cls_feats_mean = cls_feats.mean(0) / cls_feats.mean(0).norm()
                self.exemplar_means.append(cls_feats_mean)

    # Algorithm 2: iCaRL Incremental Train
    def train_loop(self, t, trn_loader, trn_loader_crops, val_loader):
        """Contains the epochs loop"""

        # remove mean of exemplars during training since Alg. 1 is not used during Alg. 2
        self.exemplar_means = []

        # Algorithm 3: iCaRL Update Representation
        # Alg. 3. "form combined training set", add exemplars to train_loader
        if t > 0:
            trn_loader = torch.utils.data.DataLoader(trn_loader.dataset + self.exemplars_dataset,
                                                     batch_size=trn_loader.batch_size,
                                                     shuffle=True,
                                                     num_workers=trn_loader.num_workers,
                                                     pin_memory=trn_loader.pin_memory)

        # FINETUNING TRAINING -- contains the epochs loop
        super().train_loop(t, trn_loader, val_loader)

        # EXEMPLAR MANAGEMENT -- select training subset
        # Algorithm 4: iCaRL ConstructExemplarSet and Algorithm 5: iCaRL ReduceExemplarSet
        if t == 0:
            self.exemplars_dataset.collect_exemplars(self.model, trn_loader_crops, val_loader.dataset.transform)
        else:
            trn_loader_crops = torch.utils.data.DataLoader(trn_loader_crops.dataset + self.exemplars_dataset,
                                                    batch_size=trn_loader.batch_size,
                                                    shuffle=True,
                                                    num_workers=trn_loader.num_workers,
                                                    pin_memory=trn_loader.pin_memory)
                                                    
            self.exemplars_dataset.collect_exemplars(self.model, trn_loader_crops, val_loader.dataset.transform)

        # compute mean of exemplars
        self.compute_mean_of_exemplars(trn_loader_crops, val_loader.dataset.transform)

    def post_train_process(self, t, trn_loader, trn_loader_crops):
        """Runs after training all the epochs of the task (after the train session)"""

        # Save old model to extract features later. This is different from the original approach, since they propose to
        #  extract the features and store them for future usage. However, when using data augmentation, it is easier to
        #  keep the model frozen and extract the features when needed.
        self.model_old = deepcopy(self.model)
        self.model_old.eval()
        self.model_old.freeze_all()
        current_exp_path = self.logger.exp_path
        torch.save(self.exemplar_means, os.path.join(current_exp_path, "exemplar_means_task_{}.pt".format(t)))
        


    def train_epoch(self, t, trn_loader):
        """Runs a single epoch"""
        self.model.train()
 
        for x, targets, ids  in trn_loader:
            # Forward old model
            outputs_old = None
            x = list_to_device(x, self.device)
            if t > 0:
                outputs_old = self.model_old(x)
            # Forward current model
            outputs = self.model(x)
            loss = self.criterion(t, outputs, targets.to(self.device), outputs_old)
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clipgrad)
            self.optimizer.step()

    def eval(self, t, val_loader):
        """Contains the evaluation code"""
        with torch.no_grad():
            total_loss_patch, total_acc_taw_patch, total_acc_tag_patch, total_num_patch = 0, 0, 0, 0

            total_loss_img, total_acc_taw_img, total_acc_tag_img  = 0, 0, 0, 
            overall_taw_probs, overall_tag_probs = [], []
            overall_targets = []
            overall_indices = []
            overall_features = []

            self.model.eval()

            for x, targets, ids in val_loader:
                # Forward old model
                outputs_old = None
                x = list_to_device(x, self.device)
                
                if t > 0:
                    outputs_old = self.model_old(x)
                # Forward current model
                outputs, feats = self.model(x, return_features=True)
                loss = self.criterion(t, outputs, targets.to(self.device), outputs_old)
                # during training, the usual accuracy is computed on the outputs
                if not self.exemplar_means:
                    hits_taw_patch, hits_tag_patch = self.calculate_metrics_per_patch(outputs, targets)
                else:
                    hits_taw_patch, hits_tag_patch = self.classify(t, feats, targets)
                # Log
                # Log
                total_loss_patch += loss.item() * len(targets)
                total_acc_taw_patch += hits_taw_patch.sum().item()
                total_acc_tag_patch += hits_tag_patch.sum().item()
                total_num_patch += len(targets)
                # accumulate output for image metric
                
                if not self.exemplar_means:
                   taw_probs = [torch.exp(torch.nn.functional.log_softmax(output, dim=1)) for output in outputs]
                   taw_probs = torch.cat(taw_probs, dim=1)
                   overall_taw_probs.append(taw_probs)
                   tag_probs = torch.cat(outputs, dim=1)
                   tag_probs = torch.exp(torch.nn.functional.log_softmax(tag_probs, dim=1))
                   overall_tag_probs.append(tag_probs)

                else:
                    overall_features.append(feats)

                overall_targets.append(targets)
                overall_indices.append(ids)
                

        # get last output to get classes per head 
        classes_per_head = [o.shape[1] for o in outputs]
        overall_indices = torch.cat(overall_indices)
        overall_targets = torch.cat(overall_targets)
        if not self.exemplar_means:
            overall_taw_probs = torch.cat(overall_taw_probs, dim=0)
            overall_tag_probs = torch.cat(overall_tag_probs, dim=0)
        else:
            overall_features = torch.cat(overall_features)

        set_ids = torch.unique(overall_indices)
        total_num_imgs = len(set_ids)
        for _id in set_ids:
            image_index = (overall_indices == _id)
            img_target = overall_targets[image_index]
            if not self.exemplar_means:
                taw_img_probs = overall_taw_probs[image_index, :]
                tag_img_probs = overall_tag_probs[image_index, :]
                hits_taw_img, hits_tag_img = self.calculate_metrics_per_image(taw_img_probs, tag_img_probs, 
                                                                              img_target, classes_per_head)
            else:
                img_feats = overall_features[image_index, :]
                hits_taw_img, hits_tag_img = self.classify_images(t, img_feats, img_target)


            total_acc_taw_img += hits_taw_img.sum().item()
            total_acc_tag_img += hits_tag_img.sum().item()

        return total_loss_patch / total_num_patch, total_acc_taw_patch / total_num_patch, \
                total_acc_tag_patch / total_num_patch, total_acc_taw_img/ total_num_imgs, total_acc_tag_img/total_num_imgs
    

    # Algorithm 3: classification and distillation terms -- original formulation has no trade-off parameter (lamb=1)
    def criterion(self, t, outputs, targets, outputs_old=None):
        """Returns the loss value"""

        # Classification loss for new classes
        loss = torch.nn.functional.cross_entropy(torch.cat(outputs, dim=1), targets)
        # Distillation loss for old classes
        if t > 0:
            # The original code does not match with the paper equation, maybe sigmoid could be removed from g
            g = torch.sigmoid(torch.cat(outputs[:t], dim=1))
            q_i = torch.sigmoid(torch.cat(outputs_old[:t], dim=1))
            loss += self.lamb * sum(torch.nn.functional.binary_cross_entropy(g[:, y], q_i[:, y]) for y in
                                    range(sum(self.model.task_cls[:t])))
        return loss
