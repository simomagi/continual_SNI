import torch
import numpy as np

from .incremental_learning import Inc_Learning_Appr
from datasets.exemplars_dataset import ExemplarsDataset
from tqdm import tqdm
from utils import list_to_device
import os 

class Appr(Inc_Learning_Appr):
    """Class implementing the Class Incremental Learning With Dual Memory (IL2M) approach described in
    https://openaccess.thecvf.com/content_ICCV_2019/papers/Belouadah_IL2M_Class_Incremental_Learning_With_Dual_Memory_ICCV_2019_paper.pdf
    """

    def __init__(self, model, device, nepochs=100, optimizer_type="adam", lr=0.1, lr_min=1e-4, lr_factor=3, lr_patience=10, clipgrad=10000,
                 momentum=0.9, wd=0.0001,  logger=None, exemplars_dataset=None):
        super(Appr, self).__init__(model, device, nepochs, optimizer_type, lr, lr_min, lr_factor, lr_patience, clipgrad, momentum, wd,
                                   logger, exemplars_dataset)
        
        self.init_classes_means = []
        self.current_classes_means = []
        self.models_confidence = []
        # FLAG to not do scores rectification while finetuning training
        self.ft_train = False

        have_exemplars = self.exemplars_dataset.max_num_exemplars  
        assert (have_exemplars > 0), 'Error: IL2M needs exemplars.'

    @staticmethod
    def exemplars_dataset_class():
        return ExemplarsDataset

    def il2m(self, t, trn_loader):
        """Compute and store statistics for score rectification"""
        old_classes_number = sum(self.model.task_cls[:t])
        classes_counts = [0 for _ in range(sum(self.model.task_cls))]
        models_counts = 0

        # to store statistics for the classes as learned in the current incremental state
        self.current_classes_means = [0 for _ in range(old_classes_number)]
        # to store statistics for past classes as learned in their initial states
        for cls in range(old_classes_number, old_classes_number + self.model.task_cls[t]):
            self.init_classes_means.append(0)
        # to store statistics for model confidence in different states (i.e. avg top-1 pred scores)
        self.models_confidence.append(0)
        print("IL2M Rectification")
        # compute the mean prediction scores that will be used to rectify scores in subsequent tasks
        with torch.no_grad():
            self.model.eval()
            for x, targets, _ in tqdm(trn_loader):
                x = list_to_device(x, self.device)
                
                outputs = self.model(x)
                scores = np.array(torch.cat(outputs, dim=1).data.cpu().numpy(), dtype=np.float32)
                for m in range(len(targets)):
                    if targets[m] < old_classes_number:
                        # computation of class means for past classes of the current state.
                        self.current_classes_means[targets[m]] += scores[m, targets[m]]
                        classes_counts[targets[m]] += 1
                    else:
                        # compute the mean prediction scores for the new classes of the current state
                        self.init_classes_means[targets[m]] += scores[m, targets[m]]
                        classes_counts[targets[m]] += 1
                        # compute the mean top scores for the new classes of the current state
                        self.models_confidence[t] += np.max(scores[m, ])
                        models_counts += 1
        # Normalize by corresponding number of images
        for cls in range(old_classes_number):
            self.current_classes_means[cls] /= classes_counts[cls]
        for cls in range(old_classes_number, old_classes_number + self.model.task_cls[t]):
            self.init_classes_means[cls] /= classes_counts[cls]
        self.models_confidence[t] /= models_counts
        
        current_exp_path = self.logger.exp_path
    
        torch.save(self.init_classes_means, os.path.join(current_exp_path, "init_classes_means_task_{}.pt".format(t)))
        torch.save(self.current_classes_means, os.path.join(current_exp_path, "current_classes_means_task_{}.pt".format(t)))
        torch.save(self.models_confidence, os.path.join(current_exp_path, "models_confidence_task_{}.pt".format(t)))

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
        self.ft_train = True
        super().train_loop(t, trn_loader, val_loader)
        self.ft_train = False

        if t == 0:
            # IL2M outputs rectification
            self.il2m(t, trn_loader_crops)

            self.exemplars_dataset.collect_exemplars(self.model, trn_loader_crops, val_loader.dataset.transform)
        else:
            trn_loader_crops = torch.utils.data.DataLoader(trn_loader_crops.dataset + self.exemplars_dataset,
                                                    batch_size=trn_loader.batch_size,
                                                    shuffle=True,
                                                    num_workers=trn_loader.num_workers,
                                                    pin_memory=trn_loader.pin_memory)
            # IL2M outputs rectification
            self.il2m(t, trn_loader_crops)
                                                    
            self.exemplars_dataset.collect_exemplars(self.model, trn_loader_crops, val_loader.dataset.transform)
    
        
    def eval(self, t, val_loader):
        """Contains the evaluation code"""
        with torch.no_grad():
            total_loss_patch, total_acc_taw_patch, total_acc_tag_patch, total_num_patch = 0, 0, 0, 0

            total_loss_img, total_acc_taw_img, total_acc_tag_img  = 0, 0, 0, 
            overall_taw_probs, overall_tag_probs = [], []
            self.model.eval()
 
            overall_targets = []
            overall_indices = []
            overall_outs = []
            from tqdm import tqdm 
          

            for x, targets, ids  in tqdm(val_loader):
                x = list_to_device(x, self.device)
                
                outputs = self.model(x)
     
                loss = self.criterion(t, outputs, targets.to(self.device))
                hits_taw_patch, hits_tag_patch = self.calculate_metrics_per_patch(outputs, targets)
                # Log
                total_loss_patch += loss.item() * len(targets)
                total_acc_taw_patch += hits_taw_patch.sum().item()
                total_acc_tag_patch += hits_tag_patch.sum().item()
                total_num_patch += len(targets)
                # accumulate output for image metric
                taw_probs = [torch.exp(torch.nn.functional.log_softmax(output, dim=1)) for output in outputs]
                taw_probs = torch.cat(taw_probs, dim=1)
                overall_taw_probs.append(taw_probs)

                tag_probs = torch.cat(outputs, dim=1)
                tag_probs = torch.exp(torch.nn.functional.log_softmax(tag_probs, dim=1))
                overall_tag_probs.append(tag_probs)

                overall_targets.append(targets)
                overall_indices.append(ids)
                overall_outs.append(torch.cat(outputs, dim=1))
                 
            # get last output to get classes per head 
            classes_per_head = [o.shape[1] for o in outputs]

            overall_outs = torch.cat(overall_outs, dim=0)
            overall_indices = torch.cat(overall_indices)
            overall_targets = torch.cat(overall_targets)
            overall_taw_probs = torch.cat(overall_taw_probs, dim=0)
            overall_tag_probs = torch.cat(overall_tag_probs, dim=0)

            set_ids = torch.unique(overall_indices)

            total_num_imgs = len(set_ids)
            for _id in set_ids:
                image_index = (overall_indices == _id)
                img_target = overall_targets[image_index]
                taw_img_probs = overall_taw_probs[image_index, :]
                tag_img_probs = overall_tag_probs[image_index, :]
                img_outs = overall_outs[image_index, :]
                
                hits_taw_img, hits_tag_img = self.calculate_metrics_per_image(taw_img_probs, tag_img_probs, img_target, classes_per_head, img_outs)
                total_acc_taw_img += hits_taw_img.sum().item()
                total_acc_tag_img += hits_tag_img.sum().item()



            return total_loss_patch / total_num_patch, total_acc_taw_patch / total_num_patch, \
                   total_acc_tag_patch / total_num_patch, total_acc_taw_img/ total_num_imgs, total_acc_tag_img/total_num_imgs
    
    

 

    def calculate_metrics_per_patch(self, outputs, targets):
        """Contains the main Task-Aware and Task-Agnostic metrics"""
        if self.ft_train:
            # no score rectification while training
            hits_taw, hits_tag = super().calculate_metrics_per_patch(outputs, targets)
        else:
            # Task-Aware Multi-Head
            pred = torch.zeros_like(targets.to(self.device))
            for m in range(len(pred)):
                this_task = (self.model.task_cls.cumsum(0) <= targets[m]).sum()
                pred[m] = outputs[this_task][m].argmax() + self.model.task_offset[this_task]
            hits_taw = (pred == targets.to(self.device)).float()
            # Task-Agnostic Multi-Head
 
            # Eq. 1: rectify predicted scores
            old_classes_number = sum(self.model.task_cls[:-1])
            for m in range(len(targets)):
                rectified_outputs = torch.cat(outputs, dim=1)
                pred[m] = rectified_outputs[m].argmax()
                if old_classes_number:
                    # if the top-1 class predicted by the network is a new one, rectify the score
                    if int(pred[m]) >= old_classes_number:
                        for o in range(old_classes_number):
                            o_task = int((self.model.task_cls.cumsum(0) <= o).sum())
                            rectified_outputs[m, o] *= (self.init_classes_means[o] / self.current_classes_means[o]) * \
                                                       (self.models_confidence[-1] / self.models_confidence[o_task])
                        pred[m] = rectified_outputs[m].argmax()
                    # otherwise, rectification is not done because an old class is directly predicted
            hits_tag = (pred == targets.to(self.device)).float()
        return hits_taw, hits_tag
    
    def calculate_metrics_per_image(self, taw_img_probs, tag_img_probs,   targets, classes_per_head, outputs, aggregation_type="mean"):
        if self.ft_train:
            hits_taw, hits_tag = super().calculate_metrics_per_image(taw_img_probs, tag_img_probs, targets, classes_per_head, aggregation_type)
        else:
            # TaW Calculation Is The Same
            current_target = targets[0].unsqueeze(0)
            avg_probs = torch.mean(taw_img_probs, dim=0)
    
            head_probs = []
            for probs in torch.split(avg_probs,classes_per_head):
                head_probs.append(probs.unsqueeze(0))
            
            pred = torch.zeros_like(current_target.to(self.device))
  
            for m in range(len(pred)):
                this_task = (self.model.task_cls.cumsum(0) <= current_target[m]).sum() 

                pred[m] = head_probs[this_task][m].argmax() + self.model.task_offset[this_task]
            
            hits_taw = (pred == current_target.to(self.device)).float()
            
            # TaG Rectify the outputs before if the top-1 class predicted by the network is a new one
            # after that compute average
            old_classes_number = sum(self.model.task_cls[:-1])

            pred = torch.zeros_like(targets.to(self.device))
            
            rectified_outputs = outputs

            for m in range(len(targets)):
                pred[m] = rectified_outputs[m].argmax()
                if old_classes_number:
                    # if the top-1 class predicted by the network is a new one, rectify the score
                    if int(pred[m]) >= old_classes_number:
                        for o in range(old_classes_number):
                            o_task = int((self.model.task_cls.cumsum(0) <= o).sum())
                            rectified_outputs[m, o] *= (self.init_classes_means[o] / self.current_classes_means[o]) * \
                                                       (self.models_confidence[-1] / self.models_confidence[o_task])
                # otherwise, rectification is not done because an old class is directly predicted
            rectified_probs =  torch.exp(torch.nn.functional.log_softmax(rectified_outputs, dim=1)) 
            avg_rectified_probs = torch.mean(rectified_probs, dim=0)
            pred = avg_rectified_probs.unsqueeze(0).argmax(1)    
            hits_tag = (pred == current_target.to(self.device)).float()

        return hits_taw, hits_tag
 
 

    def criterion(self, t, outputs, targets):
        """Returns the loss value"""
        if len(self.exemplars_dataset) > 0:
            return torch.nn.functional.cross_entropy(torch.cat(outputs, dim=1), targets)
        return torch.nn.functional.cross_entropy(outputs[t], targets - self.model.task_offset[t])
