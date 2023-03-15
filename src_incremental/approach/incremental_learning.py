import time
import torch
import numpy as np
from argparse import ArgumentParser
from loggers.exp_logger import ExperimentLogger
from datasets.exemplars_dataset import ExemplarsDataset
import numpy as np

from utils import list_to_device

class Inc_Learning_Appr:
    """Basic class for implementing incremental learning approaches"""

    def __init__(self, model, device, nepochs=100, optimizer_type="adam", lr=0.05, lr_min=1e-4, lr_factor=3, lr_patience=5, clipgrad=10000,
                 momentum=0, wd=0,  logger: ExperimentLogger = None, exemplars_dataset: ExemplarsDataset = None):
        self.model = model
        self.device = device
        self.nepochs = nepochs
        self.lr = lr
        self.lr_min = lr_min
        self.lr_factor = lr_factor
        self.lr_patience = lr_patience
        self.clipgrad = clipgrad
        self.momentum = momentum
        self.wd = wd
        self.logger = logger
        self.exemplars_dataset = exemplars_dataset
        self.optimizer_type = optimizer_type
        self.optimizer = None
 

    @staticmethod
    def extra_parser(args):
        """Returns a parser containing the approach specific parameters"""
        parser = ArgumentParser()
        return parser.parse_known_args(args)

    @staticmethod
    def exemplars_dataset_class():
        """Returns a exemplar dataset to use during the training if the approach needs it
        :return: ExemplarDataset class or None
        """
        return None

    def _get_optimizer(self):
        """Returns the optimizer"""
        if self.optimizer_type == "adam":
            return torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.wd)
        else:
            return torch.optim.SGD(self.model.parameters(), lr=self.lr, weight_decay=self.wd, momentum=self.momentum)

    def train(self, t, trn_loader, trn_loader_crops, val_loader):
        """Main train structure"""
        self.pre_train_process(t, trn_loader, trn_loader_crops)
        self.train_loop(t, trn_loader, trn_loader_crops, val_loader)
        self.post_train_process(t, trn_loader, trn_loader_crops)

    def pre_train_process(self, t, trn_loader, trn_loader_crops):
        pass 
    

    def train_loop(self, t, trn_loader, val_loader):
        """Contains the epochs loop"""
        lr = self.lr
        best_loss = np.inf
        patience = self.lr_patience
    
        best_model = self.model.get_copy()
        self.optimizer = self._get_optimizer()

        # Loop epochs
        for e in range(self.nepochs):
            # Train
            clock0 = time.time()
            self.train_epoch(t, trn_loader)
            clock1 = time.time()
        
            print('| Epoch {:3d}, time={:5.1f}s | Train: skip eval |'.format(e + 1, clock1 - clock0), end='')

            # Valid
            clock3 = time.time()
            valid_loss, patch_valid_acc_taw, patch_valid_acc_tag, img_valid_acc_taw, img_valid_acc_tag = self.eval(t, val_loader)
            # valid_loss, patch_valid_acc_taw, patch_valid_acc_tag, img_valid_acc_taw, img_valid_acc_tag = 10, 10, 10, 10, 10 
            clock4 = time.time()
            print(' Valid: time={:5.1f}s loss={:.3f}, Patch TAw acc={:5.1f}% |'
                  ' Patch TAg acc={:5.1f}, Img TAw acc={:5.1f}, Img TAg acc={:5.1f}, '.format(
                clock4 - clock3, valid_loss, 100 * patch_valid_acc_taw, 100 * patch_valid_acc_tag, 
                 100 * img_valid_acc_taw, 100 * img_valid_acc_tag
                ), end='')

            self.logger.log_scalar(task=t, iter=e + 1, name="loss", value=valid_loss, group="valid")
            self.logger.log_scalar(task=t, iter=e + 1, name="patch_acc", value=100 * patch_valid_acc_taw, group="valid")
            self.logger.log_scalar(task=t, iter=e + 1, name="img_acc", value=100 * img_valid_acc_taw, group="valid")
            # Adapt learning rate - patience scheme - early stopping regularization
            if valid_loss < best_loss:
                # if the loss goes down, keep it as the best model and end line with a star ( * )
                best_loss = valid_loss
 
                best_model = self.model.get_copy()
                patience = self.lr_patience
                print(' *', end='')
            else:
                # if the loss does not go down, decrease patience
                patience -= 1
                if patience <= 0:
                    # if it runs out of patience, reduce the learning rate
                    lr /= self.lr_factor
                    print(' lr={:.1e}'.format(lr), end='')
                    if lr < self.lr_min:
                        # if the lr decreases below minimum, stop the training session
                        print()
                        break
                    # reset patience and recover best model so far to continue training
                    patience = self.lr_patience
                    self.optimizer.param_groups[0]['lr'] = lr
                    self.model.set_state_dict(best_model)
            self.logger.log_scalar(task=t, iter=e + 1, name="patience", value=patience, group="train")
            self.logger.log_scalar(task=t, iter=e + 1, name="lr", value=lr, group="train")
            print()
        self.model.set_state_dict(best_model)

    def post_train_process(self, t, trn_loader, trn_loader_crops):
        """Runs after training all the epochs of the task (after the train session)"""
        pass

    def train_epoch(self, t, trn_loader):
        """Runs a single epoch"""
        self.model.train()
 
        from tqdm import tqdm
        for x, targets, _  in tqdm(trn_loader):
            # Forward current model
            x = list_to_device(x, self.device)

            outputs = self.model(x)
            loss = self.criterion(t, outputs, targets.to(self.device)) 
            
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clipgrad)
            self.optimizer.step()
       


    def eval(self, t, val_loader):
        """Contains the evaluation code"""
        with torch.no_grad():
            total_loss_patch, total_acc_taw_patch, total_acc_tag_patch, total_num_patch = 0, 0, 0, 0
            
            total_acc_taw_img, total_acc_tag_img  = 0, 0
            overall_taw_probs, overall_tag_probs = [], []

            self.model.eval()
          
            overall_targets = []
            overall_indices = []
            from tqdm import tqdm 
          

            for x, targets, ids  in tqdm(val_loader):
                
                x = list_to_device(x, self.device)

                outputs = self.model(x)
     
                loss = self.criterion(t, outputs, targets.to(self.device)) #, weights=w
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
                 
            # get last output to get classes per head 
            classes_per_head = [o.shape[1] for o in outputs]

            overall_indices = torch.cat(overall_indices)
            overall_targets = torch.cat(overall_targets)
            # overall_probs = torch.cat(overall_probs, dim=0)
            overall_taw_probs = torch.cat(overall_taw_probs, dim=0)
            overall_tag_probs = torch.cat(overall_tag_probs, dim=0)

            set_ids = torch.unique(overall_indices)
            total_num_imgs = len(set_ids)
            for _id in set_ids:
                image_index = (overall_indices == _id)
                img_target = overall_targets[image_index]
                taw_img_probs = overall_taw_probs[image_index, :]
                tag_img_probs = overall_tag_probs[image_index, :]

                hits_taw_img, hits_tag_img = self.calculate_metrics_per_image(taw_img_probs, tag_img_probs, 
                                                                              img_target, classes_per_head)
                total_acc_taw_img += hits_taw_img.sum().item()
                total_acc_tag_img += hits_tag_img.sum().item()



            return total_loss_patch / total_num_patch, total_acc_taw_patch / total_num_patch, \
                   total_acc_tag_patch / total_num_patch, total_acc_taw_img/ total_num_imgs, total_acc_tag_img/total_num_imgs
    
    
    def calculate_metrics_per_image(self, taw_probs, tag_probs, targets, classes_per_head, aggregation_type="mean"):
        # aggregation_type: max, mean, median 

        n_images = len(targets)
        targets = targets[0].unsqueeze(0)    # targets is all equal for an image

        if aggregation_type == "mean":

            # TaW Evaluation

            avg_taw_probs = torch.mean(taw_probs, dim=0)
    
            head_probs = []
            for probs in torch.split(avg_taw_probs,classes_per_head):
                head_probs.append(probs.unsqueeze(0))
            
            pred = torch.zeros_like(targets.to(self.device))
  
            for m in range(len(pred)):
                this_task = (self.model.task_cls.cumsum(0) <= targets[m]).sum() 
                pred[m] = head_probs[this_task][m].argmax() + self.model.task_offset[this_task]
            
            hits_taw = (pred == targets.to(self.device)).float()

            # TaG Evaluation

            if n_images > 1:
                avg_tag_probs = torch.mean(tag_probs, dim=0).unsqueeze(0)
            else:
                avg_tag_probs = tag_probs

            pred = avg_tag_probs.argmax(1)
            
            hits_tag = (pred == targets.to(self.device)).float()

        elif aggregation_type == "majority_vote":
            pred = torch.zeros_like(targets.to(self.device))
            # Task-Aware Multi-Head

            head_probs = []
            for i in range(probs.shape[0]):
                head_probs.append(probs[i].unsqueeze(0))

            for m in range(len(pred)):
                this_task = (self.model.task_cls.cumsum(0) <= targets[m]).sum() 
                pred[m] = torch.mode(head_probs[this_task][m].argmax(dim=1))[0] + self.model.task_offset[this_task]
            

            hits_taw = (pred == targets.to(self.device)).float()
            pred = torch.mode(torch.cat(head_probs, dim=1).argmax(dim=2))[0] 
            hits_tag = (pred == targets.to(self.device)).float()

        return hits_taw, hits_tag



    def calculate_metrics_per_patch(self, outputs, targets):
        """Contains the main Task-Aware and Task-Agnostic metrics"""
        pred = torch.zeros_like(targets.to(self.device))
        # Task-Aware Multi-Head
        for m in range(len(pred)):
 
            this_task = (self.model.task_cls.cumsum(0) <= targets[m]).sum() 
            pred[m] = outputs[this_task][m].argmax() + self.model.task_offset[this_task]

        hits_taw = (pred == targets.to(self.device)).float()
        # Task-Agnostic Multi-Head
        
        pred = torch.cat(outputs, dim=1).argmax(1)

        hits_tag = (pred == targets.to(self.device)).float()
        return hits_taw, hits_tag

    def criterion(self, t, outputs, targets):
        """Returns the loss value"""
        return torch.nn.functional.cross_entropy(outputs[t], targets - self.model.task_offset[t])
    
 