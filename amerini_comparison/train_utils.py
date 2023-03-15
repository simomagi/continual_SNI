
from tqdm import tqdm
import torch 
import os 
from datetime import datetime
import sys 


def calculate_metrics_per_patch( outputs, targets):
    pred = outputs.argmax(1)
    hits  = (pred == targets).float()
    return hits 


def calculate_metrics_per_image(probs,   targets):
    n_images = len(targets)
    targets = targets[0].unsqueeze(0)    # targets is all equal for an image
      

    if n_images > 1:
        avg_probs = torch.mean(probs, dim=0).unsqueeze(0)
    else:
        avg_probs = probs

    avg_pred = avg_probs.argmax(1)
    avg_hits = (avg_pred == targets).float()
    
    patch_preds = probs.argmax(dim=1) 
    majority_pred = torch.mode(patch_preds)[0]
    majority_hits = (majority_pred == targets).float()

    return  avg_hits, majority_hits


def criterion(outputs, targets):
    """Returns the loss value"""
    return torch.nn.functional.cross_entropy(outputs, targets )

    
def train_epoch(model, optimizer, trn_loader, device):
    model.train()
    for res, hist, targets, _ in tqdm(trn_loader):
        # Forward current model
        
        outputs = model(res.to(device), hist.to(device))

        loss = criterion(outputs, targets.to(device))  
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
 
        
def eval_model(model, tst_loader, device):
    
      with torch.no_grad():
        total_loss_patch, total_acc_patch, total_num_patch = 0, 0, 0
        total_acc_img_avg, total_acc_img_majority = 0, 0
        overall_probs, overall_targets, overall_indices = [], [], []
        
        model.eval()
        
        for res, hist, targets, ids in tqdm(tst_loader):
            res = res.to(device)
            hist = hist.to(device)
            targets = targets.to(device)
            outputs = model(res , hist)
            
            loss = criterion(outputs, targets)  
            hits_patch = calculate_metrics_per_patch(outputs, targets)
            total_loss_patch += loss.item() * len(targets)
            total_acc_patch += hits_patch.sum().item()
            total_num_patch += len(targets)
            
            probs = torch.exp(torch.nn.functional.log_softmax(outputs, dim=1))  
          
            overall_probs.append(probs)
            overall_targets.append(targets)
            overall_indices.append(ids)
                    
        overall_indices = torch.cat(overall_indices)
        overall_targets = torch.cat(overall_targets)
        overall_probs = torch.cat(overall_probs, dim=0)
        set_ids = torch.unique(overall_indices)
        total_num_imgs = len(set_ids)
        
        for _id in set_ids:
            image_index = (overall_indices == _id)
            img_target = overall_targets[image_index]
            img_probs = overall_probs[image_index, :]
            avg_hits_img, majority_hits_img = calculate_metrics_per_image(img_probs, img_target)
            total_acc_img_avg +=  avg_hits_img.sum().item()
            total_acc_img_majority +=  majority_hits_img.sum().item()
    
        return total_loss_patch / total_num_patch, total_acc_patch / total_num_patch, total_acc_img_avg/total_num_imgs, total_acc_img_majority/total_num_imgs



class Logger():
    def __init__(self, out_path, begin_time=None) -> None:
 

        self.out_path = os.path.join(out_path, "logger")
        
        if begin_time is None:
            self.begin_time = datetime.now()
        else:
            self.begin_time = begin_time
        
        self.begin_time_str = self.begin_time.strftime("%Y-%m-%d-%H-%M")
        sys.stdout = FileOutputDuplicator(sys.stdout,
                                          os.path.join(out_path, 'stdout-{}.txt'.format(self.begin_time_str)), 'w')
        sys.stderr = FileOutputDuplicator(sys.stderr,
                                          os.path.join(out_path, 'stderr-{}.txt'.format(self.begin_time_str)), 'w')
    
    def print_training_time(self, i,  clock1, clock0):
        print('\n | Epoch {:3d}, time={:5.1f}s | Train: skip eval |'.format(i + 1, clock1 - clock0), end='')
    
    def print_val_accuracy(self, clock3, clock4, valid_loss, patch_valid_acc, img_valid_acc_avg, img_valid_acc_maj):
        print('\n Valid: time={:5.1f} loss={:.3f}, Patch acc={:5.1f}% | Avg Img acc={:5.1f} | Majority Img acc={:5.1f} '.format(
               clock4 - clock3, valid_loss, 100 * patch_valid_acc , 100 * img_valid_acc_avg, 100 * img_valid_acc_maj), end='')
        
    def print_test_accuracy(self, test_loss, test_patch_acc,  test_img_acc_avg, test_img_acc_maj):
        print("\n Test Loss {:.3f} | Test Patch Accuracy {:5.1f} | Test Avg Img Accuracy {:5.1f}| Test Majority Img Accuracy {:5.1f}".format(test_loss, test_patch_acc*100, test_img_acc_avg*100,
                                                                                                                                            test_img_acc_maj*100))
    
    def save_model(self, model_path, state_dict):
        torch.save(state_dict, os.path.join(model_path, "final_model.ckpt"))
        
        

class FileOutputDuplicator(object):
    def __init__(self, duplicate, fname, mode):
        self.file = open(fname, mode)
        self.duplicate = duplicate

    def __del__(self):
        self.file.close()

    def write(self, data):
        self.file.write(data)
        self.duplicate.write(data)

    def flush(self):
        self.file.flush()