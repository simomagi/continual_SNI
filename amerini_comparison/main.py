
import torch  
from  dataset import AmeriniDataset, get_data, CustomDataset
from torch.utils.data import DataLoader
from  amerini_network import FusionNet
from  incremental_fusionnet import IncrementalFusionNet
from  train_utils import train_epoch, eval_model, Logger
import time 
import numpy as np 
import random 
import os 
import argparse
import sys 
from pathlib import Path
sys.path.append(str(Path('..').absolute().parent))
from path_variables import *


def seed_everything(seed=0):
    """Fix all random seeds"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True

def parse_args():
    parser = argparse.ArgumentParser(description='Amerini Approach')
    
    # miscellaneous args
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU (default=%(default)s)')
    parser.add_argument('--results_path', type=str, default='../results',
                        help='Results path (default=%(default)s)')
   
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed (default=%(default)s)')
 
    parser.add_argument('--num_workers', default=4, type=int, required=False,
                        help='Number of subprocesses to use for dataloader (default=%(default)s)')
 
    parser.add_argument('--batch_size', default=64, type=int, required=False, 
                        help='Number of samples per batch to load (default=%(default)s)')
    
    parser.add_argument("--net", default="amerini", choices=["amerini", "our_fusion"])
    parser.add_argument("--residual_type", default="amerini", choices=["our", "amerini"])
    parser.add_argument("--model_path",type=str, default=None, help="useful only for testing")
    parser.add_argument("--only_eval", action="store_true")
    
    args = parser.parse_args()
    
    return args
 

if __name__=="__main__":
    dataset_path = FACIL_BOLOGNA_PATH
    args = parse_args()
    if not args.only_eval:
        args.results_path = os.path.expanduser(args.results_path)
        
        if not os.path.exists(args.results_path):
            os.mkdir(args.results_path)
            
        model_path = os.path.join(args.results_path,"models")
        
        if not os.path.exists(model_path):
            os.mkdir(model_path)
        
        seed_everything(args.seed)
        logger = Logger(out_path=args.results_path)
        all_data = get_data(path=dataset_path)
    
        print('=' * 108)
        print('Arguments =')
        for arg in np.sort(list(vars(args).keys())):
            print('\t' + arg + ':', getattr(args, arg))
        print('=' * 108)
        
        if args.residual_type == "amerini":
            trn_dset = AmeriniDataset(all_data['trn'], dataset_path, train=True )
            val_dset = AmeriniDataset(all_data['val'], dataset_path, train=False)
            test_dset = AmeriniDataset(all_data['tst'], dataset_path,  train=False)
        elif args.residual_type == "our":
            trn_dset = CustomDataset(all_data['trn'], dataset_path, train=True)
            val_dset = CustomDataset(all_data['val'], dataset_path, train=False)
            test_dset = CustomDataset(all_data['tst'], dataset_path,  train=False)
            
        trn_loader = DataLoader(trn_dset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        val_loader = DataLoader(val_dset, batch_size=args.batch_size*2, shuffle=False, num_workers=args.num_workers)
        test_loader = DataLoader(test_dset, batch_size=args.batch_size*2, shuffle=False, num_workers=args.num_workers)
        
        
        # Args -- CUDA
        if torch.cuda.is_available():
            torch.cuda.set_device(args.gpu)
            device = 'cuda'

        else:
            print('WARNING: [CUDA unavailable] Using CPU instead!')
            device = 'cpu'
        
        if args.net == "amerini":
            print("Using Amerini Network")
            model = FusionNet()
 
        elif args.net == "our_fusion":
            print("Using Incremental Fusion Net")
            model = IncrementalFusionNet()
        
        model.to(device) 
        
        
        N_epochs = 200
        lr_min = 1e-6
        lr_factor = 10
        lr_patience = 20 
        best_loss = np.inf
        starting_lr = 1e-3
        
        optimizer = torch.optim.Adam(model.parameters(), lr=starting_lr, weight_decay=2e-4)
        
        lr = starting_lr
        
        for i in range(N_epochs):
            """
            Train Epoch
            """
            clock0 = time.time()
            train_epoch(model, optimizer, trn_loader, device)
            clock1 = time.time()
            logger.print_training_time(i,  clock1, clock0)
            
            """
            Eval on Validation Set
            """
            clock3 = time.time()
            valid_loss, patch_valid_acc, img_valid_acc_avg, img_valid_acc_maj =  eval_model(model, val_loader, device)
            clock4 = time.time()
            logger.print_val_accuracy(clock3, clock4, valid_loss, patch_valid_acc, img_valid_acc_avg, img_valid_acc_maj)
            
            if valid_loss < best_loss:
                # if the loss goes down, keep it as the best model and end line with a star ( * )
                best_loss = valid_loss
                best_model = model.get_copy()
                patience = lr_patience
                print(' *', end='')
            else:
                # if the loss does not go down, decrease patience
                patience -= 1
                if patience <= 0:
                    # if it runs out of patience, reduce the learning rate
                    lr /= lr_factor
                    print(' lr={:.1e}'.format(lr), end='')
                    if lr < lr_min:
                        # if the lr decreases below minimum, stop the training session
                        print()
                        break
                    # reset patience and recover best model so far to continue training
                    patience = lr_patience
                    optimizer.param_groups[0]['lr'] = lr
                    model.set_state_dict(best_model)
        
        
        print("\n Evaluating Test and Saving Model")  
        model.set_state_dict(best_model)
        logger.save_model(model_path, best_model)
        test_loss,  test_patch_acc, test_img_acc_avg, test_img_acc_maj =  eval_model(model, test_loader, device)
        logger.print_test_accuracy(test_loss, test_patch_acc, test_img_acc_avg, test_img_acc_maj)
    
    
    else:
        
 
        model_path = os.path.join(args.model_path, "models")
        
        seed_everything(args.seed)
        all_data = get_data(path=dataset_path)
    
        print('=' * 108)
        print('Arguments =')
        for arg in np.sort(list(vars(args).keys())):
            print('\t' + arg + ':', getattr(args, arg))
        print('=' * 108)
        
        if args.residual_type == "amerini":
            test_dset = AmeriniDataset(all_data['tst'], dataset_path,  train=False)
        elif args.residual_type == "our":
            test_dset = CustomDataset(all_data['tst'], dataset_path,  train=False)
            
        test_loader = DataLoader(test_dset, batch_size=512, shuffle=False, num_workers=28)
        
        # Args -- CUDA
        if torch.cuda.is_available():
            torch.cuda.set_device(args.gpu)
            device = 'cuda'

        else:
            print('WARNING: [CUDA unavailable] Using CPU instead!')
            device = 'cpu'
        
        if args.net == "amerini":
            print("Using Fusion Net")
            model = FusionNet()
 
        elif args.net == "our_fusion":
            model = IncrementalFusionNet()
        
        model.to(device) 
        model.set_state_dict(torch.load(os.path.join(model_path, "final_model.ckpt"), map_location=device))
        test_loss,  test_patch_acc, test_img_acc_avg, test_img_acc_maj =  eval_model(model, test_loader, device)
        print("\n Test Loss {:.3f} | Test Patch Accuracy {:5.1f} | Test Avg Img Accuracy {:5.1f}| Test Majority Img Accuracy {:5.1f}".format(test_loss, test_patch_acc*100, 
                                                                                                                                             test_img_acc_avg*100,
                                                                                                                                             test_img_acc_maj*100))
    
        
        
        
        
        