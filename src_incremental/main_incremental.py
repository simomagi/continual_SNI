from ast import parse
import os
import time
import torch
import argparse
import importlib
import numpy as np
from functools import reduce
import utils
import approach
from loggers.exp_logger import MultiLogger
from datasets.data_loader import get_loaders
from datasets.dataset_config import dataset_config
from networks.incremental_fusionnet import FusionFc
from networks.resnet18 import Resnet18
import sys 
from pathlib import Path
sys.path.append(str(Path('..').absolute().parent))

from path_variables import *
from datasets.dataset_config import dataset_config

def main(argv=None):
    tstart = time.time()
    # Arguments
    parser = argparse.ArgumentParser(description='FACIL - Framework for Analysis of Class Incremental Learning')

    # miscellaneous args
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU (default=%(default)s)')
    parser.add_argument('--results-path', type=str, default='../results',
                        help='Results path (default=%(default)s)')
    parser.add_argument('--exp-name', default=None, type=str,
                        help='Experiment name (default=%(default)s)')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed (default=%(default)s)')
    parser.add_argument('--log', default=['disk'], type=str, choices=['disk', 'tensorboard'],
                        help='Loggers used (disk, tensorboard) (default=%(default)s)', nargs='*', metavar="LOGGER")
    parser.add_argument('--save-models', action='store_true',
                        help='Save trained models (default=%(default)s)')
 
    parser.add_argument('--no-cudnn-deterministic', action='store_true',
                        help='Disable CUDNN deterministic (default=%(default)s)')
    # dataset args
    parser.add_argument('--datasets', default=['bologna'], type=str, choices=list(dataset_config.keys()),
                        help='Dataset or datasets used (default=%(default)s)', nargs='+', metavar="DATASET")
    
    parser.add_argument('--num-workers', default=4, type=int, required=False,
                        help='Number of subprocesses to use for dataloader (default=%(default)s)')
    parser.add_argument('--pin-memory', default=False, type=bool, required=False,
                        help='Copy Tensors into CUDA pinned memory before returning them (default=%(default)s)')
    parser.add_argument('--batch-size', default=64, type=int, required=False, 
                        help='Number of samples per batch to load (default=%(default)s)')
    parser.add_argument('--num-tasks', default=4, type=int, required=False,
                        help='Number of tasks per dataset (default=%(default)s)')
    parser.add_argument('--nc-first-task', default=None, type=int, required=False,
                        help='Number of classes of the first task (default=%(default)s)')
    parser.add_argument('--use-valid-only', action='store_true',
                        help='Use validation split instead of test (default=%(default)s)')
    parser.add_argument('--stop-at-task', default=0, type=int, required=False,
                        help='Stop training after specified task (default=%(default)s)')

    parser.add_argument('--keep-existing-head', action='store_true',
                        help='Disable removing classifier last layer (default=%(default)s)')
    parser.add_argument('--pretrained', action='store_true',
                        help='Use pretrained backbone (default=%(default)s)')
    # training args
    parser.add_argument('--approach', default='finetuning', type=str, choices=approach.__all__,
                        help='Learning approach used (default=%(default)s)', metavar="APPROACH")
    parser.add_argument('--nepochs', default=200, type=int, required=False, 
                        help='Number of epochs per training session (default=%(default)s)') # 00

    parser.add_argument('--clipping', default=10000, type=float, required=False,
                        help='Clip gradient norm (default=%(default)s)')

    # optimizer config
    parser.add_argument("--optimizer_type", default="adam", type=str, choices=["adam", "sgd"])
    
    parser.add_argument('--lr', default=None, type=float, required=False, 
                        help='Starting learning rate (default=%(default)s)')
    parser.add_argument('--lr-min', default=None, type=float, required=False,
                        help='Minimum learning rate (default=%(default)s)')
    parser.add_argument('--lr-factor', default=None, type=float, required=False,
                        help='Learning rate decreasing factor (default=%(default)s)')
    parser.add_argument('--lr-patience', default=None, type=int, required=False,
                        help='Maximum patience to wait before decreasing learning rate (default=%(default)s)')

    parser.add_argument('--momentum', default=None, type=float, required=False,
                        help='Momentum factor (default=%(default)s)')
    parser.add_argument('--weight-decay', default=None, type=float, required=False,
                        help='Weight decay (L2 penalty) (default=%(default)s)')
    # end optimizer config 
  
    # social detection args

    parser.add_argument("--input", type=str, choices=["rgb", "gray", "residual", "residual_hist"], default="residual")
    
    # model args
    parser.add_argument('--network', default='resnet18', type=str, choices=["resnet18", 
                                                                            "fusion_fc"],
                        help='Network architecture used (default=%(default)s)', metavar="NETWORK")
 
 
    # Args -- Incremental Learning Framework
    args, extra_args = parser.parse_known_args(argv)
    

    # setting manual hyperparameter optimizer 
    if  args.optimizer_type == "adam":
        print("SETTING default Adam Optimizer parameters")
        args.lr = 1e-3
        args.lr_patience = 20
        args.lr_min = 1e-5 
        args.lr_factor = 10
        args.weight_decay = 2e-4

    elif args.optimizer_type == "sgd":
        print("SETTING default SGD Optimizer parameters")
        args.lr = 5e-2
        args.momentum = 0.9
        args.lr_min = 1e-4
        args.lr_patience = 20 
        args.lr_factor = 3 
        args.weight_decay = 2e-4
 
    args.results_path = os.path.expanduser(args.results_path)
    base_kwargs = dict(nepochs=args.nepochs, optimizer_type=args.optimizer_type, lr=args.lr, lr_min=args.lr_min, lr_factor=args.lr_factor,
                       lr_patience=args.lr_patience, clipgrad=args.clipping, momentum=args.momentum,
                       wd=args.weight_decay)

    if args.no_cudnn_deterministic:
        print('WARNING: CUDNN Deterministic will be disabled.')
        utils.cudnn_deterministic = False

    utils.seed_everything(seed=args.seed)
    print('=' * 108)
    print('Arguments =')
    for arg in np.sort(list(vars(args).keys())):
        print('\t' + arg + ':', getattr(args, arg))
    print('=' * 108)

 
    # Args -- CUDA
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        device = 'cuda'

    else:
        print('WARNING: [CUDA unavailable] Using CPU instead!')
        device = 'cpu'
 
    # Args -- Network
    from networks.network import LLL_Net 

    if args.network == "fusion_fc":
        print("Using Fusion Net")
        if args.input != "residual_hist":
            sys.exit("Fusion Net cannot be used without residual hist")
            
        init_model = FusionFc(data_type=args.input)
    elif args.network  == "resnet18":
        print("Using Resnet18")
        if args.input == "residual_hist":
            sys.exit("Error Resnet 18 cannot be used with residual hist")
        init_model =  Resnet18(data_type=args.input)
        
    # Args -- Continual Learning Approach
    from approach.incremental_learning import Inc_Learning_Appr
    Appr = getattr(importlib.import_module(name='approach.' + args.approach), 'Appr')
    assert issubclass(Appr, Inc_Learning_Appr)
    appr_args, extra_args = Appr.extra_parser(extra_args)
    print('Approach arguments =')
    for arg in np.sort(list(vars(appr_args).keys())):
        print('\t' + arg + ':', getattr(appr_args, arg))
    print('=' * 108)

    # Args -- Exemplars Management
    from datasets.exemplars_dataset import ExemplarsDataset
    Appr_ExemplarsDataset = Appr.exemplars_dataset_class()
    if Appr_ExemplarsDataset:
        assert issubclass(Appr_ExemplarsDataset, ExemplarsDataset)
        appr_exemplars_dataset_args, extra_args = Appr_ExemplarsDataset.extra_parser(extra_args)
        print('Exemplars dataset arguments =')
        for arg in np.sort(list(vars(appr_exemplars_dataset_args).keys())):
            print('\t' + arg + ':', getattr(appr_exemplars_dataset_args, arg))
        print('=' * 108)
    else:
        appr_exemplars_dataset_args = argparse.Namespace()

 
    assert len(extra_args) == 0, "Unused args: {}".format(' '.join(extra_args))
    ####################################################################################################################

    # Log all arguments
    full_exp_name = reduce((lambda x, y: x[0] + y[0]), args.datasets) if len(args.datasets) > 0 else args.datasets[0]
    full_exp_name += '_' + args.approach
    if args.exp_name is not None:
        full_exp_name += '_' + args.exp_name
    logger = MultiLogger(args.results_path, full_exp_name, loggers=args.log, save_models=args.save_models)
    logger.log_args(argparse.Namespace(**args.__dict__, **appr_args.__dict__, **appr_exemplars_dataset_args.__dict__))

    # Loaders
    utils.seed_everything(seed=args.seed)
    trn_loader, val_loader, tst_loader, trn_crops_load, taskcla = get_loaders(args.datasets, args.num_tasks, args.nc_first_task,
                                                              args.batch_size, num_workers=args.num_workers,
                                                              pin_memory=args.pin_memory,
                                                              dataset_type=args.input)
    # Apply arguments for loaders
    if args.use_valid_only:
        tst_loader = val_loader
        
    max_task = len(taskcla) if args.stop_at_task == 0 else args.stop_at_task

    # Network and Approach instances
    utils.seed_everything(seed=args.seed)
 
    net = LLL_Net(init_model, remove_existing_head=not args.keep_existing_head)
 
    print("Total Number Of Parameters {}:".format(sum(p.numel() for p in net.parameters() if p.requires_grad)))

 

    utils.seed_everything(seed=args.seed)
    # taking transformations and class indices from first train dataset
    first_train_ds = trn_loader[0].dataset
    transform, class_indices = first_train_ds.transform, first_train_ds.class_indices
 
    appr_kwargs = {**base_kwargs, **dict(logger=logger, **appr_args.__dict__)}
    if Appr_ExemplarsDataset:
        appr_kwargs['exemplars_dataset'] = Appr_ExemplarsDataset(args.input, dataset_config[args.datasets[0]]["path"], 
                                                                  class_indices,
                                                                 **appr_exemplars_dataset_args.__dict__)
    utils.seed_everything(seed=args.seed)
    appr = Appr(net, device,  **appr_kwargs)


    # Loop tasks
    print(taskcla)
    acc_taw_patch = np.zeros((max_task, max_task))
    acc_tag_patch  = np.zeros((max_task, max_task))
    acc_taw_img =  np.zeros((max_task, max_task))
    acc_tag_img  = np.zeros((max_task, max_task))

    forg_taw_patch = np.zeros((max_task, max_task))
    forg_tag_patch = np.zeros((max_task, max_task))
    
    forg_taw_img = np.zeros((max_task, max_task))
    forg_tag_img = np.zeros((max_task, max_task))
    
    for t, (_, ncla) in enumerate(taskcla):
        # Early stop tasks if flag
        if t >= max_task:
            continue

        print('*' * 108)
        print('Task {:2d}'.format(t))
        print('*' * 108)

 
        net.add_head(taskcla[t][1])
        net.to(device)

 

        # Train
        appr.train(t, trn_loader[t], trn_crops_load[t],  val_loader[t])
        print('-' * 108)

        # Test
        for u in range(t + 1):
            test_loss,   acc_taw_patch[t, u],  acc_tag_patch[t, u], acc_taw_img[t, u],  acc_tag_img[t, u] = appr.eval(u, tst_loader[u])
            # test_loss,   acc_taw_patch[t, u],  acc_tag_patch[t, u], acc_taw_img[t, u],  acc_tag_img[t, u]  = 10 ,10 ,10 ,10, 10
            if u < t:
                forg_taw_patch[t, u] = acc_taw_patch[:t, u].max(0) - acc_taw_patch[t, u]
                forg_tag_patch[t, u] = acc_tag_patch[:t, u].max(0) - acc_tag_patch[t, u]
                forg_taw_img[t, u] = acc_taw_img[:t, u].max(0) - acc_taw_img[t, u]
                forg_tag_img[t, u] = acc_tag_img[:t, u].max(0) - acc_tag_img[t, u]

            print('>>> Test on task {:2d} : loss={:.3f} | Patch TAw acc={:5.1f}%, forg={:5.1f}%'
                  '| Patch TAg acc={:5.1f}%, forg={:5.1f}% |'
                  'Img TAw acc={:5.1f}%, forg={:5.1f}% |'
                  'Img TAg acc={:5.1f}%, forg={:5.1f}% =<<<'.format(u, test_loss,
                                                                 100 * acc_taw_patch[t, u], 100 * forg_taw_patch[t, u],
                                                                 100 * acc_tag_patch[t, u], 100 * forg_tag_patch[t, u],
                                                                100 * acc_taw_img[t, u], 100 * forg_taw_img[t, u],
                                                                 100 * acc_tag_img[t, u], 100 * forg_tag_img[t, u]
                                                                 ))

            logger.log_scalar(task=t, iter=u, name='loss', group='test', value=test_loss)
            logger.log_scalar(task=t, iter=u, name='acc_taw_patch', group='test', value=100 * acc_taw_patch[t, u])
            logger.log_scalar(task=t, iter=u, name='acc_tag_patch', group='test', value=100 * acc_tag_patch[t, u])
            logger.log_scalar(task=t, iter=u, name='forg_taw_patch', group='test', value=100 * forg_taw_patch[t, u])
            logger.log_scalar(task=t, iter=u, name='forg_tag_patch', group='test', value=100 * forg_tag_patch[t, u])

            logger.log_scalar(task=t, iter=u, name='acc_taw_img', group='test', value=100 * acc_taw_img[t, u])
            logger.log_scalar(task=t, iter=u, name='acc_tag_img', group='test', value=100 * acc_tag_img[t, u])
            logger.log_scalar(task=t, iter=u, name='forg_taw_img', group='test', value=100 * forg_taw_img[t, u])
            logger.log_scalar(task=t, iter=u, name='forg_tag_img', group='test', value=100 * forg_tag_img[t, u])

        # Save
        print('Save at ' + os.path.join(args.results_path, full_exp_name))
        logger.log_result(acc_taw_patch, name="acc_taw_patch", step=t)
        logger.log_result(acc_tag_patch, name="acc_tag_patch", step=t)
        logger.log_result(forg_taw_patch, name="forg_taw_patch", step=t)
        logger.log_result(forg_tag_patch, name="forg_tag_patch", step=t)
        logger.save_model(net.state_dict(), task=t)
        logger.log_result(acc_taw_patch.sum(1) / np.tril(np.ones(acc_taw_patch.shape[0])).sum(1), name="avg_accs_taw_patch", step=t)
        logger.log_result(acc_tag_patch.sum(1) / np.tril(np.ones(acc_tag_patch.shape[0])).sum(1), name="avg_accs_tag_patch", step=t)
        aux = np.tril(np.repeat([[tdata[1] for tdata in taskcla[:max_task]]], max_task, axis=0))
        logger.log_result((acc_taw_patch * aux).sum(1) / aux.sum(1), name="wavg_accs_taw_patch", step=t)
        logger.log_result((acc_tag_patch * aux).sum(1) / aux.sum(1), name="wavg_accs_tag_patch", step=t)

        logger.log_result(acc_taw_img, name="acc_taw_img", step=t)
        logger.log_result(acc_tag_img, name="acc_tag_img", step=t)
        logger.log_result(forg_taw_img, name="forg_taw_img", step=t)
        logger.log_result(forg_tag_img, name="forg_tag_img", step=t)
 
        logger.log_result(acc_taw_img.sum(1) / np.tril(np.ones(acc_taw_img.shape[0])).sum(1), name="avg_accs_taw_img", step=t)
        logger.log_result(acc_tag_img.sum(1) / np.tril(np.ones(acc_tag_img.shape[0])).sum(1), name="avg_accs_tag_img", step=t)
 
        logger.log_result((acc_taw_img * aux).sum(1) / aux.sum(1), name="wavg_accs_taw_img", step=t)
        logger.log_result((acc_tag_img * aux).sum(1) / aux.sum(1), name="wavg_accs_tag_img", step=t)

    
    # Print Summary
    utils.print_summary(acc_taw_patch, acc_tag_patch, forg_taw_patch, forg_tag_patch, acc_taw_img, acc_tag_img, forg_taw_img, forg_tag_img)
    print('[Elapsed time = {:.1f} h]'.format((time.time() - tstart) / (60 * 60)))
    print('Done!')

    return acc_taw_patch, acc_tag_patch, forg_taw_patch, forg_tag_patch, acc_taw_img, acc_tag_img, forg_taw_img, forg_tag_img, logger.exp_path
    ####################################################################################################################


if __name__ == '__main__':
    main()
