import os
import torch
import random
import numpy as np

cudnn_deterministic = True


def seed_everything(seed=0):
    """Fix all random seeds"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = cudnn_deterministic


def print_summary(acc_taw_patch, acc_tag_patch, forg_taw_patch, forg_tag_patch, acc_taw_img, acc_tag_img, forg_taw_img, forg_tag_img):
    """Print summary of results"""
    for name, metric in zip(['TAw Acc Patch', 'TAg Acc Patch', 'TAw Forg Patch', 'TAg Forg Patch', 'TAw Acc Img', 'TAg Acc Img', 'TAw Forg Img', 'TAg Forg Img'], [acc_taw_patch, acc_tag_patch, forg_taw_patch, forg_tag_patch, acc_taw_img, acc_tag_img, forg_taw_img, forg_tag_img]):
        print('*' * 108)
        print(name)
        for i in range(metric.shape[0]):
            print('\t', end='')
            for j in range(metric.shape[1]):
                print('{:5.1f}% '.format(100 * metric[i, j]), end='')
            if np.trace(metric) == 0.0:
                if i > 0:
                    print('\tAvg.:{:5.1f}% '.format(100 * metric[i, :i].mean()), end='')
            else:
                print('\tAvg.:{:5.1f}% '.format(100 * metric[i, :i + 1].mean()), end='')
            print()
    print('*' * 108)


def list_to_device(l, device):
    new_l = []
    for element in l:
        new_l.append(element.to(device))
    return new_l