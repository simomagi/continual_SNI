from os.path import join
import sys 
from pathlib import Path
sys.path.append(str(Path('..').absolute().parent))

from path_variables import *

 


dataset_config = {
    'bologna': {
        'path': FACIL_BOLOGNA_PATH,
        'flip': True,
    },
    
    ## add here other dataset with the same structure with Smart Data Bologna
}


# Add missing keys:
for dset in dataset_config.keys():
    for k in ['normalize','class_order']:
        if k not in dataset_config[dset].keys():
            dataset_config[dset][k] = None
    if 'flip' not in dataset_config[dset].keys():
        dataset_config[dset]['flip'] = False
