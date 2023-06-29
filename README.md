# Towards Continual Social Network Identification

## Abstract
Social networks have become most widely used channels for sharing images and videos, and discovering the social platform of origin of multimedia content is of great interest to the forensics community. Several techniques address this problem, however the rapid development of new social platforms, and the deployment of updates to existing ones, often render forensic tools obsolete shortly after their introduction. This effectively requires constant updating of methods and models, which is especially cumbersome when dealing with techniques based on neural networks, as trained models cannot be easily fine-tuned to handle new classes without drastically reducing the performance on the old ones -- a phenomenon known as *catastrophic forgetting*. Updating a model thus often entails retraining the network from scratch on all available data, including that used for training previous versions of the model. Continual learning refers to techniques specifically designed to mitigate catastrophic forgetting, hus making it possible to extend an existing model requiring no or a limited number of examples from the original dataset.
In this paper, we investigate the potential of continual learning techniques to build an extensible social network identification neural network.
We introduce a simple yet effective neural network architecture for Social Network Identification (SNI) and perform extensive experimental validation of continual learning approaches on it. Our results demonstrate that, although Continual SNI remains a challenging problem, catastrophic forgetting can be significantly reduced by only retaining a fraction of the original training data.

## Authors
 - Simone Magistri, simone.magistri@unifi.it
 - Daniele Baracchi, daniele.baracchi@unifi.it
 - Dasara Shullani, dasara.shullani@unifi.it
 - Andrew D. Bagdanov, andrew.bagdanov@unifi.it
 - Alessandro Piva, alessandro.piva@unifi.it

## Download Smart Data Bologna Dataset

Please download **Smartphone images** from  http://smartdata.cs.unibo.it/datasets


## Conda environment

1) Set the anaconda prefix in the file environment.yml (e.g. prefix:/home/user/miniconda3/envs/sn_facil) and the env name (e.g name: sn_facil)

2) Install anaconda environment

  ```
 conda env create -f environment.yml
  ```
2) Activate environment
  ```
  conda activate sn_facil
  ```
3) Install some extra packages 
  ```
  pip install jpegio==0.2.8
  conda install -c conda-forge pywavelets
  pip install opencv-python
  ```
## Set data path
In  `path_variables.py`, set the paths where the Smart Data Bologna (SDB) Dataset is stored, and where the preprocessed dataset will be saved.
```
BOLOGNA_PATH =  "/images/images/forensic_datasets/Container_Datasets/SmartDataBologna
FACIL_BOLOGNA_PATH = "/Scratch/incremental_dataset/IncrementalSmartDataBologna" 
```
```
mkdir /Scratch/incremental_dataset
mkdir /Scratch/incremental_dataset/IncrementalSmartDataBologna
```

## Dataset-Structure
This is the overall dataset structure after the pre-processing steps.
```
incremental_dataset
└─ IncrementalSmartDataBologna
    ├── Images # Original Smart Data Bologna Images
    ├── Info   # Information on the dataset
    │   ├── train.txt                 # Image Train Split
    │   ├── val.txt                   # Image Val Split    
    │   ├── test.txt                  # Image Test Split
    │   ├── train_crops.txt           # Patch Train Split
    |   ├── val_crops.txt             # Patch Val Split     
    │   ├── test_crops.txt            # Patch Test Split       
    │   ├── train_crop_count.pickle   # Train Dictionary {Img_name:N_crops} 
    │   ├── val_crop_count.pickle     # Val Dictionary {Img_name:N_crops}
    |   ├── test_crop_count.pickle    # Test Dictionary {Img_name:N_crops}
    |   ├── discarded_images.pkl      # Images discarded from the preprocessing
    |   └── summary.csv               # statistics on the dataset
    |
    ├── Patches_256 # Image Patches 256 x 256 
    |      |──${FolderDevice}_${SocialNetwork}_${CameraType}_${OriginalImageName}_crops
    |      |                   |
    |     ...             ${FolderDevice}_${SocialNetwork}_${CameraType}_${OriginalImageName}_${CropID}.png
    |                          |
    |                         ...
    ├── Histograms # Dct Histogram Computed Over Patches 256 x 256
    |      |──${FolderDevice}_${SocialNetwork}_${CameraType}_${OriginalImageName}_crops
    |      |                   |
    |     ...                 ${FolderDevice}_${SocialNetwork}_${CameraType}_${OriginalImageName}_${CropID}_hist.npy
    |                          |
    |                         ...
    └── PrnuResidual # Prnu Residual computed over patches 256 x 256 
           |──${FolderDevice}_${SocialNetwork}_${CameraType}_${OriginalImageName}_crops
           |                   |
          ...               ${FolderDevice}_${SocialNetwork}_${CameraType}_${OriginalImageName}_${CropID}_res.npy
                               |
                              ...     

```

Let 
```
Info=$FACIL_BOLOGNA_PATH/Info/,  
Images=$FACIL_BOLOGNA_PATH/Images
Patches_256=$FACIL_BOLOGNA_PATH/Patches_256/,  
Histograms=$FACIL_BOLOGNA_PATH/Histograms/, 
PrnuResidual=$FACIL_BOLOGNA_PATH/PrnuResidual/. 
```

The structure of directories `Patches_256`, `Histograms`, `PrnuResidual` is the same. They contain  one directory per image and the name of each directory is a pointer to the original image in the SDB dataset. A single directory contains the set of patch images for the full image (`Patches_256`), the set of patch histogram for the full Image (`Histograms`) and the set of patch prnu residuals for the full image (`PrnuResidual`). The nomenclature of the folder is taken from the file `summary.csv`. See Pre-process Data for more details.

## Pre-Process Data
The following operations should be performed sequentially.

1) Generate SDB summary

```
python data_preprocessing/get_dataset_summary_1.py 
```

This python script copy the images from `$BOLOGNA_PATH`  to $Images  and generate a file  `$Info/summary.csv` with the information about the dataset:

- **FolderDevice**: Identifier of the device which captured the images in the SmartDataBolognaDataset. These devices are used to compute the split between train,val, test.
- **SocialNetwork**: Social Network Class
- **Downloaded**: If the Social Network is downloaded. Some Social Network contains only the original images and they will be discarded.
- **SocialNetworkID**: Integer identifier of the social network  class
- **CameraType**: Camera type which captured the image (ofront, rear, ...)
- **OriginalImagePath**: Original path of the image in `$BOLOGNA_PATH`
- **NewImageName**: New name of the image after preprocessing. It contains some information about the images. The new image name  is the following: 

```
${FolderDevice}_${SocialNetwork}_${CameraType}_${OriginalImageName} # OriginalImageName: original name of the image in the Smart Data Bologna.
```
 
- Width: Image Width
- Height: Image Height
- Channels: Image Channels



2) Split the  dataset in train-val-test

```
python data_preprocessing/split_train_val_test_2.py 
```

Generate three files with the list of train, val, test images:   
- `$Info/train.txt`
- `$Info/val.txt`
- `$Info/test.txt`

 The columns of the three files represent respectively: the image name, the SocialNetworkID, UniqueID (unique identifier of the image in the dataset). Furthermore,  in $Info there  are the files relative to the train, val, test patches (`train_crops.txt`, `val_crops.txt`, `test_crops.txt`). The structure is the same.
 
Finally it creates the dictionaries:

- `$Info/train_crop_count.pickle`
- `$Info/val_crop_count.pickle`
- `$Info/test_crop_count.pickle`

 which contain the information of the  number of crops per image in train, val, test. 

3) Generate SDB crops 

```
python data_preprocessing/generate_crops_3.py --patch_size 256 
```

It creates a folder `$FACIL_BOLOGNA_PATH/Patches_256` with the image patches of size 256 x 256.  

4)  Generate the histogram and the prnu residual for the comparison with the network of Amerini et al. 
```
python data_preprocessing/preprocess_amerini_4.py  
```

*This script may take  few hours*.  

5) Compute dataset statistics, they will be used for standardizing/normalizing the Images.
```
python data_preprocessing/compute_statistics_5.py
```

## Amerini Comparison

Script bash to run amerini network on the full SmartDataBologna dataset.
```
bash script/amerini_exp.sh  

```
Script bash to run our network on the full SmartDataBologna dataset.

```
  bash script/our_exp.sh > our_out.txt  
```

## Incremental Experiments
Here some scripts to run the experiments 

```
# exemplar based
bash script/exemplar_based.sh 
```

```
# exemplar free
bash script/exemplar_free.sh 
```

```
# joint incremental
bash script/joint_incremental.sh 
```
## Results Structure 

The scripts create a folder in the path specified by the argument **--results-path** . Each result folder contains:

- **models** : folder containing the models trained after each task. The models are saved in the format  `task_{TASK_ID}.ckpt`
- **results** : patch and image accuracy collected.
  - **acc_tag_*.txt**, **acc_taw_*.txt** :  accuracy task agnostic and task aware after each task. An element **(i,j)** of these matrices is the accuracy tag (or taw) of task **j** after learned task **i**. 
  - **avg_acc_tag_*.txt**, **avg_acc_taw_*.txt** :  average accuracy task agnostic and task aware. An element **i** of these vectors is  the average accuracy after learned task i. The average accuracy learned after the last task is reported in the tables of the paper.
  - **forg_taw_*.txt**, **forg_tag_*.txt**: forgetting task aware and task agnostic.
  - **.pt**: some method use extra information/parameters during training. For example, the bias layers of BIC are saved as *bias_layers_task_{TASK_ID}.pt* 
- **stdout*.txt**: standard output
- **stderr*.txt**: standard error




Most of the code is based on FACIL. Please refer to  *https://github.com/mmasana/FACIL*  for a more detailed explanation of the  continual learning approaches or/and of the whole framework.

## CITE

If you use this code please cite 
```
@INPROCEEDINGS{10157835,
  author={Magistri, Simone and Baracchi, Daniele and Shullani, Dasara and Bagdanov, Andrew D. and Piva, Alessandro},
  booktitle={2023 11th International Workshop on Biometrics and Forensics (IWBF)}, 
  title={Towards Continual Social Network Identification}, 
  year={2023},
  volume={},
  number={},
  pages={1-6},
  doi={10.1109/IWBF57495.2023.10157835}}

```
```

@article{masana2022class,
  title={Class-Incremental Learning: Survey and Performance Evaluation on Image Classification},
  author={Masana, Marc and Liu, Xialei and Twardowski, Bartlomiej and Menta, Mikel and Bagdanov, Andrew D and van de Weijer, Joost},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  doi={10.1109/TPAMI.2022.3213473},
  pages={1-20},
  year={2022}
}
```

Feel free to contribute or propose new features by opening an issue!
