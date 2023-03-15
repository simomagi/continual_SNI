# Framework for Analysis of Class-Incremental Learning
Run the code with:
```
python3 -u src/main_incremental.py
```
followed by general options:

* `--gpu`: index of GPU to run the experiment on (default=0)
* `--results-path`: path where results are stored (default='../results')
* `--exp-name`: experiment name (default=None)
* `--seed`: random seed (default=0)
* `--save-models`: save trained models (default=False)
* `--last-layer-analysis`: plot last layer analysis (default=False)
* `--no-cudnn-deterministic`: disable CUDNN deterministic (default=False)

and specific options for each of the code parts (corresponding to folders):
  
* `--approach`: learning approach used (default='finetuning') [[more](approach)]
* `--datasets`: dataset or datasets used (default=['bologna']) [[more](datasets)]
* `--network`: network architecture used (default='fusion_fc') [[more](networks)]
* `--log`: loggers used (default='disk') [[more](loggers/README.md)]

go to each of their respective readme to see all available options for each of them.

## Approaches
Initially, the approaches included in the framework correspond to the ones presented in
_**Class-incremental learning: survey and performance evaluation**_ (preprint , 2020). The regularization-based
approaches are EWC, LwF, RWalk, Finetuning. The rehearsal approach is  iCaRL.
The bias-correction approaches are IL2M, BiC (orange).

![alt text](../docs/_static/cil_survey_approaches.png "Survey approaches")

More approaches will be included in the future. To learn more about them refer to the readme in
[src/approach](approaches).

## Datasets
To learn about the dataset management refer to the readme in [src/datasets](datasets).

## Networks
To learn about the different torchvision and custom networks refer to the readme in [src/networks](networks).

 
## Utils
We have some utility functions added into `utils.py`.
