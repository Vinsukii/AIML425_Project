# MY CONTRIBUTION
## Algorithm modification
* Where the code in between the comments -MY CONTRIBUTION (START)- and -MY CONTRIBUTION (END)- denotes my contribution
* Added a 'GATedge_ops' class in graph/hgnn.py to perform the modified algorithm for operation node embeddings (line 170-356)
* Apply the new class for the opeartion node embedding in PPO_model.py (line 144-152)


## Results folder (on 10x5 instances)
* Each folder contains the experimental results of the trained model
* Train & Test output files include training/testing time (in seconds), mean/max/min makespan
* The mean validation makespan graph is included in each 'train_...' folder
* All runs were done on the lab computer CPU (Intel i7-8700 12 cores @ 3.2GHz)

## Other changes
* The output files are now in .csv format instead of .xls
* During testing, only the sampling strategy (DRL-S) was used
* Train and test folder name now contain "_paper" or "_modified" to indicate the algorithm used (orignal or modified)

Original code can be found here: https://github.com/songwenas12/fjsp-drl

---

# fjsp-drl
Implementation of the IEEE TII paper [Flexible Job Shop Scheduling via Graph Neural Network and Deep Reinforcement Learning](https://ieeexplore.ieee.org/document/9826438). *IEEE Transactions on Industrial Informatics*, 2022.


```
@ARTICLE{9826438,  
   author={Song, Wen and Chen, Xinyang and Li, Qiqiang and Cao, Zhiguang},  
   journal={IEEE Transactions on Industrial Informatics},   
   title={Flexible Job Shop Scheduling via Graph Neural Network and Deep Reinforcement Learning},   
   year={2022},  
   volume={},  
   number={},  
   pages={1-11},  
   doi={10.1109/TII.2022.3189725}
 }
```

## Get Started

### Installation

* python $\ge$ 3.6.13
* pytorch $\ge$ 1.8.1
* gym $\ge$ 0.18.0 & $\le$ 0.20.0
* numpy $\ge$ 1.19.5
* pandas $\ge$ 1.1.5
* visdom $\ge$ 0.1.8.9

Note that pynvml is used in ```test.py``` to avoid excessive memory usage of GPU, please modify the code when using CPU.

### Introduction

* ```data_dev``` and ```data_test``` are the validation sets and test sets, respectively.
* ```data``` saves the instance files generated by ```./utils/create_ins.py```
* ```env``` contains code for the DRL environment
* ```graph``` is part of the code related to the graph neural network
* ```model``` saves the model for testing
* ```results``` saves the trained models
* ```save``` is the folder where the experimental results are saved
* ```utils``` contains some helper functions
* ```config.json``` is the configuration file
* ```mlp.py``` is the MLP code (referenced from L2D)
* ```PPO_model.py``` contains the implementation of the algorithms in this article, including HGNN and PPO algorithms
* ```test.py``` for testing
* ```train.py``` for training
* ```validate.py``` is used for validation without manual calls

## Reproduce result in paper

There are various experiments in this article, which are difficult to be covered in a single run. Therefore, please change ```config.json``` before running.

Note that disabling the ```validate_gantt()``` function in ```schedule()``` can improve the efficiency of the program, which is used to check whether the solution is feasible.

### train

```
python train.py
```

Note that there should be a validation set of the corresponding size in ```./data_dev```.

### test

```
python test.py
```
Note that there should be model files (```*.pt```) in ```./model```.

## Reference

* https://github.com/zcaicaros/L2D
* https://github.com/yd-kwon/MatNet
* https://github.com/dmlc/dgl/tree/master/examples/pytorch/han
