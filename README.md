# Graph-PS
Sequence-Based Prediction of Phase-Separated Protein Classification by Graph Convolutional Network
## Introduction
The latest research findings have demonstrated that phase separation reveals the mechanism of membrane-less organelle formation, and an increasing number of proteins have been confirmed to possess this characteristic. Investigating phase-separated proteins holds significant importance in understanding related diseases. Numerous machine learning methods have emerged for classifying phase-separated proteins. This study introduces Graph-PS, a machine learning model that uses graph convolutional neural networks and multi-head attention modules. It classifies proteins by analyzing amino acid and structural features extracted from protein sequences. Our model addresses the issue of other models being biased towards predicting a specific class of phase-separated proteins and demonstrates the ability to accurately predict each class of phase-separated proteins.
## Requirements
To run this program, you may need:
 * Python 3.6 or later
 * Pytorch 1.12.1 and other related packages
 * Windows 10 enviroment
 * GPU (optional for cuda)
## How to use
1. Set up your enviroment and download the code from github:
  ```
     git clone https://github.com/ken0414/Graph-PS.git
  ```
2. Put your data into the appropriate folder:
  ```
     [network embedding feature(node2vec)] --> ./data/n2v
     [protein structure feature] --> ./data/graph    
     [protein node feature(PSSM)] --> ./data/pssm
  ```
3. Activate your enviroment and run main.py:
  ```
     $ python main --mode cv --n2v example_n2v --run 10
  ```
  Within this line of code, you can choose between cross-validation mode or independent test set mode by modifying the value after mode, n2v to change the file used, and run to select the number of repetitive runs.
| option  | value |
| ------------- | ------------- |
| `mode` | `cv`or`out` |
| `n2v` | filename of node2vec |
| `run` | int value for run times |
4. Get result:
  For each fold in a single cv, you can get the best epoch of the train in `train_result.txt`.
  After all fold trained in a single cv, you can get the evaluation of all fold in `predict_result.txt` and the result of prediction in fold `./result`.
  If you run on `out` mode, there will be only 1 result in `train_result.txt` and `predict_result.txt`.
  If the value of `run` is bigger than 1, former result in `train_result.txt` of a single `cv` or `out` will be override. If you want to save the result of this file, please modify the code on your own.
  Samely, if you run the main.py again, all the result will be override.
