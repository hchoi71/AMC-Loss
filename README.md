_______
# AMC-Loss: Angular Margin Contrastive Loss for Improved Explainability in Image Classification

## Overview
This is re-implementation of the AMC loss described in: [paper](https://arxiv.org/pdf/2004.09805.pdf)

## Requirements
* tensorflow>=1.12.0
* python>=3.6.0

## Image Classification
We use CIFAR10 classification as an example with a simple architecture. In order to reproduce the results described on the paper, please modify the hyperparameters in Train_AMC.py and then run > - <python Train_AMC.py>.
  
> - Test 123
> - Follow  [testing artifacts](http://2.bp.blogspot.com) (more Unit )

The users can also change the data to other dataset at their interest. This code contains visualization by Grad-CAM from here [https://github.com/Ankush96/grad-cam.tensorflow]. 
