# DAE-for-Mocap-Representation-Learning
This repository is about Deep Autoencoders(DAE) usage for the Motion Caption task ( Mocap ) with emphasis on Representation Learning

DAE is based on https://github.com/cmgreen210/TensorFlowDeepAutoencoder. 
BVH parser is based on https://github.com/lawrennd/mocap

CUSTOMIZING

You can play around with the run options, including the neural net size and shape, input corruption, learning rates, etc. in flags.py.

SETUP

It is expected that Python2.7 is installed and your default python version.
After cloning repository do the following:

Ubuntu/Linux

$ cd DAE-for-Mocap-Representation-Learning
$ sudo chmod +x setup_linux
$ sudo ./setup_linux  # If you want GPU version specify -g or --gpu
$ source venv/bin/activate 

Mac OS X

$ cd DAE-for-Mocap-Representation-Learning
$ sudo chmod +x setup_mac
$ sudo ./setup_mac
$ source venv/bin/activate 
Run

RUN

$ ./learn

VIZUALIZING

Navigate to http://localhost:6006 to explore TensorBoard and view training progress.
