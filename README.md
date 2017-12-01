# Two Neural Network Approaches to Human Pose Reconstruction

About the branches:

Master - removing occlusion experiments
missing_markers - random missing markers
denoising - removing gaussian noise



## Data preparation

1. Download CMU Mocap dataset in BVH format from https://sites.google.com/a/cgspeed.com/cgspeed/motion-capture/daz-friendly-release
2. Set the address to this data in the code/ae/utils/flag.py as data_dir

## SETUP

It is expected that Python2.7 is installed and your default python version.
After cloning repository do the following:


Ubuntu/Linux
'''
$ cd DAE-for-Mocap-Representation-Learning
$ sudo chmod +x setup_linux
$ sudo ./setup_linux  # If you want GPU version specify -g or --gpu
$ source venv/bin/activate 
'''

Mac OS X
'''
$ cd DAE-for-Mocap-Representation-Learning
$ sudo chmod +x setup_mac
$ sudo ./setup_mac
$ source venv/bin/activate 
'''

##Run
To run the default example execute the following command. 
NOTE: this will take a very long time if you are running on a CPU as opposed to a GPU
```bash
$ python code/run.py
```

##Customizing
You can play around with the run options, including the neural net size and shape, input corruption, learning rates, etc.
in the file flags.py.
