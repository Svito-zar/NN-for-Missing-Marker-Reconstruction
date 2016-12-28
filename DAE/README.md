#Deep Autoencoder with TensorFlow for Represantation Learning of CMU MoCap dataset

The goal of this project is to learn a mapping from the human posture to the robot posture, such that robot can convey a human emotion.

First we need to learn a good represation of the human motion. We will do it by Deep Autoencoder.

I am going to try different architectures of AutoEncoder, as well as, recurrency. They will be in different branches.

```
##Run
To run the default example execute the following command. 
NOTE: this will take a very long time if you are running on a CPU as opposed to a GPU
```bash
$ python code/run.py
```

##Customizing
You can play around with the run options, including the neural net size and shape, input corruption, learning rates, etc.
in the file flags.py.
