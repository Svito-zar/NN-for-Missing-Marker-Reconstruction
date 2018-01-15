# Two Neural Network Approaches to Human Pose Reconstruction

Adress all the question to Taras Kucherenko at tarask@kth.se

This github has 3 branches

- master          - removing occlusion experiments (Section 5.2.1)
- missing_markers - random missing markers (Section 5.1)
- denoising       - removing gaussian noise (Section 5.2.2)

Each branch will have a slightly different README file.

## Data preparation

In my experiments I have been using CMU Mocap dataset. There are 2 options on how to get it:

1. Get already preprocessed dataset from my dropbox:


2. Preprocess dataset by yourself
   Note: the results depends a lot on amount and type of the data you use
   - Download CMU Mocap dataset in BVH format from https://sites.google.com/a/cgspeed.com/cgspeed/motion-capture/daz-friendly-release
   - Create folders "train" and "eval" (if flag "evaluate" is true) or "dev" (if flag "evaluate" is false)
   - Move the files you want to use for training into the "train" folder (they should be in the folder themself)
   - Move the files you want to use for testig into the "eval" or "dev" folder (they should be in the folder themself)
   - Set the address to this data in the code/ae/utils/flag.py as data_dir
   - Preprocess it by running the script code/ae/utils/data.py

Afterwards you need to put all test sequences you want to test on into the folder "test_seq", which should be in the same directory as the main folder with the data.

So final configuration should look like this:

.../adress_to_the_data/...
/folder_with_the_data
   eval.binary
   maximums.binary
   mean.binary
   train.binary
   variance.binary
/test_seq
  

## Run
To run the default example execute the following command. 

```bash
$ python code/train.py
```

## Customizing
You can play around with the run options, including the neural net size and shape, input corruption, learning rates, etc. in the file flags.py.

Adress all the question to Taras Kucherenko at tarask@kth.se
