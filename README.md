# A Neural Network Approach to Missing Marker Reconstruction


This is an implementation for the paper [A Neural Network Approach to Missing Marker Reconstruction in Human Motion Capture](https://arxiv.org/abs/1803.02665).
Youtube video of the experimental results is [here](https://www.youtube.com/watch?v=mi75gzEhbHI).

## Requirements
- Python 3
- Tensorflow >= 1.0
- Additional Python libraries:
  - numpy
  - matplotlib (if you want to visualize the results)
  - btk (if you want to preprocess test sequences by yourself)


## Data preparation

In my experiments I have been using CMU Mocap dataset. There are 2 options on how to get it:

1. Download [already preprocessed dataset](https://kth.box.com/s/w3xjeozhshgnemjeik05rzcr1kvaoww1):

   Take the test sequences, I used in the paper [here](https://kth.box.com/s/5p0sbt1ufmwzddlfi2sw66cjaamno7cl)



2. Preprocess dataset by yourself

    (** Note: the results depends a lot on amount and type of the data you use **)

   - Download [CMU Mocap dataset in BVH format](https://sites.google.com/a/cgspeed.com/cgspeed/motion-capture/daz-friendly-release)
   - Create folders "train" and "eval" (if flag "evaluate" is true) or "dev" (if flag "evaluate" is false)
   - Move the files you want to use for training into the "train" folder (they should be in the folder themself)
   - Move the files you want to use for testig into the "eval" or "dev" folder (they should be in the folder themself)
   - Set the address to this data in the code/ae/utils/flag.py as data_dir
   - Preprocess it by running the script code/ae/utils/data.py

   Afterwards you need to put all test sequences you want to test on into the folder "test_seq", which should be in the same directory as the main folder with the data.
   Then you preprocess those sequence by function "write_test_seq_in_binary" from the file ae/utils/data.py, which will write the test sequences in the binary format for the faster and easier access.

So final configuration should look like this:

'''
.../adress_to_the_data/...

/folder_with_the_data
- eval.binary
- maximums.binary
- mean.binary
- train.binary
- variance.binary

/test_seq
- basketball.binary
- boxing.binary
- salto.binary
...
'''
  

## Run
To run the default example execute the following command. 

```bash
$ python code/ae/train.py
```

## Customizing
You can play around with the run options, including the neural net size and shape, input corruption, learning rates, etc. in the file flags.py.
Otherwise - you can find the Best Flags in the folder BestFlags

## Contact
If you encounter any problems/bugs/issues please contact me on github or by emailing me at tarask@kth.se for any bug reports/questions/suggestions. I prefer questions and bug reports on github as that provides visibility to others who might be encountering same issues or who have the same questions.

## Citation
Here is the citation in bib format:
```
@article{kucherenko2018neural,
  title={A Neural Network Approach to Missing Marker Reconstruction in Human Motion Capture},
  author={Kucherenko, Taras and Beskow, Jonas and Kjellstr{\"o}m, Hedvig},
  journal={https://arxiv.org/abs/1803.02665v4},
  year={2018}
}
```
