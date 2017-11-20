import numpy as np
import subprocess
from ae.utils.flags import FLAGS

def run_bash(cmd):
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, executable='/bin/bash')
    out = p.stdout.read().strip()
    return out  # This is the stdout from the shell command

# Set the maximal number of jobs per GPU
max_job_num_per_gpu=1
run_bash('ts -S %d'%max_job_num_per_gpu)

# Set the combinations of hyper-parameters we want to check
#initial_lr = 0.0000125
param_list = list( {'variance_of_noise': i} for i in np.linspace(0.02,0.3,num=15) )
#param_list = list( {'batch_size': int(np.power(2,bt)), 'learning_rate': 0.00004*np.power(5,lr)} for bt in np.linspace(4,7,num=4) for lr in np.linspace(1,3,3))
#param_list = list( {'chunk_length': int(np.power(2,i)), 'chunking_stride': int(np.power(2,j))} for i in np.linspace(4,10,num=7) for j in np.linspace(2,i,num=i-2+1) )

length_vals = [10, 10,10,10] #, 60, 100, 150, 300]
#param_list = list( {'amount_of_frames_as_input': 1} for i in length_vals )

#Do random search
#param_list = list( {'numb_of_LSTM_layers': int(i), 'training_learning_rate': 0.0002 * np.power(3,k), 'recurrent_layer': int(j)} for i in np.linspace(1,3, num=3) for j in np.linspace(1,2,2) for k in np.linspace(0,2,3))  #np.lohspace(0,3, num=4, base=7))
#param_list = list( {'network_width': int(i), 'num_hidden_layers': int(j), 'training_learning_rate': 0.00005 * np.power(4,k)} for i in np.logspace(8,10, num=3, base=2) for j in np.linspace(1,2,2) for k in np.linspace(0,1,2))
#param_list = list( {'chunk_length': int(np.power(2,i)), 'chunking_stride': int(np.power(2,j))} for i in np.linspace(4,10,num=7) for j in np.linspace(2,i,num=i-2+1) )

#Do random search
#param_list = list( {'numb_of_LSTM_layers': int(i), 'training_learning_rate': 0.0002 * np.power(3,k), 'recurrent_layer': int(j)} for i in np.linspace(1,3, num=3) for j in np.linspace(1,2,2) for k in np.linspace(0,2,3))  #np.lohspace(0,3, num=4, base=7))
#param_list = list( {'network_width': int(i), 'num_hidden_layers': int(j), 'learning_rate': 0.00004 * np.power(5,k)} for i in np.logspace(9,11, num=3, base=2) for j in np.linspace(1,2,2) for k in np.linspace(0,2,3))

# Empty the file with results
f = open(FLAGS.results_file, 'w')
#f.write('For the dropout ' + str(dropout)+' the final train error was '+str(train_err)+' and test error was '+str(test_err))
f.close()  # y

GPU = 0

# Run the scripts with all the parameter sets
for combination_dict in param_list:

    job_cmd = 'CUDA_VISIBLE_DEVICES='+str(GPU)+' python ae/train.py ' + '  '.join(
            ['--{}={}'.format(flag, value) for flag, value in combination_dict.iteritems()])

    submit_cmd = "ts bash -c '%s'" % job_cmd
    run_bash(submit_cmd)

    GPU = 1 - GPU

print('The parameter list is: ', param_list)
print('The results will be written to ', FLAGS.results_file)
print('HAVE YOU CHANED TRAIN.PY ACCORDINGLY?')
