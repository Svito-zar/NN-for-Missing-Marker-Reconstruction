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
#param_list = list( {'variance_of_noise': i, 'training_learning_rate': 0.0003 } for i in np.linspace(0,0.5,num=11) )
param_list = list( {'chunk_length': int(np.power(2,i)), 'chunking_stride': int(np.power(2,j))} for i in np.linspace(4,10,num=7) for j in np.linspace(2,i,num=i-2+1) )

#Do random search
#param_list = list( {'numb_of_LSTM_layers': int(i), 'training_learning_rate': 0.0002 * np.power(3,k), 'recurrent_layer': int(j)} for i in np.linspace(1,3, num=3) for j in np.linspace(1,2,2) for k in np.linspace(0,2,3))  #np.lohspace(0,3, num=4, base=7))
#param_list = list( {'network_width': int(i), 'num_hidden_layers': int(j), 'training_learning_rate': 0.00002 * np.power(4,k)} for i in np.logspace(6,11, num=6, base=2) for j in np.linspace(2,8,7) for k in np.linspace(0,3,4))
#param_list = list( {'chunk_length': int(np.power(2,i)), 'chunking_stride': int(np.power(2,j))} for i in np.linspace(4,10,num=7) for j in np.linspace(2,i,num=i-2+1) )

#Do random search
#param_list = list( {'numb_of_LSTM_layers': int(i), 'training_learning_rate': 0.0002 * np.power(3,k), 'recurrent_layer': int(j)} for i in np.linspace(1,3, num=3) for j in np.linspace(1,2,2) for k in np.linspace(0,2,3))  #np.lohspace(0,3, num=4, base=7))
param_list = list( {'network_width': int(i), 'num_hidden_layers': int(j), 'training_learning_rate': 0.00002 * np.power(4,k)} for i in np.logspace(6,11, num=6, base=2) for j in np.linspace(2,8,7) for k in np.linspace(0,3,4))

# Empty the file with results
f = open(FLAGS.results_file, 'w')
#f.write('For the dropout ' + str(dropout)+' the final train error was '+str(train_err)+' and test error was '+str(test_err))
f.close()  # y

# Run the scripts with all the parameter sets
for combination_dict in param_list:

    job_cmd = 'CUDA_VISIBLE_DEVICES=0 python ae/train.py ' + '  '.join(
            ['--{}={}'.format(flag, value) for flag, value in combination_dict.iteritems()])

    submit_cmd = "ts bash -c '%s'" % job_cmd
    run_bash(submit_cmd)

print('The parameter list is: ', param_list)
print('The results will be written to ', FLAGS.results_file)
print('HAVE YOU CHANED TRAIN.PY ACCORDINGLY?')
