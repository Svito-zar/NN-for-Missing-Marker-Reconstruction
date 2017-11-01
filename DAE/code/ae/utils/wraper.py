import numpy as np
import subprocess

def run_bash(cmd):
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, executable='/bin/bash')
    out = p.stdout.read().strip()
    return out  # This is the stdout from the shell command

# Set the maximal number of jobs per GPU
#max_job_num_per_gpu=int(1/gpu_memory_fraction)
#run_bash('ts -S %d'%max_job_num_per_gpu)

# Set the combinations of hyper-parameters we want to check
param_list = list( {'dropout': i } for i in np.linspace(0.7, 0.9, 5))
print param_list
combinations_list = param_list

# Run the scripts with all the parameter sets
for combination_dict in combinations_list:

    job_cmd = 'python code/ae/train.py ' + '  '.join(
            ['--{}={}'.format(flag, value) for flag, value in combination_dict.iteritems()])

    submit_cmd = "ts bash -c '%s'" % job_cmd
    run_bash(submit_cmd)
