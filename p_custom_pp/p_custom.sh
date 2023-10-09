#!/bin/bash


# sbatch -x idc-beta-batch-pvc-node-[03,20,21] --priority 0 --job-name pcs1 p_custom.sh
export batch_script="p_custom.sh"
# -----------set new job dep--------------
echo "got current job name=$SLURM_JOB_NAME"
export cji=$(echo -n $SLURM_JOB_NAME | tail -c 1)
export nji=$(( cji + 1 ))
export njname="pcs$nji"
echo "new job name=$njname"
export njid=$(sbatch -x idc-beta-batch-pvc-node-[03,20,21] --priority 0 --job-name $njname --begin=now+60 --dependency=afterany:$SLURM_JOB_ID $batch_script | sed -n 's/.*job //p')
echo "new job created with id: $njid"
# -------------------end------------------




echo "----------checking if gpu available on current job-----------------"
# oneapi env and checking gpu
echo "-------------------------------------------"
groups  # Key group is render, PVC access is unavailable if you do not have render group present.
source /opt/intel/oneapi/setvars.sh --force
sycl-ls
export num_gpu="$(sycl-ls |grep "GPU" |wc -l)"
echo "num_gpu=$num_gpu\n"
export num_cpu="$(sycl-ls |grep "Xeon" |wc -l)"
echo "num_cpu=$num_cpu\n"
if [ $num_gpu == 0 && $num_cpu == 1] 
then 
    echo "---GPU not available exiting--------"
    scancel $SLURM_JOB_ID
fi 
echo "-------------------------------------------"



echo "staring prediction"

source /opt/intel/oneapi/setvars.sh

# conda acti
pip install torch
pip install transformers
pip install peft



python /home/u131168/mh_shell/p_custom_pp/p_custom.py

echo "finished precition"


# sbatch -x idc-beta-batch-pvc-node-[03,20] --priority 0 --job-name pc0 p_custom.sh
# sbatch -x idc-beta-batch-pvc-node-[03,20] --priority 0 --job-name pc1 p_custom.sh
# sbatch -x idc-beta-batch-pvc-node-[03,20] --priority 0 --job-name pc2 p_custom.sh


# idc-beta-batch-pvc-node-08:
# idc-beta-batch-pvc-node-10
