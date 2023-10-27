# mh_shell


<hr>


MachineHack | Shell cybersecurity hackathon 2023 -

![image](https://github.com/blur0b0t/mh_shell/assets/143605527/9f9a65a8-82fb-4f4e-b03d-dc1f5b135d8f)


<img src=https://github.com/blur0b0t/mh_shell/assets/143605527/0c1952ba-6ab7-4cc1-bcfa-81e26b783c9a width=30% height=30%>



<br />
<br />

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/blur0b0t/mh_shell)

# About Hackathon:

Shell, in collaboration with MachineHack & Analytics India Magazine, asked participants to tackle the exponential growth of cyber threats and improve the security and resilience of web applications. The aim of the Cyber Threat Detection Hackathon is to build a next-gen model capable of identifying code in a body of text. 
 

 
# Problem statement
Protecting our software landscapes is not an easy task. Malicious actors are frequently trying to enter systems and get access to resources, whether operational or data. The ability for an actor to compromise systems, elevate their privileges, and move laterally within infrastructure typically hinges on executing hidden code. One common method they employ is embedding this code in seemingly harmless media—whether it's images, videos, or even simple text files.


# Detailed Architecture Flow:

<img width="424" alt="image" src="https://github.com/redR0b0t/mh_one_api/assets/143605527/1c8effb2-c0b2-44cb-a638-da3e46814e6d">




# Step-by-Step Code Execution Instructions:


- Clone the Repository

```bash
 $ git clone https://github.com/blur0b0t/mh_shell
 $ cd mh_shell
```

- Train/Fine-tune the flan-t5-xl model on slurm workload manager.


```bash
#!/bin/bash


export batch_script="ft_mod1.sh"
# -----------set new job dep--------------
echo "got current job name=$SLURM_JOB_NAME"
export cji=$(echo -n $SLURM_JOB_NAME | tail -c 1)
export nji=$(( cji + 1 ))
# export nji=$($SLURM_JOB_NAME | tail -c 1| awk '{print $1 + 1}')
export njname="fts$nji"
echo "new job name=$njname"
export njid=$(sbatch -x idc-beta-batch-pvc-node-[03,09,14,20,21] --priority 0 --job-name $njname --begin=now+60 --dependency=afterany:$SLURM_JOB_ID --mem=0 --exclusive $batch_script | sed -n 's/.*job //p')
echo "new job created with id: $njid"
# -------------------end------------------




echo "----------checking if gpu available on current job-----------------"
conda init bash
# oneapi env and checking gpu
echo "-------------------------------------------"
groups
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



echo "starting fine tuning model"
cd "/home/u131168/mh_shell/ft_model_pp/itp"
pip install -r "requirements.txt"
# To use ccl as the distributed backend in distributed training on CPU requires to install below requirement.
python -m pip install oneccl_bind_pt -f https://developer.intel.com/ipex-whl-stable-cpu

cd "/home/u131168/mh_shell/ft_model_pp/itp"
# pip install intel-extension-for-transformers
# pip install fschat==0.1.2
pip install git+https://github.com/huggingface/transformers
pip install tokenizers



# sbatch -x idc-beta-batch-pvc-node-[03,20,21] --job-name fts1 --priority=0 ft_mod1.sh
# sbatch -w idc-beta-batch-pvc-node-[05] --job-name fts1 --priority=0 ft_mod1.sh


export train_file="/home/u131168/mh_shell/data/f_traind_v1.csv"

export model_path="google/flan-t5-xl"
# export model_path="/home/u131168/mh_shell/ft_models/flan-t5-xl_mt5_v1"

# export checkpoint_path="/home/u131168/mh_shell/ft_models/flan-t5-xl_peft_finetuned_model/checkpoint-36000"
export checkpoint_dir="/home/u131168/mh_shell/ft_models/flan-t5-xl_mt5_v4/"
export checkpoint_name=$(ls $checkpoint_dir | grep checkpoint | tail -2 | head -n 1)
export checkpoint_path="$checkpoint_dir$checkpoint_name"
echo $checkpoint_path

export output_dir="$checkpoint_dir"




python finetune_seq2seq.py \
        --model_name_or_path $model_path \
        --resume_from_checkpoint $checkpoint_path \
        --bf16 True \
        --train_file $train_file \
        --per_device_train_batch_size 2 \
        --per_device_eval_batch_size 2 \
        --gradient_accumulation_steps 1 \
        --do_train \
        --learning_rate 1.0e-6 \
        --warmup_ratio 0.03 \
        --weight_decay 0.0 \
        --num_train_epochs 1 \
        --logging_steps 10 \
        --save_steps 100 \
        --save_total_limit 2 \
        --overwrite_output_dir \
        --output_dir $output_dir \
        --peft lora

echo "finished fine tuning model"

```

- Perform inference on the test dataset with finetuned flan-t5-xl-peft model

```bash
#!/bin/bash
export batch_script="p_custom.sh"
conda init bash
# -----------set new job dep--------------
echo "got current job name=$SLURM_JOB_NAME"
export cji=$(echo -n $SLURM_JOB_NAME | tail -c 1)
export nji=$(( cji + 1 ))
export njname="pcs$nji"
echo "new job name=$njname"
export njid=$(sbatch -x idc-beta-batch-pvc-node-[03,20,21] --priority 0 --job-name $njname --begin=now+60 --dependency=afterany:$SLURM_JOB_ID --mem=0 --exclusive $batch_script | sed -n 's/.*job //p')
echo "new job created with id: $njid"
# -------------------end------------------




echo "----------checking if gpu available on current job-----------------"
conda init bash
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

# conda acti
pip install torch
pip install transformers
pip install peft



python /home/u131168/mh_shell/p_custom_pp/p_custom.py

echo "finished precition"



```

# Run web application to interact with the finteuned flan-t5-xl-peft model to detect code snippets in the given paragraph.

- Run python app to serve predictions(code snippets) to the frontend.
- (*the webapp wont work ,if the python app is not running)

```bash
 
cd python_api
pip install -r ./reqs.txt
python ./ft5_cf.py


```

![Screenshot (30)](https://github.com/blur0b0t/mh_shell/assets/143605527/3a9fe31c-7fd9-42f3-98b1-7c1d271621fc)


![Screenshot (31)](https://github.com/blur0b0t/mh_shell/assets/143605527/55f89773-9587-464f-8cc3-4371eab6eadb)




# Run frontend application(webapp) to extract code snippets from a give paragraph.
- (*make sure that the python application is running before using the webapp)
  <br />



- option 1: use the web app hosted on huggingface spaces:
```bash
https://huggingface.co/spaces/blur0b0t/mh_shell
```

![Screenshot (29)](https://github.com/blur0b0t/mh_shell/assets/143605527/7f75a20e-7205-4bed-b52b-d89f020b8f53)




- option 2: use the prebuild files
```bash
cd mhs_pred_app/build/web
# run index.html file from browser to access the webapp
```

- option 3: build app from flutter sdk (*flutter sdk need to be installed on the system)
```bash
cd mhs_pred_app
flutter run -d web-server --host=0.0.0.0
```

![Screenshot (27)](https://github.com/blur0b0t/mh_shell/assets/143605527/a21c0a41-401b-4e10-a08b-eddb70460154)


![Screenshot (28)](https://github.com/blur0b0t/mh_shell/assets/143605527/48866e12-5f47-4950-ac8d-08bfb0f00a71)





 <br />
 <br />
    
- (*hugging face currently does not support inference api for peft models, so we need to run the python app to serve predictions for the webapp to work.)
- webapp available on Huggingface Spaces (https://huggingface.co/spaces/blur0b0t/mh_shell)
- model available on Huggingface Hub (https://huggingface.co/blur0b0t/mh_shell)


