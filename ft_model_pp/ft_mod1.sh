#!/bin/sh


# sbatch -x idc-beta-batch-pvc-node-[03,09,20,21] --priority 0 --job-name fts1 ft_mod1.sh
# sbatch -x idc-beta-batch-pvc-node-[03,09,20,21] --priority 0 --job-name fts2 --dependency=afterany:26374 ft_mod1.sh



export batch_script="ft_mod1.sh"
# -----------set new job dep--------------
echo "got current job name=$SLURM_JOB_NAME"
export cji=$(echo -n $SLURM_JOB_NAME | tail -c 1)
export nji=$(( cji + 1 ))
# export nji=$($SLURM_JOB_NAME | tail -c 1| awk '{print $1 + 1}')
export njname="fts$nji"
echo "new job name=$njname"
export njid=$(sbatch -x idc-beta-batch-pvc-node-[03,09,20,21] --priority 0 --job-name $njname --begin=now+60 --dependency=afterany:$SLURM_JOB_ID $batch_script | sed -n 's/.*job //p')
echo "new job created with id: $njid"
# -------------------end------------------



echo "starting fine tuning model"
cd "/home/u131168/mh_shell/ft_models/intel-extension-for-transformers/workflows/chatbot/fine_tuning"
pip install -r "requirements.txt"
cd "/home/u131168/mh_shell/ft_models/intel-extension-for-transformers/workflows/chatbot/fine_tuning/instruction_tuning_pipeline"
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
export checkpoint_dir="/home/u131168/mh_shell/ft_models/flan-t5-xl_mt5_v1/"
export checkpoint_name=$(ls $checkpoint_dir | grep checkpoint | tail -1)
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
        --learning_rate 1.0e-5 \
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
