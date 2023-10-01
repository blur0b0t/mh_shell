
import numpy as np
import pandas as pd
import torch
import subprocess

mh_dir='mh_shell'


# batch_process=2
# # specify start index for continuing...
# start_index=[0,9500,19000]
# end_index=[9500,19000,29000]
# file_name=['0_10k.csv','10_20k.csv','20_30k.csv']
pred_file_name="full_pred.csv"
pred_file_path=f"/home/u131168/{mh_dir}/data/custom_pred/{pred_file_name}"
# start_index=subprocess.get_output("tail {} -n 1 | awk -F' ' '{print $1}'".format(file_name))
start_index=subprocess.check_output("tail "+pred_file_path+" -n 1 | awk -F' ' '{print $1}'",shell=True)
print(start_index)

# start_index=int(start_index)+1    #-------------------------
start_index=0  #---------comment this line--------------------

end_index=29000

print(f"---------------got start index============{start_index}")




test_path=f"/home/u131168/{mh_dir}/data/f_data.csv"
test_data=pd.read_csv(test_path)



# !pip install bitsandbytes
# !pip install accelerate
# !pip install scipy

# load model
from peft import AutoPeftModelForSeq2SeqLM
from transformers import AutoTokenizer
import torch

model_path=f"/home/u131168/{mh_dir}/ft_models/flan-t5-xl_mt5/checkpoint-3600"
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-xl")
model = AutoPeftModelForSeq2SeqLM.from_pretrained(model_path, )

# ------predict--------------
bs=10
for i in range(start_index,end_index,bs):
    print(f"predicting {i} to {i+bs-1} prompt")
    prompts = test_data.loc[i:i+bs-1,['Story','Question']].values.tolist()
    prompts=prompts
    t_prompts=[]
    for p in prompts:
        # context=str(p[0]).replace(r"\n",'.')
        context=str(p[0])
        question=p[1]
        t_prompts.append([f"paragraph:  {context}",f"Answer the following question from the paragraph : Question: {question}"])
        # t_prompts+=[f"input: {context}\n\ninstruction: {question}"]
    prompts=t_prompts
        
    # print(prompts)
    res=[]
    input_ids = tokenizer(prompts, return_tensors="pt" ,padding=True,truncation=True, max_length=512).input_ids
    # sample up to 30 tokens
    torch.manual_seed(0)  # doctest: +IGNORE_RESULT
    outputs = model.generate(input_ids=input_ids, do_sample=True, max_length=150)
    res+=tokenizer.batch_decode(outputs, skip_special_tokens=True)
        
    # Writing data to a file
    with open(pred_file_path, "a+") as file1:
        file1.writelines(f"{i+i1} $$ {res[i1]}\n" for i1 in range(len(res)))
    print(f"\n-------------wrote {i} to {i+bs-1} preds")


print("-----------------Prediction_finished-----------------------")
print(subprocess.check_output("scancel $((SLURM_JOB_ID+1))"))