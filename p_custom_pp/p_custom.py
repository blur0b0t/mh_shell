
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
pred_file_name="full_pred6.csv"
pred_file_path=f"/home/u131168/{mh_dir}/data/custom_pred/{pred_file_name}"
# start_index=subprocess.get_output("tail {} -n 1 | awk -F' ' '{print $1}'".format(file_name))
start_index=subprocess.check_output("tail "+pred_file_path+" -n 1 | awk -F' ' '{print $1}'",shell=True)
print(start_index)

start_index=int(start_index)+1    #-------------------------
# start_index=0  #---------comment this line--------------------

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
from tokenizers import AddedToken
from transformers import T5Tokenizer

# checkpoint_dir="/home/u131168/mh_shell/ft_models/flan-t5-xl_peft_finetuned_model/"
checkpoint_dir=f"/home/u131168/{mh_dir}/ft_models/flan-t5-xl_mt5_v4/"
checkpoint_name=subprocess.check_output(f"ls {checkpoint_dir} | grep checkpoint | tail -1",shell=True)
checkpoint_name=str(checkpoint_name).replace("b'","").replace("\\n'","")
checkpoint_path=checkpoint_dir+checkpoint_name
model_path=checkpoint_path
# model_path=f"/home/u131168/{mh_dir}/ft_models/flan-t5-xl_mt5/checkpoint-17500"
# tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-xl")
model = AutoPeftModelForSeq2SeqLM.from_pretrained(model_path, )


# tokenizer:
# tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-xl")
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-xl")

# stokens_v1=["`","~","!","@","#","$","%","^","&","*","(",")","-","_","+","=","{","}","[","]","|","\\",":",";","\"","'","<",">","?","/","\n","\t"," "]
# stokens_v2=["{","}","<","<<"]
stokens_v3=["{","}","<","`","\\"]

stokens=stokens_v3
for st in stokens:
    tokenizer.add_tokens(AddedToken(st, normalized=False),special_tokens=False)
model.resize_token_embeddings(len(tokenizer))


# ------predict--------------
bs=1
end_index=len(test_data)
for i in range(start_index,end_index,bs):
    res=[]

    if test_data.loc[i,['output']].notnull().any() and i>15:
    # if False:
        res+=[test_data.iloc[i,2]]

    else:
        print(f"predicting {i} to {i+bs-1} prompt")
        prompts = test_data.loc[i:i+bs-1,['input','instruction']].values.tolist()
        prompts=prompts
        t_prompts=[]
        for p in prompts:
            # context=str(p[0]).replace(r"\n",'.')
            context=str(p[0])
            question=p[1]
            t_prompts.append([context,question])
            # t_prompts+=[f"input: {context}\n\ninstruction: {question}"]
        prompts=t_prompts
            
        # print(prompts)
        input_ids = tokenizer(prompts, return_tensors="pt" ,padding=True,truncation=True, max_length=512).input_ids
        # sample up to 30 tokens
        torch.manual_seed(0)  # doctest: +IGNORE_RESULT
        outputs = model.generate(input_ids=input_ids, do_sample=True, max_length=312)
        # outputs[outputs == 0] = 2 # convert output id 0 to 2 (eos_token_id)
        # outputs[outputs == -1] = 1 # convert improper tokens to ''
        res+=tokenizer.batch_decode(outputs, skip_special_tokens=True,spaces_between_special_tokens = False)
            
    # Writing data to a file
    with open(pred_file_path, "a+") as file1:
        file1.writelines(f"{i+i1} $$ {res[i1]}\n" for i1 in range(len(res)))
    print(f"\n-------------wrote {i} to {i+bs-1} preds")


print("-----------------Prediction_finished-----------------------")
print(subprocess.check_output("scancel $((SLURM_JOB_ID+1))",shell=True))