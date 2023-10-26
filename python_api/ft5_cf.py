
import datetime
import json
import threading
import time

from pandas import Timestamp
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
from tokenizers import AddedToken







def predict(context,question):
    print("-------------generating prediction-----------------")
    # data = json.loads(request.data)
    # data.context=data.context.replace('\n','\\n')
    # data.context=data.context.replace('\t','\\t')

    # context=request.args.get('context')
    # question=request.args.get('question')
    context=context.replace('\n',' \\n ')
    context=context.replace('\t',' \\t ')



    input=f'paragraph: {context}'
    instruction="extract code snippets from the paragraph, if it does not contain code snippets then say none"
    prompts=[[input,instruction]]
    res=[]
    input_ids = tokenizer(prompts, return_tensors="pt" ,padding=True,truncation=True, max_length=512).input_ids
    outputs = model.generate(input_ids=input_ids, do_sample=True, max_length=150)
    res+=tokenizer.batch_decode(outputs, skip_special_tokens=True)

    return  { 'output': f'{res[0]}' }

def init_model():
    print("-------------loading model------------------")
    from peft import AutoPeftModelForSeq2SeqLM
    from transformers import AutoTokenizer
    import torch
    import numpy as np
    import pandas as pd
    import torch
    import subprocess
    global model
    global tokenizer
    global mh_dir
    
    mh_dir="mh_shell"
    checkpoint_dir=f"/home/uc4ddc6536e59d9d8f8f5069efdb4e25/{mh_dir}/ft_models/flan-t5-xl_mt5_v4/"
    checkpoint_name=subprocess.check_output(f"ls {checkpoint_dir} | grep checkpoint | tail -1",shell=True)
    checkpoint_name=str(checkpoint_name).replace("b'","").replace("\\n'","")
    checkpoint_path=checkpoint_dir+checkpoint_name
    model_path=checkpoint_path

    # model_path=f"/home/u131168/{mh_dir}/model/ft_models/flan-t5-xl_peft_finetuned_model/checkpoint-36000"
    
    model = AutoPeftModelForSeq2SeqLM.from_pretrained(model_path, )
    
    
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-xl")
    stokens_v3=["{","}","<","`","\\"]
    stokens=stokens_v3
    for st in stokens:
        tokenizer.add_tokens(AddedToken(st, normalized=False),special_tokens=False)
    model.resize_token_embeddings(len(tokenizer))

    print("-------------loaded custom model-----------------")
    print("--------------ready----------------------")

def listen_msgs():
    coll_name='mhs_pred_app'
    user_uid='mhs_pred'
        
    # Use a service account.
    cred = credentials.Certificate(f'/home/uc4ddc6536e59d9d8f8f5069efdb4e25/{mh_dir}/python_api/mh-pred-app-21be554adb6d.json')

    app = firebase_admin.initialize_app(cred)

    db = firestore.client()
    
    # Create an Event for notifying main thread.
    callback_done = threading.Event()

    # Create a callback on_snapshot function to capture changes
    def on_snapshot(doc_snapshot, changes, read_time):
        for doc in doc_snapshot:
            print(f"Received document snapshot: {doc.id}")
            print(doc.to_dict())
            data=doc.to_dict()

        
            pred_res=predict(data['context'],data['message'])
            # pred_res={'output':"model response"}

            data['senderId']='chatbot@red'
            data['message']=pred_res['output']
            data['timestamp']=datetime.datetime.utcnow()
            print(f'sending response: '+data['message'])
            doc_id=str(round(time.time() * 1000))
            db.collection(coll_name).document(user_uid).collection('allMessages').document(doc_id).set(data)
            print(f"----------sent----------------with id: {doc_id} ")

        callback_done.set()

    doc_ref = db.collection(coll_name).document(user_uid).collection('userMessages').document("message")

    # Watch the document
    doc_watch = doc_ref.on_snapshot(on_snapshot)
    

    while True:
        print('', end='', flush=True)
        time.sleep(1)



if __name__ == '__main__':
    init_model()
    try:
        listen_msgs()
    except Exception as e:
        print(f'error occured:\n\n\n {e}')  