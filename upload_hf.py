from huggingface_hub import login
login()
from huggingface_hub import HfApi
api = HfApi()

model_path="/home/uc4ddc6536e59d9d8f8f5069efdb4e25/mh_shell/ft_models/flan-t5-xl_mt5_v4/checkpoint-140600"
# model_path="/workspaces/mh_one_api/model/ft_models/ft_final"
api.upload_folder(
    folder_path=model_path,
    repo_id="blur0b0t/mh_shell",
    repo_type="model",
)


# webapp_path="/workspaces/mhs_pred_app/build/web"
# api.upload_folder(
#     folder_path=webapp_path,
#     repo_id="blur0b0t/mh_one_api",
#     repo_type="space",
# )