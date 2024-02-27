from transformers import SamModel
import os

# download the pretrained model from huggingface
hf_pretrain_name = "facebook/sam-vit-base"
# hf_pretrain_name = "facebook/sam-vit-large"
# hf_pretrain_name = "facebook/sam-vit-huge"

cache_dir = f"../../work_dirs/sam_cache/{os.path.basename(hf_pretrain_name).replace('-', '_')}"
os.makedirs(cache_dir, exist_ok=True)
# download the pretrained model from huggingface, the pretrained model will be saved in cache_dir
model = SamModel.from_pretrained(hf_pretrain_name, use_safetensors=False)
model.save_pretrained(cache_dir, safe_serialization=False)


