from transformers import SamModel

# download the pretrained model from huggingface
hf_pretrain_name = "facebook/sam-vit-base"
# hf_pretrain_name = "facebook/sam-vit-large"
# hf_pretrain_name = "facebook/sam-vit-huge"

# download the pretrained model from huggingface, the pretrained model will be saved in ~/.cache/huggingface/hub
model = SamModel.from_pretrained(hf_pretrain_name)


