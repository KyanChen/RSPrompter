from transformers import AutoTokenizer, CLIPModel
from PIL import Image
import requests
from transformers import AutoProcessor, CLIPModel

config_ = "openai/clip-vit-large-patch14-336"
# config_ = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(config_)
tokenizer = AutoTokenizer.from_pretrained(config_)
processor = AutoProcessor.from_pretrained(config_)

inputs = tokenizer("a photo of a building", return_tensors="pt")
text_features = model.get_text_features(**inputs)

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

inputs = processor(images=image, return_tensors="pt")

image_features = model.get_image_features(**inputs)

vision_outputs = model.vision_model(**inputs)
img_dense_embs = vision_outputs['last_hidden_state'][:, 1:, :]
image_embeds = model.visual_projection(image_embeds)

# text_outputs = model.text_model(
#             input_ids=input_ids,
#             attention_mask=attention_mask,
#             position_ids=position_ids,
#             output_attentions=output_attentions,
#             output_hidden_states=output_hidden_states,
#             return_dict=return_dict,
#         )
# text_embeds = text_outputs[1]
# text_embeds = self.text_projection(text_embeds)

# normalized features
image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
# text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)

# cosine similarity as logits
# logit_scale = self.logit_scale.exp()
# logits_per_text = torch.matmul(text_embeds, image_embeds.t()) * logit_scale
# logits_per_image = logits_per_text.t()

