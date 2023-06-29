# import numpy as np
# from transformers import GPT2Tokenizer
# tokenizer = GPT2Tokenizer.from_pretrained('distilgpt2')
# data = 'hello world'
# data = tokenizer.tokenize(data)
# data = tokenizer.convert_tokens_to_ids(data)
#
# token = tokenizer.encode(data)
# print(data)
# print(token)
#
# import torch

import numpy as np
from transformers import AutoTokenizer, GPT2LMHeadModel, GPT2Config

tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
tokenizer.pad_token = tokenizer.eos_token
config = GPT2Config.from_pretrained('distilgpt2')
model = GPT2LMHeadModel(config)

# inputs = tokenizer(["Hello, my dog is cute", 'Hell'], padding=True, truncation=True, return_tensors="pt")
# outputs = model(**inputs, labels=inputs["input_ids"])
# loss = outputs.loss
# logits = outputs.logits

# Example 1: Print the scores for each token generated with Greedy Search
inputs = tokenizer(["Today is"], return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=5, return_dict_in_generate=True, output_scores=True)
transition_scores = model.compute_transition_scores(
    outputs.sequences, outputs.scores, normalize_logits=True
)
# input_length is the length of the input prompt for decoder-only models, like the GPT family, and 1 for
# encoder-decoder models, like BART or T5.
input_length = 1 if model.config.is_encoder_decoder else inputs.input_ids.shape[1]
generated_tokens = outputs.sequences[:, input_length:]
for tok, score in zip(generated_tokens[0], transition_scores[0]):
    print(f"| {tok:5d} | {tokenizer.decode(tok):8s} | {score.numpy():.3f} | {np.exp(score.numpy()):.2%}")


# Example 2: Reconstruct the sequence scores from Beam Search
outputs = model.generate(
    **inputs,
    max_new_tokens=5,
    num_beams=4,
    num_return_sequences=4,
    return_dict_in_generate=True,
    output_scores=True,
)
transition_scores = model.compute_transition_scores(
    outputs.sequences, outputs.scores, outputs.beam_indices, normalize_logits=False
)
# If you sum the generated tokens' scores and apply the length penalty, you'll get the sequence scores.
# Tip: recomputing the scores is only guaranteed to match with `normalize_logits=False`. Depending on the
# use case, you might want to recompute it with `normalize_logits=True`.
output_length = input_length + np.sum(transition_scores.numpy() < 0, axis=1)
length_penalty = model.generation_config.length_penalty
reconstructed_scores = transition_scores.sum(axis=1) / (output_length**length_penalty)
print(np.allclose(outputs.sequences_scores, reconstructed_scores))
