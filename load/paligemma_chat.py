from transformers import (
    AutoProcessor,
    PaliGemmaForConditionalGeneration,
    GenerationConfig,
)
from PIL import Image
import requests
import torch

model_id = "../models/paligemma-3b-mix-448"

url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg?download=true"
image = Image.open(requests.get(url, stream=True).raw)

model = PaliGemmaForConditionalGeneration.from_pretrained(model_id).eval()
processor = AutoProcessor.from_pretrained(model_id)

# Instruct the model to create a caption in Spanish
prompt = "caption es"
model_inputs = processor(text=prompt, images=image, return_tensors="pt")
print(model_inputs.keys())


input_len = model_inputs["input_ids"].shape[-1]

generation_config = GenerationConfig(
    max_new_tokens=1024,
    do_sample=True,
    num_beams=1,
    temperature=0.8,
    top_k=40,
    top_p=0.8,
    eos_token_id=[processor.tokenizer.eos_token_id],
)

with torch.inference_mode():
    generation = model.generate(**model_inputs, generation_config=generation_config)

generation = generation[0][input_len:]
decoded = processor.decode(generation, skip_special_tokens=True)
print(decoded)
