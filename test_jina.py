import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model_name = "jinaai/jina-reranker-v3"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1, trust_remote_code=True)

# Try with a single text
inputs = tokenizer("Just a single text", return_tensors="pt")
outputs = model(**inputs)
print("Single text logits shape:", outputs.logits.shape)
