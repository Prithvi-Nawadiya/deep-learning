from transformers import BertTokenizer, BertForSequenceClassification
import torch

# Load tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Sample data
texts = ["I love AI", "I hate bugs"]
labels = torch.tensor([1, 0])

# Tokenize
encodings = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

# Load model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# Training setup
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

# Training loop (simple)
model.train()
for epoch in range(2):
    optimizer.zero_grad()
    
    outputs = model(**encodings, labels=labels)
    loss = outputs.loss
    
    loss.backward()
    optimizer.step()
    
    print(f"Epoch {epoch+1}, Loss: {loss.item()}")

import transformers
print(transformers.__version__)
