import torch

from transformers import AutoTokenizer, AutoModelForMaskedLM

# load bert model
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForMaskedLM.from_pretrained(model_name, num_labels=2)

# load dataset : todo
lr = 1e-5
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

def train_step(lr, batch_size, num_epochs, dataloader):
    for batch in dataloader:
        inputs = tokenizer(batch['text'], padding=True, truncation=True, return_tensors="pt")
        labels = torch.tensor(batch['label'])
        outputs = model(**inputs, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    return model

def eval_step(dataloader):
    total_loss = 0
    for batch in dataloader:
        inputs = tokenizer(batch['text'], padding=True, truncation=True, return_tensors="pt")
        labels = torch.tensor(batch['label'])
        outputs = model(**inputs, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()
    
    return total_loss / len(dataloader)

def train(lr, batch_size, num_epochs, train_dataloader, val_dataloader):
    for epoch in range(num_epochs):
        train_step(lr, batch_size, num_epochs, train_dataloader)
        eval_step(val_dataloader)




