from typing import Tuple
import torch
import torch.nn as nn
import src
from src.utils.tokenizer import Tokenizer
from src.dataset import WowDataset, DotaDataset
from src.arch import NaiveRNN
import json

batch_size = 64
learning_rate = 5e-3
weight_decay = 5e-4
epochs = 400

names = []
vocab = []

with open("./data/wow_names.json", "r") as foo:
    data = json.load(foo)
    for race in data:
        for subrace in data[race]:
            for name in data[race][subrace]:
                names.append(name)


with open("./data/dota_names.txt", "r") as foo:
    names += foo.read().split("\n")

for name in names:
    for n in name.lower():
        if n not in vocab:
            vocab.append(n)

max_length = max([len(name) for name in names])

vocab = sorted(vocab)

tokenizer = Tokenizer(vocab, max_length=max_length)

ds = torch.utils.data.ConcatDataset([WowDataset(tokenizer), DotaDataset(tokenizer)])
print(len(ds))

def collate_fn(batch) -> Tuple[torch.Tensor, torch.Tensor]:
    inputs, targets = zip(*batch)
    inputs = torch.stack(inputs, dim=0)
    targets = torch.stack(targets, dim=0)
    return inputs, targets

model = NaiveRNN(vocab_size=len(tokenizer.vocab), num_classes=len(tokenizer.vocab), num_layers=2, hidden_size=32)

model.cuda()

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=collate_fn, pin_memory=True)

model.train()
for e in range(epochs):
    for inputs, targets in dl:
        inputs = inputs.cuda()
        targets = targets.cuda()

        optimizer.zero_grad()

        seq_size = inputs.size(1)
        losses = []

        logits = model.forward_train(inputs)
        loss = sum([loss_fn(logit, targets[:, i]) for i,logit in enumerate(logits)]) / batch_size
        loss.backward()
        optimizer.step()
        print(loss)

model.eval()

first_char = tokenizer.random_char_select()

t = tokenizer.tokenize(first_char).unsqueeze(0)
preds = model.forward(t.cuda())
name = first_char + tokenizer.detokenize(preds)
print(name)
torch.save(model.state_dict(), "model.pt")