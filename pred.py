import torch
from src.utils.tokenizer import Tokenizer
from src.arch import NaiveRNN
import pytorch_lightning as pl

pl.utilities.seed.seed_everything(42)

random_forward_size = 20

tokenizer = Tokenizer()

model = NaiveRNN(vocab_size=len(tokenizer.vocab), num_classes=len(tokenizer.vocab), num_layers=2, hidden_size=32)

model.load_state_dict(torch.load("model.pt"))

model.cuda()
model.eval()

for i in range(random_forward_size):
    #first_char = tokenizer.random_char_select()
    first_char = "i"
    t = tokenizer.tokenize(first_char)
    preds = model.predict(t.cuda())
    name = first_char + tokenizer.detokenize(preds)
    print(name)