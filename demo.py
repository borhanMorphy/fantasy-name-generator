import src
import pytorch_lightning as pl

model = src.NameGenerator.from_pretrained("./checkpoints/NaiveRNN_best.ckpt")
model.eval()

for _ in range(20):
    print(model.predict("i"))