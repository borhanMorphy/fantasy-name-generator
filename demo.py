import src
import pytorch_lightning as pl
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", "-ckpt", type=str, required=True)
ap.add_argument("--random-n", "-n", type=int, default=20)
args = ap.parse_args()

model = src.NameGenerator.from_pretrained(args.ckpt)

for ch in model.tokenizer.select_random_n_char(k=args.random_n):
    name = model.predict(ch)
    print(name)