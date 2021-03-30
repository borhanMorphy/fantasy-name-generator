import src


model = src.NameGenerator.from_pretrained("./checkpoints/NaiveRNN_best.ckpt")
model.eval()

for _ in range(10):
    print(model.predict("j"))