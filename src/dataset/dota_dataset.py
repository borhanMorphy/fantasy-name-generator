from torch.utils.data import Dataset
import os

class DotaDataset(Dataset):
    def __init__(self, tokenizer, root_path: str = "./data/", transform=None):
        super().__init__()
        names = []
        with open(os.path.join(root_path, "dota_names.txt"), "r") as foo:
            names = [name.lower() for name in foo.read().split("\n")]

        self.transform = transform
        self.tokenizer = tokenizer
        self.data = []
        self.targets = []

        for name in names:
            name = name.lower()
            inputs = self.tokenizer(name, length=tokenizer.max_length)
            # shift by 1 character
            targets = inputs[1:] # ignore first input
            inputs = inputs[:-1] # ignore last input

            self.data.append(inputs)
            self.targets.append(targets)

    def __getitem__(self, idx: int):
        input = self.data[idx].clone()
        target = self.targets[idx].clone()

        if self.transform:
            input = self.transform(input)

        return input, target

    def __len__(self) -> int:
        return len(self.data)