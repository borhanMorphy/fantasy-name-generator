from torch.utils.data import Dataset
import json
import os

class WowDataset(Dataset):
    def __init__(self, tokenizer, root_path: str = "./data/", transform=None):
        super().__init__()
        names = []
        with open(os.path.join(root_path, "wow_names.json"), "r") as foo:
            data = json.load(foo)
            for race in data:
                for subrace in data[race]:
                    for name in data[race][subrace]:
                        names.append(name)

        self.transform = transform
        self.tokenizer = tokenizer
        self.data = []
        self.targets = []

        for name in names:
            name = name.lower()
            inputs = self.tokenizer(name, length=tokenizer.max_length)
            # shift by 1 character
            targets = self.tokenizer(name[1:], length=tokenizer.max_length, add_end_token=True)
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