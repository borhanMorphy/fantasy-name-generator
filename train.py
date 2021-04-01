from typing import Tuple
import numpy as np
import torch
import pytorch_lightning as pl
import argparse

import src

def parse_arguments():
    ap = argparse.ArgumentParser()

    ap.add_argument("--batch-size", "-bs", type=int, default=32)

    ap.add_argument("--yaml-file", "-y", type=str,
        default="./configs/baseline_rnn.yaml", help="yaml file path")

    ap.add_argument("--datasets", "-ds", nargs='+', type=str, choices=src.list_datasets(),
        default=["wowdb", "dotadb", "lotrdb"], help="name of the datasets")

    ap.add_argument("--data-splits", "-dsp",
        type=lambda s: [float(i.strip()) for i in s.split(",")], default="0.8, 0.2",
        help="train and val ratios with given order, splitted with comma")

    ap.add_argument("--auto-lr", "-alr", action="store_true",
        help="if true than it will try to select learning rate automatically")

    ap.add_argument("--resume", "-r", action="store_true",
        help="if true than training will resume from checkpoint")

    ap.add_argument("--seed", "-s", type=int)

    return ap.parse_args()

def collate_fn(batch) -> Tuple[torch.Tensor, torch.Tensor]:
    inputs, targets = zip(*batch)
    inputs = torch.stack(inputs, dim=0)
    targets = torch.stack(targets, dim=0)

    return inputs, targets

def main(args):
    if args.seed:
        pl.seed_everything(args.seed)

    model, trainer = src.build_from_yaml(args.yaml_file, resume=args.resume)

    ds = torch.utils.data.ConcatDataset(
        [src.get_dataset_by_name(dataset, model.tokenizer) for dataset in args.datasets]
    )

    train_ds, val_ds = src.utils.data.random_split(ds, args.data_splits)

    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size,
        shuffle=True, num_workers=4, collate_fn=collate_fn)

    val_dl = torch.utils.data.DataLoader(val_ds, batch_size=args.batch_size,
        shuffle=False, num_workers=2, collate_fn=collate_fn)

    if (not args.resume) and args.auto_lr:
        # Run learning rate finder
        lr_finder = trainer.tuner.lr_find(model,
            train_dataloader=train_dl, val_dataloaders=[val_dl],
            min_lr=1e-5, max_lr=1e-1)

        # Plot with
        lr_finder.plot(suggest=True, show=True)

        # Pick point based on plot, or get suggestion
        new_lr = lr_finder.suggestion()
        print("learning rate suggestion: ", new_lr)
        # update hparams of the model
        model.hparams.lr = new_lr

    # training / validation loop
    trainer.fit(model, train_dataloader=train_dl, val_dataloaders=[val_dl])

if __name__ == '__main__':
    args = parse_arguments()
    main(args)