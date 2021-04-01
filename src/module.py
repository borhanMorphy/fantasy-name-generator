import os
from typing import Dict, Union, List, Optional
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from . import api
from .utils.tokenizer import Tokenizer

# TODO pydoc

class NameGenerator(pl.LightningModule):

    def __init__(self, arch: nn.Module = None, hparams: Dict = {}):
        super().__init__()
        # save hyper parameters
        self.save_hyperparameters(hparams)

        # set architecture
        self.__arch = arch

        # initialize loss
        self.__loss_fn = api.get_loss_by_name(
            hparams.get("loss", {"id": "CE"}).get("id")
        )

        # TODO make it parametric
        self.tokenizer = Tokenizer()

    def set_loss_fn(self, loss: Union[str, nn.Module]):
        assert isinstance(loss, (str, nn.Module)), "loss must be eather string or nn.Module but found {}".format(type(loss))
        if isinstance(loss, str):
            loss = api.get_loss_by_name(loss)
        self.__loss_fn = loss

    def forward(self, *args, **kwargs):
        return self.__arch.forward(*args, **kwargs)

    @torch.no_grad()
    def generate(self) -> str:
        word = self.tokenizer.get_start_token()

        tokenized_word = self.tokenizer.tokenize(word).to(self.device)

        preds = self.__arch.predict(tokenized_word.unsqueeze(0))

        return self.tokenizer.detokenize(preds.squeeze(0))

    def on_train_epoch_start(self):
        # TODO log hyperparameters/lr
        pass

    def training_step(self, batch, batch_idx):
        batch, targets = batch

        logits = self.forward(batch)

        losses = [self.__loss_fn(logit, targets[:, i]) for i, logit in enumerate(logits)]

        mean_loss = sum(losses) / len(losses)

        return mean_loss

    def on_train_epoch_end(self, outputs):
        losses = [output["minimize"] for output in outputs[0][0]]
        mean_loss = sum(losses) / len(losses)
        self.log("loss/train", mean_loss.item())

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        batch, targets = batch

        logits = self.forward(batch)

        losses = [self.__loss_fn(logit, targets[:, i]) for i, logit in enumerate(logits)]

        mean_loss = sum(losses) / len(losses)

        return mean_loss.item()

    def validation_epoch_end(self, val_outputs: List):
        loss = sum(val_outputs)/len(val_outputs)

        self.log('loss/val', loss)

    def configure_optimizers(self):

        # initialize optimizer
        optimizer_cfg = self.hparams.get("optimizer", {"id": "adam", "lr": 1e-3, "weight_decay": 0}).copy()
        optimizer_name = optimizer_cfg.pop("id")
        optimizer_cfg.update({"lr": self.hparams.lr})
        optimizer = api.get_optimizer_by_name(
            self.parameters(), optimizer_name, **optimizer_cfg
        )

        # initialize scheduler
        scheduler_cfg = self.hparams.get("scheduler", {"id": None}).copy()
        scheduler = None
        if ("id" in scheduler_cfg) and (scheduler_cfg["id"] is not None):
            scheduler_name = scheduler_cfg.pop("id")
            scheduler = api.get_scheduler_by_name(
                optimizer, scheduler_name, **scheduler_cfg)

        if scheduler is None:
            return optimizer
        else:
            return [optimizer], [scheduler]

    @classmethod
    def build(cls, arch: Union[str, nn.Module], hparams: Dict) -> pl.LightningModule:
        assert isinstance(arch, (str, nn.Module)), "architecture must be eather string or nn.Module but found {}".format(type(arch))
        if isinstance(arch, str):
            arch = api.get_arch_by_name(arch, **hparams.get("kwargs", {}))
        return cls(arch=arch, hparams=hparams)

    @classmethod
    def from_pretrained(cls, ckpt_path: str) -> pl.LightningModule:
        # TODO assert
        ckpt = torch.load(ckpt_path, map_location='cpu')
        state_dict = ckpt["state_dict"]
        hparams = ckpt["hyper_parameters"]

        arch = os.path.basename(ckpt_path).split("_")[0]

        arch = api.get_arch_by_name(arch, **hparams.get("kwargs", {}))

        model = cls(arch=arch, hparams=hparams)

        model.load_state_dict(state_dict)

        model.eval()

        return model