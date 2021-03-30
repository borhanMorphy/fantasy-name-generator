from typing import Optional, Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F
import json

class NaiveGRU(nn.Module):
    def __init__(self, vocab_size: int = 34, embed_dim: int = 10, hidden_size: int = 64,
            num_classes: int = 34, padding_idx: int = 0, num_layers: int = 1):
        super().__init__()

        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)

        self.gru_block = nn.GRU(input_size=embed_dim, hidden_size=hidden_size,
            num_layers=num_layers, batch_first=True, dropout=0.2)

        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        # x: batch_size, seq_size

        vec = self.embed(x)
        # vec: batch_size, seq_size, embed_dim

        outs, _ = self.gru_block(vec)
        # outs : batch_size, seq_size, hidden_size
        # h : num_layers, batch_size, hidden_size

        logits = []
        for i in range(outs.size(1)):
            logit = self.fc(outs[:, i, :])
            # logit : batch_size, num_classes
            logits.append(logit)
        return logits

    def predict(self, x: torch.Tensor, h: Optional[torch.Tensor] = None,
            max_length: int = 20) -> torch.Tensor:
        # x: batch_size, seq_size
        vec = self.embed(x)
        _, h = self.gru_block(vec, h)

        # get only last hidden information
        h = h[:, [-1], :]

        # get only last sequance of the input
        pred = x[:, [-1]]

        preds = []
        for _ in range(max_length):
            vec = self.embed(pred)
            # vec: batch_size, seq_size:1, embed_dim
            out, h = self.gru_block(vec, h)
            # out: batch_size, seq_size:1, hidden_size
            # h: num_layers, seq_size:1, hidden_size
            logit = self.fc(torch.flatten(out, start_dim=1))
            # logit: batch_size, num_classes
            score = F.softmax(logit, dim=1)
            # score: batch_size, num_classes

            # select sample from weighted distribution
            pred = torch.multinomial(score, 1)
            # pred: batch_size, 1
            preds.append(pred)

        return torch.cat(preds, dim=0)