import torch
from torch import nn


class TransformerEncoder(nn.Module):
    def __init__(
        self, state_dim, d_model, nhead, transformer_layers, max_seq_length, dropout
    ):
        super(TransformerEncoder, self).__init__()

        self.input_linear = nn.Linear(state_dim, d_model)
        self.pos_embedding = nn.Parameter(
            torch.zeros(1, max_seq_length, d_model), requires_grad=True
        )
        nn.init.xavier_uniform_(self.pos_embedding)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=transformer_layers
        )

    def forward(self, x):
        x = nn.functional.relu(self.input_linear(x))
        seq_len = x.size(1)
        pos_emb = self.pos_embedding[:, :seq_len, :]
        x = x + pos_emb
        x = self.transformer_encoder(x)

        return x.mean(dim=1)
