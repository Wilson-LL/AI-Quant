import torch
import torch.nn as nn

class LSTM_CondTransformer(nn.Module):
    def __init__(
        self,
        input_dim,
        lstm_hidden,
        lstm_layers,
        trans_hidden,
        trans_heads,
        trans_layers,
        trans_ff,
        dropout,
        seq_len
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0
        )

        self.trans_input_proj = nn.Linear(lstm_hidden, trans_hidden)
        self.pos_emb = nn.Parameter(torch.randn(1, seq_len, trans_hidden))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=trans_hidden,
            nhead=trans_heads,
            dim_feedforward=trans_ff,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=trans_layers)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=trans_hidden,
            num_heads=trans_heads,
            batch_first=True
        )
        self.cross_kv_proj = nn.Linear(input_dim, trans_hidden)

        self.fc = nn.Sequential(
            nn.Linear(trans_hidden, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        """
        x: (B, seq_len, input_dim)
        """
        lstm_out, _ = self.lstm(x)  # (B, seq_len, lstm_hidden)

        h = self.trans_input_proj(lstm_out) + self.pos_emb  # (B, seq_len, trans_hidden)

        # Query = h,
        # Key/Value = projected x
        kv = self.cross_kv_proj(x)  # (B, seq_len, trans_hidden)
        attn_out, _ = self.cross_attn(query=h, key=kv, value=kv)
        h = h + attn_out

        h = self.transformer(h)

        last = h[:, -1, :]
        logits = self.fc(last).squeeze(-1)
        return logits